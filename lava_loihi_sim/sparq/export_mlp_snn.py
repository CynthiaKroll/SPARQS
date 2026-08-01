"""
Export the SPARQ FP32 SNN-MLP (`snn_mlp`) to the neutral .npz for the Lava leg.

Runs in the MAIN env (py3.14 + torch). Unlike the QSNN (already INT8 + BN folded
by torch's convert), this model is FP32 with LIVE BatchNorm, so here we:
  1. fold BN into the preceding Linear   (W' = W*s, b' = (b-mean)*s + beta,
                                          s = gamma/sqrt(var+eps))
  2. quantize W' to INT8 per-tensor symmetric (scale = max|W'|/127)
     -- same scheme the QSNN's QAT used, so the two are comparable.

Topology differs from the QSNN in one important way:
  QSNN : 8 Linear, 7 LIF, readout = accumulate final Linear output.
  SNN  : 8 Linear, 8 LIF -- there IS a LIF after the final Linear, and the model
         classifies by OUTPUT SPIKE COUNTS (rate-coded readout, see mlp_snn.py
         line ~225: it records final_layer_output[0], the spike train).
So the npz carries `readout='spike_count'` and n_lif == n_layers.

NOTE: this model was trained at T=32 (vs the QSNN's T=4). T=32 means ~8x the
spikes and t_run = 32 + 8 + pad ticks, so the Lava sim is MUCH slower per sample
-- use a smaller --n for this one.

Usage (from the 5-layer-MLP dir):
  PYTHONPATH=<5-layer-MLP> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/export_mlp_snn.py --n 100 --T 32 \
      --out ../lava_loihi_sim/exports/mlp_snn.npz
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn


def fold_bn_into_linear(lin: nn.Linear, bn: nn.BatchNorm1d):
    """Return (W_folded, b_folded) absorbing BatchNorm into the Linear."""
    W = lin.weight.data.clone()
    b = lin.bias.data.clone() if lin.bias is not None else torch.zeros(W.shape[0])
    gamma = bn.weight.data.clone()
    beta = bn.bias.data.clone()
    mean = bn.running_mean.data.clone()
    var = bn.running_var.data.clone()
    s = gamma / torch.sqrt(var + bn.eps)
    return W * s.unsqueeze(1), (b - mean) * s + beta


def quantize_int8_per_tensor(W):
    """Symmetric per-tensor INT8. Returns (w_int8, scale)."""
    max_abs = float(W.abs().max().clamp(min=1e-12))
    scale = max_abs / 127.0
    w_int = torch.clamp(torch.round(W / scale), -127, 127)
    return w_int.numpy().astype(np.int8), scale


def golden_lif_sim(spikes, layers, beta, thresholds):
    """Real-valued LIF reference on dequantized int8 weights, subtract reset.

    ALL layers have a LIF; prediction = output spike COUNT over T (rate readout).
    Returns (counts[B,10], pred[B], fire_rate[n_lif]).
    """
    import torch.nn.functional as F
    W = [torch.from_numpy(L['w_int'].astype(np.float32)) * L['scale'] for L in layers]
    Bs = [torch.from_numpy(L['bias']) for L in layers]
    T, B = spikes.shape[0], spikes.shape[1]
    n = len(W)
    mems = [torch.zeros(B, W[k].shape[0]) for k in range(n)]
    fire = np.zeros(n); denom = np.zeros(n)
    counts = torch.zeros(B, W[-1].shape[0])
    with torch.no_grad():
        for t in range(T):
            h = spikes[t]
            for k in range(n):
                h = F.linear(h, W[k], Bs[k])
                mems[k] = beta * mems[k] + h
                spk = (mems[k] >= thresholds[k]).float()
                mems[k] = mems[k] - spk * thresholds[k]
                fire[k] += spk.sum().item(); denom[k] += spk.numel()
                h = spk
            counts += h                      # final layer spikes = class votes
    return counts, counts.argmax(1), fire / np.maximum(denom, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/snn_mlp.pth')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--T', type=int, default=32, help='snn_mlp was trained at T=32')
    ap.add_argument('--out', default='../lava_loihi_sim/exports/mlp_snn.npz')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from config import config
    from evaluate import load_model
    from utils import encode_poisson_rate

    model = load_model('snn_mlp', args.ckpt, 'cpu', time_steps=args.T)
    model.eval().cpu()

    # --- ordered (Linear, BN|None, LIF) triples, matching mlp_snn.py forward ---
    pairs = [
        ('feat_linear1', 'feat_bn1', 'feat_lif1'),
        ('feat_linear2', 'feat_bn2', 'feat_lif2'),
        ('feat_linear3', 'feat_bn3', 'feat_lif3'),
        ('feat_linear4', 'feat_bn4', 'feat_lif4'),
        ('feat_linear5', 'feat_bn5', 'feat_lif5'),
        ('class_linear1', 'class_bn1', 'class_lif1'),
        ('class_linear2', 'class_bn2', 'class_lif2'),
        ('class_linear3', None,       'class_lif3'),   # final Linear has NO BN
    ]
    layers = []; thresholds = []
    for lname, bname, sname in pairs:
        lin = getattr(model, lname)
        if bname is not None:
            Wf, bf = fold_bn_into_linear(lin, getattr(model, bname))
        else:
            Wf = lin.weight.data.clone()
            bf = lin.bias.data.clone() if lin.bias is not None else torch.zeros(Wf.shape[0])
        w_int, scale = quantize_int8_per_tensor(Wf)
        layers.append(dict(name=lname, w_int=w_int, scale=scale,
                           bias=bf.numpy().astype(np.float32)))
        thresholds.append(float(getattr(model, sname).threshold))

    beta = float(getattr(model, pairs[0][2]).beta)
    print(f"[export] {len(layers)} Linear(+BN folded)->INT8, {len(layers)} LIF, "
          f"beta={beta:.3f}, thresholds={thresholds}, T={args.T}")
    for L in layers:
        print(f"   {L['name']:<14} {L['w_int'].shape}  scale={L['scale']:.4g} "
              f"|w|max={np.abs(L['w_int']).max()}")

    # --- test data (num_workers=0: Windows workers crash when spawned here) ---
    import torchvision
    from torchvision import transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    images = torch.stack([ds[i][0] for i in range(N)])
    labels = torch.tensor([ds[i][1] for i in range(N)])

    spikes = encode_poisson_rate(images, args.T)          # [T, N, 784]

    # --- FP32 model reference ---
    with torch.no_grad():
        fp32_out = model(spikes)                          # [T, N, 10] spike train
        fp32_pred = fp32_out.sum(0).argmax(1)
    fp32_acc = (fp32_pred == labels).float().mean().item()

    # --- golden INT8 reference (what Lava must reproduce) ---
    g_counts, g_pred, g_fire = golden_lif_sim(spikes, layers, beta, thresholds)
    golden_acc = (g_pred == labels).float().mean().item()
    print(f"[export] FP32 model acc={fp32_acc*100:.2f}%  "
          f"golden INT8 acc={golden_acc*100:.2f}%  "
          f"fire={[f'{r*100:.1f}%' for r in g_fire]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = dict(
        n_layers=len(layers),
        layer_names=np.array([L['name'] for L in layers]),
        beta=beta, threshold=float(thresholds[0]),
        thresholds=np.array(thresholds, np.float32),
        n_lif=len(layers),                 # EVERY layer has a LIF here
        readout='spike_count',
        T=args.T,
        spikes=spikes.permute(1, 0, 2).contiguous().numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.int64),
        golden_logits=g_counts.numpy().astype(np.float32),
        golden_pred=g_pred.numpy().astype(np.int64),
        golden_fire=g_fire.astype(np.float32),
        int8_pred=fp32_pred.numpy().astype(np.int64),
        int8_acc=np.float32(fp32_acc), golden_acc=np.float32(golden_acc),
    )
    for k, L in enumerate(layers):
        blob[f'w{k}'] = L['w_int']
        blob[f'scale{k}'] = np.float32(L['scale'])
        blob[f'bias{k}'] = L['bias']
    np.savez_compressed(args.out, **blob)
    print(f"[export] wrote {args.out}  (spikes {blob['spikes'].shape}, N={N})")


if __name__ == '__main__':
    main()
