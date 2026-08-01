"""
Export the SPARQ MLP-QSNN (INT8) to a neutral .npz for the Lava/Loihi leg.

Runs in the MAIN env (Python 3.14 + torch quantization). Produces a single
.npz that the py3.10 `venv-lava` env consumes -- so the fragile quantized
`.pth` is never loaded across torch versions.

What it writes (per the Loihi plan, [[loihi-implementation]]):
  - w{k}      int8 weights for Linear k, shape [out, in]     (k=0..n_layers-1)
  - scale{k}  per-tensor weight scale for Linear k           (float)
  - bias{k}   folded FP32 bias for Linear k, shape [out]
  - beta, threshold, n_lif   snntorch LIF params (last Linear has NO LIF)
  - T                        timesteps used for encoding/sim
  - spikes    [N, T, in0] uint8   EXACT Poisson input spike trains (so Lava
              reproduces the same inputs; encode_poisson_rate is random)
  - labels    [N] int
  - golden_logits [N,10], golden_pred [N], golden_fire [n_lif]   real-LIF
              reference (dequantized int8 weights, subtract reset) -- the
              bridge Lava must match
  - int8_pred [N], int8_acc, golden_acc

Usage (from the 5-layer-MLP dir):
  PYTHONPATH=<5-layer-MLP> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/export_mlp_qsnn.py --n 500 --T 4 --out ../lava_loihi_sim/exports/mlp_qsnn.npz
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import snntorch as snn


def extract_int8_linears(model):
    """Return ordered list of quantized Linear layers as dicts."""
    layers = []
    for seqname in ('features_seq', 'classifier_seq'):
        seq = getattr(model, seqname)
        for i, m in enumerate(seq):
            if m.__class__.__name__.startswith('Linear') and hasattr(m, 'weight') and callable(m.weight):
                try:
                    w = m.weight()
                except Exception:
                    continue
                if getattr(w, 'is_quantized', False):
                    b = m.bias()
                    layers.append(dict(
                        name=f'{seqname}.{i}',
                        w_int=w.int_repr().numpy().astype(np.int8),
                        scale=float(w.q_scale()),
                        bias=(b.detach().numpy().astype(np.float32) if b is not None
                              else np.zeros(w.shape[0], np.float32)),
                    ))
    return layers


def golden_lif_sim(spikes, layers, beta, threshold, n_lif):
    """Real-valued LIF reference using dequantized int8 weights, subtract reset.

    spikes: [T, B, in0] float/binary. Returns (logits[B,10], pred[B], fire_rate[n_lif]).
    """
    W = [torch.from_numpy(L['w_int'].astype(np.float32)) * L['scale'] for L in layers]
    Bs = [torch.from_numpy(L['bias']) for L in layers]
    T, B = spikes.shape[0], spikes.shape[1]
    mems = [torch.zeros(B, W[k].shape[0]) for k in range(n_lif)]
    fire = np.zeros(n_lif); denom = np.zeros(n_lif)
    logits = torch.zeros(B, W[-1].shape[0])
    with torch.no_grad():
        for t in range(T):
            h = spikes[t]
            for k in range(len(W)):
                h = F.linear(h, W[k], Bs[k])
                if k < n_lif:
                    mems[k] = beta * mems[k] + h
                    spk = (mems[k] >= threshold).float()
                    mems[k] = mems[k] - spk * threshold
                    fire[k] += spk.sum().item(); denom[k] += spk.numel()
                    h = spk
                else:
                    logits += h
    return logits, logits.argmax(1), fire / np.maximum(denom, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/quantized_snn_mlp.pth')
    ap.add_argument('--n', type=int, default=500, help='num test samples')
    ap.add_argument('--T', type=int, default=4, help='timesteps (QSNN trained at 4)')
    ap.add_argument('--out', default='../lava_loihi_sim/exports/mlp_qsnn.npz')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from config import config
    from evaluate import load_model
    from utils import encode_poisson_rate

    model = load_model('quantized_snn_mlp', args.ckpt, 'cpu', time_steps=args.T)
    model.eval()

    layers = extract_int8_linears(model)
    lifs = [m for m in model.modules() if isinstance(m, snn.Leaky)]
    beta = float(lifs[0].beta); threshold = float(lifs[0].threshold); n_lif = len(lifs)
    print(f"[export] {len(layers)} int8 Linears, {n_lif} LIF, beta={beta:.3f}, "
          f"thr={threshold:.3f}, reset={lifs[0].reset_mechanism}, T={args.T}")

    # --- test data (num_workers=0: Windows workers crash when spawned here) ---
    import torchvision
    from torchvision import transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    images = torch.stack([ds[i][0] for i in range(N)])
    labels = torch.tensor([ds[i][1] for i in range(N)])

    # --- exact Poisson spike trains (reused verbatim in Lava) ---
    spikes = encode_poisson_rate(images, args.T)          # [T, N, 784]

    # --- true int8 forward ---
    with torch.no_grad():
        int8_out = model(spikes)                          # [T, N, 10]
        int8_pred = int8_out.sum(0).argmax(1)
    int8_acc = (int8_pred == labels).float().mean().item()

    # --- golden real-LIF reference ---
    g_logits, g_pred, g_fire = golden_lif_sim(spikes, layers, beta, threshold, n_lif)
    golden_acc = (g_pred == labels).float().mean().item()
    print(f"[export] int8 acc={int8_acc*100:.2f}%  golden acc={golden_acc*100:.2f}%  "
          f"fire={[f'{r*100:.1f}%' for r in g_fire]}")

    # --- pack npz ---
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = dict(
        n_layers=len(layers),
        layer_names=np.array([L['name'] for L in layers]),
        beta=beta, threshold=threshold, n_lif=n_lif, T=args.T,
        spikes=spikes.permute(1, 0, 2).contiguous().numpy().astype(np.uint8),  # [N,T,784]
        labels=labels.numpy().astype(np.int64),
        golden_logits=g_logits.numpy().astype(np.float32),
        golden_pred=g_pred.numpy().astype(np.int64),
        golden_fire=g_fire.astype(np.float32),
        int8_pred=int8_pred.numpy().astype(np.int64),
        int8_acc=np.float32(int8_acc), golden_acc=np.float32(golden_acc),
    )
    for k, L in enumerate(layers):
        blob[f'w{k}'] = L['w_int']
        blob[f'scale{k}'] = np.float32(L['scale'])
        blob[f'bias{k}'] = L['bias']
    np.savez_compressed(args.out, **blob)
    print(f"[export] wrote {args.out}  (spikes {blob['spikes'].shape}, N={N})")


if __name__ == '__main__':
    main()
