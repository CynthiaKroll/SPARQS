"""
Export the SPARQ LeNet QSNN (`quantized_snn_lenet_int8`) for the Lava/Loihi leg.
Runs in the MAIN env (py3.14 + torch). See lenet_common.py for why conv+pool are
unrolled into Dense matrices.

Topology (AvgPool folded into the FOLLOWING layer, so every inter-layer signal
stays binary spikes -- which is what Loihi requires):

  L0 : conv1                    (1,28,28)=784   -> (6,28,28)=4704   + LIF
  L1 : pool1 -> conv2           4704            -> (16,10,10)=1600  + LIF
  L2 : pool2 -> flatten -> fc1  1600            -> 120              + LIF
  L3 : fc2                      120             -> 84               + LIF
  L4 : fc3                      84              -> 10   (integrator readout)

Usage (from the LeNet-Clean dir):
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/export_lenet_qsnn.py --n 200 --T 4 \
      --out ../lava_loihi_sim/exports/lenet_qsnn.npz
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import snntorch as snn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lenet_common import (conv2d_to_dense, avgpool_matrix, pack_layer)


def dequant_conv(m):
    """Dequantized weight/bias of a converted INT8 Conv2d or Linear."""
    w = m.weight()
    W = (w.int_repr().float() * w.q_scale())
    b = m.bias()
    b = b.detach().float() if b is not None else torch.zeros(W.shape[0])
    return W, b


def golden_lif_sim(spikes, layers, beta, thr, n_lif):
    """Real-valued LIF reference on the dequantized dense layers (subtract reset).
    spikes: [T,B,in0]. Readout = sum of final layer output over T."""
    W = [torch.from_numpy(L['w_int'].astype(np.float32)) * L['scale'] for L in layers]
    Bs = [torch.from_numpy(L['bias']) for L in layers]
    T, B = spikes.shape[0], spikes.shape[1]
    mems = [torch.zeros(B, W[k].shape[0]) for k in range(n_lif)]
    fire = np.zeros(n_lif); den = np.zeros(n_lif)
    logits = torch.zeros(B, W[-1].shape[0])
    with torch.no_grad():
        for t in range(T):
            h = spikes[t]
            for k in range(len(W)):
                h = F.linear(h, W[k], Bs[k])
                if k < n_lif:
                    mems[k] = beta * mems[k] + h
                    s = (mems[k] >= thr).float()
                    mems[k] = mems[k] - s * thr
                    fire[k] += s.sum().item(); den[k] += s.numel()
                    h = s
                else:
                    logits += h
    return logits, logits.argmax(1), fire / np.maximum(den, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/quantized_snn_lenet_int8.pth')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--out', default='../lava_loihi_sim/exports/lenet_qsnn.npz')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--direct', action='store_true',
                    help="DIRECT/constant-current encoding instead of Poisson spikes: "
                         "layer-0 current is injected as a constant LIF bias every "
                         "tick, so the pipeline never starves. lenet_snn uses this "
                         "and survives Loihi (96 pct) where Poisson collapses (22 pct).")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    import torch.ao.quantization as tq
    from config import config
    from models.quantized_snn_lenet import (QuantizedSNNLeNet, encode_poisson,
                                            get_quant_engine)

    # NOTE: do NOT use LeNet-Clean/evaluate.py load_model here -- for
    # 'quantized_snn_lenet' it loads the INT8 checkpoint into an FP32 skeleton,
    # which FAILS and silently falls back to RANDOM WEIGHTS. Rebuild the INT8
    # skeleton the way evaluate_quantized_pair.py does, and load STRICTLY.
    model = QuantizedSNNLeNet(num_classes=config["num_classes"],
                              beta=config["snn"]["beta"], time_steps=args.T)
    state = torch.load(args.ckpt, map_location='cpu')
    torch.backends.quantized.engine = get_quant_engine()
    model.fuse_model()
    model.qconfig = tq.QConfig(
        activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=tq.MinMaxObserver.with_args(dtype=torch.qint8,
                                           qscheme=torch.per_tensor_symmetric))
    for _, m in model.named_modules():
        if isinstance(m, snn.Leaky):
            m.qconfig = None
    tq.prepare_qat(model, inplace=True)
    model.eval().cpu()
    model = tq.convert(model, inplace=False)
    model.load_state_dict(state)          # strict: mismatch must fail loudly
    model.eval().cpu()
    print(f"[export] rebuilt INT8 skeleton and strict-loaded {args.ckpt}")

    # --- collect the quantized weight layers in forward order ---
    q = []
    for seqname in ('features_seq', 'classifier_seq'):
        for m in getattr(model, seqname):
            if hasattr(m, 'weight') and callable(getattr(m, 'weight', None)):
                try:
                    w = m.weight()
                except Exception:
                    continue
                if getattr(w, 'is_quantized', False):
                    q.append(m)
    if len(q) != 5:
        raise RuntimeError(f"expected 5 quantized layers (conv1,conv2,fc1,fc2,fc3), got {len(q)}")
    conv1, conv2, fc1, fc2, fc3 = q
    print(f"[export] found {len(q)} quantized layers")

    # --- build dense layers, folding AvgPool into the following layer ---
    W1, b1 = dequant_conv(conv1)
    M0, bias0, shape1 = conv2d_to_dense(W1, b1, (1, 28, 28), stride=1, padding=2)

    P1, pooled1 = avgpool_matrix(shape1, 2, 2)                  # (6,28,28)->(6,14,14)
    W2, b2 = dequant_conv(conv2)
    Mc2, bias2, shape2 = conv2d_to_dense(W2, b2, pooled1, stride=1, padding=0)
    M1 = Mc2 @ P1                                               # fold pool1

    P2, pooled2 = avgpool_matrix(shape2, 2, 2)                  # (16,10,10)->(16,5,5)
    W3, b3 = dequant_conv(fc1)
    M2 = W3.detach().numpy().astype(np.float64) @ P2            # fold pool2
    bias3 = b3.detach().numpy().astype(np.float64)

    W4, b4 = dequant_conv(fc2)
    W5, b5 = dequant_conv(fc3)

    layers = [
        pack_layer('conv1', M0, bias0),
        pack_layer('pool1+conv2', M1, bias2),
        pack_layer('pool2+fc1', M2, bias3),
        pack_layer('fc2', W4.detach().numpy().astype(np.float64), b4.detach().numpy()),
        pack_layer('fc3', W5.detach().numpy().astype(np.float64), b5.detach().numpy()),
    ]
    lifs = [m for m in model.modules() if isinstance(m, snn.Leaky)]
    beta = float(lifs[0].beta); thr = float(lifs[0].threshold); n_lif = 4
    print(f"[export] {len(layers)} dense layers, {n_lif} LIF, beta={beta:.3f}, thr={thr:.3f}")
    for L in layers:
        print(f"   {L['name']:<12} {L['w_int'].shape}  scale={L['scale']:.4g}  "
              f"fanout={L['fanout']:.1f} (dense would be {L['w_int'].shape[0]})")

    # --- data ---
    import torchvision
    from torchvision import transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    images = torch.stack([ds[i][0] for i in range(N)])
    labels = torch.tensor([ds[i][1] for i in range(N)])

    spk = encode_poisson(images, args.T)                 # [T,N,1,28,28]
    spikes_flat = spk.reshape(args.T, N, -1)             # [T,N,784]
    x_flat = images.reshape(N, -1)                       # analog, for --direct

    # --- reference: the real INT8 model ---
    with torch.no_grad():
        out = model(spk)
        int8_pred = out.sum(0).argmax(1)
    int8_acc = (int8_pred == labels).float().mean().item()

    if args.direct:
        # constant current every tick == the noiseless mean of the Poisson code
        spikes_flat = x_flat.unsqueeze(0).repeat(args.T, 1, 1)
    g_logits, g_pred, g_fire = golden_lif_sim(spikes_flat, layers, beta, thr, n_lif)
    golden_acc = (g_pred == labels).float().mean().item()
    print(f"[export] INT8 model acc={int8_acc*100:.2f}%  "
          f"golden(dense-unrolled) acc={golden_acc*100:.2f}%  "
          f"agree={(int8_pred==g_pred).float().mean()*100:.1f}%")
    print(f"[export] firing/layer = {[f'{r*100:.1f}%' for r in g_fire]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = dict(
        n_layers=len(layers), layer_names=np.array([L['name'] for L in layers]),
        beta=beta, threshold=thr, n_lif=n_lif, T=args.T, readout='integrator',
        ac_fanout=(np.array([0.0] + [L['fanout'] for L in layers[1:]], np.float64)
                   if args.direct else
                   np.array([L['fanout'] for L in layers], np.float64)),
        input_mode=('bias' if args.direct else 'spikes'),
        input_current=(((torch.from_numpy(layers[0]['w_int'].astype(np.float32))
                         * layers[0]['scale']) @ x_flat.T
                        + torch.from_numpy(layers[0]['bias']).unsqueeze(1)
                        ).T.contiguous().numpy().astype(np.float32)
                       if args.direct else np.zeros((1, 1), np.float32)),
        spikes=(np.zeros((N, args.T, 1), np.uint8) if args.direct else
                spikes_flat.permute(1, 0, 2).contiguous().numpy().astype(np.uint8)),
        labels=labels.numpy().astype(np.int64),
        golden_logits=g_logits.numpy().astype(np.float32),
        golden_pred=g_pred.numpy().astype(np.int64),
        golden_fire=g_fire.astype(np.float32),
        int8_pred=int8_pred.numpy().astype(np.int64),
        int8_acc=np.float32(int8_acc), golden_acc=np.float32(golden_acc),
    )
    for k, L in enumerate(layers):
        blob[f'w{k}'] = L['w_int']; blob[f'scale{k}'] = np.float32(L['scale'])
        blob[f'bias{k}'] = L['bias']
    np.savez_compressed(args.out, **blob)
    print(f"[export] wrote {args.out} (spikes {blob['spikes'].shape}, N={N})")


if __name__ == '__main__':
    main()
