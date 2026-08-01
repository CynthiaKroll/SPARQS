"""
Export the SPARQ FP32 LeNet SNN (`lenet_snn`) for the Lava/Loihi leg.
MAIN env (py3.14 + torch). Conv/AvgPool unrolled to Dense -- see lenet_common.py.

FP32 -> deployable: fold BN into conv/linear, then INT8 per-tensor symmetric
(same scheme as the QAT pair, so the two are comparable).

Topology (AvgPool folded into the FOLLOWING layer):
  L0 conv1            784  -> 4704   + LIF
  L1 pool1->conv2     4704 -> 1600   + LIF
  L2 pool2->fc1       1600 -> 120    + LIF
  L3 fc2              120  -> 84     + LIF
  L4 fc3              84   -> 10     + LIF   <-- final LIF exists here
=> readout = 'spike_count' (rate-coded), n_lif == n_layers, T=32.

Usage (from LeNet-Clean):
  PYTHONPATH=<LeNet-Clean> ../venv/Scripts/python.exe \
      ../lava_loihi_sim/sparq/export_lenet_snn.py --n 100 --T 32 \
      --out ../lava_loihi_sim/exports/lenet_snn.npz
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lenet_common import (fold_bn_conv, fold_bn_linear, conv2d_to_dense,
                          avgpool_matrix, pack_layer)


def golden_lif_sim(x_flat, T, layers, beta, thrs):
    """DIRECT (constant-current) encoding: the SAME analog input is applied every
    tick -- this model was trained that way (98.5% direct vs 21.5% Poisson).
    All layers have a LIF; prediction = output spike COUNTS over T."""
    W = [torch.from_numpy(L['w_int'].astype(np.float32)) * L['scale'] for L in layers]
    Bs = [torch.from_numpy(L['bias']) for L in layers]
    B = x_flat.shape[0]
    n = len(W)
    mems = [torch.zeros(B, W[k].shape[0]) for k in range(n)]
    fire = np.zeros(n); den = np.zeros(n)
    counts = torch.zeros(B, W[-1].shape[0])
    with torch.no_grad():
        for t in range(T):
            h = x_flat                      # constant current, every tick
            for k in range(n):
                h = F.linear(h, W[k], Bs[k])
                mems[k] = beta * mems[k] + h
                s = (mems[k] >= thrs[k]).float()
                mems[k] = mems[k] - s * thrs[k]
                fire[k] += s.sum().item(); den[k] += s.numel()
                h = s
            counts += h
    return counts, counts.argmax(1), fire / np.maximum(den, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/lenet_snn.pth')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--out', default='../lava_loihi_sim/exports/lenet_snn.npz')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from config import config
    from evaluate import load_model
    from models.quantized_snn_lenet import encode_poisson

    model = load_model('snn_lenet', args.ckpt, 'cpu', time_steps_from_args=args.T)
    model.eval().cpu()

    feats = list(model.features)
    clf = list(model.classifier)
    conv1, bn1 = feats[0], feats[1]
    conv2, bn2 = feats[4], feats[5]
    fc1, bnf1 = clf[1], clf[2]
    fc2, bnf2 = clf[4], clf[5]
    fc3 = clf[7]
    lif_mods = [m for m in list(model.features) + list(model.classifier)
                if isinstance(m, snn.Leaky)]
    thrs = [float(m.threshold) for m in lif_mods]
    beta = float(lif_mods[0].beta)

    # --- fold BN, unroll conv, fold pooling forward ---
    W1, b1 = fold_bn_conv(conv1, bn1)
    M0, bias0, shape1 = conv2d_to_dense(W1, b1, (1, 28, 28), 1, 2)

    P1, pooled1 = avgpool_matrix(shape1, 2, 2)
    W2, b2 = fold_bn_conv(conv2, bn2)
    Mc2, bias2, shape2 = conv2d_to_dense(W2, b2, pooled1, 1, 0)
    M1 = Mc2 @ P1

    P2, _ = avgpool_matrix(shape2, 2, 2)
    W3, b3 = fold_bn_linear(fc1, bnf1)
    M2 = W3.numpy().astype(np.float64) @ P2

    W4, b4 = fold_bn_linear(fc2, bnf2)
    W5 = fc3.weight.data.clone()
    b5 = fc3.bias.data.clone() if fc3.bias is not None else torch.zeros(W5.shape[0])

    layers = [
        pack_layer('conv1', M0, bias0),
        pack_layer('pool1+conv2', M1, bias2),
        pack_layer('pool2+fc1', M2, b3.numpy().astype(np.float64)),
        pack_layer('fc2', W4.numpy().astype(np.float64), b4.numpy()),
        pack_layer('fc3', W5.numpy().astype(np.float64), b5.numpy()),
    ]
    n_lif = len(layers)          # every layer has a LIF in this model
    print(f"[export] {len(layers)} dense layers, {n_lif} LIF (final LIF present), "
          f"beta={beta:.3f}, thresholds={thrs}, T={args.T}")
    for L in layers:
        print(f"   {L['name']:<12} {L['w_int'].shape} scale={L['scale']:.4g} "
              f"fanout={L['fanout']:.1f}")

    import torchvision
    from torchvision import transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    images = torch.stack([ds[i][0] for i in range(N)])
    labels = torch.tensor([ds[i][1] for i in range(N)])
    x_flat = images.reshape(N, -1)              # [N,784] analog, DIRECT encoding

    with torch.no_grad():
        out = model(images)                     # raw batch -> model repeat-encodes
        fp32_pred = out.sum(0).argmax(1)
    fp32_acc = (fp32_pred == labels).float().mean().item()

    g_counts, g_pred, g_fire = golden_lif_sim(x_flat, args.T, layers, beta, thrs[:n_lif])
    golden_acc = (g_pred == labels).float().mean().item()
    print(f"[export] FP32 acc={fp32_acc*100:.2f}%  golden INT8(dense) acc={golden_acc*100:.2f}%  "
          f"fire={[f'{r*100:.1f}%' for r in g_fire]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = dict(
        n_layers=len(layers), layer_names=np.array([L['name'] for L in layers]),
        beta=beta, threshold=float(thrs[0]),
        thresholds=np.array(thrs[:n_lif], np.float32),
        n_lif=n_lif, T=args.T, readout='spike_count',
        # layer 0 arrives as neuron bias, NOT spikes -> it performs ZERO synaptic
        # ops on-chip. Labelled explicitly: this is an encoding artefact, so
        # lenet_snn's energy is NOT directly comparable to the Poisson-input QSNN.
        ac_fanout=np.array([0.0] + [L['fanout'] for L in layers[1:]], np.float64),
        input_mode='bias',   # layer 0 is injected as a constant LIF bias, not spikes
        # layer-0 current from the INT8 weights: I = (w_int0*scale0) @ x + bias0
        input_current=((torch.from_numpy(layers[0]['w_int'].astype(np.float32))
                        * layers[0]['scale']) @ x_flat.T
                       + torch.from_numpy(layers[0]['bias']).unsqueeze(1)
                       ).T.contiguous().numpy().astype(np.float32),
        spikes=np.zeros((N, args.T, 1), np.uint8),   # placeholder; unused in bias mode
        labels=labels.numpy().astype(np.int64),
        golden_logits=g_counts.numpy().astype(np.float32),
        golden_pred=g_pred.numpy().astype(np.int64),
        golden_fire=g_fire.astype(np.float32),
        int8_pred=fp32_pred.numpy().astype(np.int64),
        int8_acc=np.float32(fp32_acc), golden_acc=np.float32(golden_acc),
    )
    for k, L in enumerate(layers):
        blob[f'w{k}'] = L['w_int']; blob[f'scale{k}'] = np.float32(L['scale'])
        blob[f'bias{k}'] = L['bias']
    np.savez_compressed(args.out, **blob)
    print(f"[export] wrote {args.out} (spikes {blob['spikes'].shape}, N={N})")


if __name__ == '__main__':
    main()
