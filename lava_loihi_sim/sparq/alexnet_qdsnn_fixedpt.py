"""
AlexNet QDSNN on Loihi-2: fixed-point PATH DECOMPOSITION (torch replica).

Loihi runs a static graph -> simulate 3 sub-networks and compose cost by the
measured off-chip exit distribution (plan §4). Each sample is routed by the OFF-CHIP
gate (fake-quant branchy model at a given Cfg), then simulated through ONLY its
exit's sub-network in the fixed-point Loihi model (same PyLifModelBitAcc as the QSNN
sim; conv=real convolution, MaxPool=OR, per-tensor vth here).

Paths (branchy_quantized_snn.py):
  exit1: conv1->lif1->pool1->conv2->lif2 -> exit1_head(AvgPool4 + BN1d + Linear)
  exit2: ...->pool2->conv3->lif3->conv4->lif4 -> exit2_head
  exit3: full backbone ->pool5->fc1->lif6->fc2->lif7->fc3 (readout)
Exit/final logits accumulate with decay d=0.9 each tick. Backbone int8 from
convert() (clean checkpoint, per-tensor); exit heads are the model's own FP32
modules (heads are FP32 in torch, int8 on-chip; LeNet showed FP32==int8 for heads).

Usage (from AlexNet/):
  PYTHONPATH=. ../venv/Scripts/python.exe \
    ../lava_loihi_sim/sparq/alexnet_qdsnn_fixedpt.py --n 100 --cfg 0.5 0.5 --delay
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.ao.quantization as tq
import torch.ao.nn.quantized as nnq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alexnet_loihi_fixedpt as afp
from alexnet_loihi_fixedpt import Stage         # reuse fixed-point stage

DECAY = 0.9


def op_from_fq(mod, kind):
    """Op-dict from a FAKE-QUANT intrinsic (ConvBn2d/LinearBn1d, BN live) or a plain
    QAT Linear (fc3, no BN). We hand-roll int8 by folding BN into the weight and
    quantizing per-tensor by the weight observer's scale -- because convert() COLLAPSES
    the QDSNN (per load_qdsnn: '~65% -> ~10%'); this reproduces the effective weights
    the fake-quant forward (which scores ~67%) actually uses."""
    w = mod.weight.detach().double()
    bn = getattr(mod, 'bn', None)
    if bn is not None:                                     # fold BN affine into weight
        sd = (bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)).double()
        shp = (-1, 1, 1, 1) if kind == 'conv' else (-1, 1)
        w_fold = w * sd.view(shp)
        b0 = mod.bias.detach().double() if mod.bias is not None else torch.zeros(w.shape[0]).double()
        b_fold = (b0 - bn.running_mean.detach().double()) * sd + bn.bias.detach().double()
    else:                                                 # fc3: no BN
        w_fold = w
        b_fold = (mod.bias.detach().double() if mod.bias is not None
                  else torch.zeros(w.shape[0]).double())
    # per-tensor symmetric int8 by the observer's scale (fallback max/127)
    wfq = getattr(mod, 'weight_fake_quant', None)
    scale = None
    if wfq is not None and hasattr(wfq, 'scale') and torch.isfinite(wfq.scale).all() and float(wfq.scale.reshape(-1)[0]) > 0:
        scale = float(wfq.scale.reshape(-1)[0])
    if scale is None or not np.isfinite(scale):
        scale = float(w_fold.abs().max()) / 127.0
    W_int = torch.round(w_fold / scale).clamp(-127, 127).cpu().numpy().astype(np.int32)
    d = dict(kind=kind, W=W_int, scale=np.full(W_int.shape[0], scale, np.float64),
             bias=b_fold.cpu().numpy().astype(np.float64))
    if kind == 'conv':
        d.update(stride=int(mod.stride[0]), pad=int(mod.padding[0]))
    return d


def MP(k=2, s=2):
    return dict(kind='maxpool', k=k, s=s)


def build_paths(fq, dev):
    """Backbone int8 hand-rolled from the FAKE-QUANT model (fq). Returns
    {1:(stages,head_name), 2:(...), 3:(stages,None)}."""
    c = fq
    CV = lambda m: op_from_fq(m, 'conv')
    LN = lambda m: op_from_fq(m, 'linear')
    s_conv1 = lambda: Stage([CV(c.conv1), dict(kind='lif'), MP()], dev)
    s_conv2 = lambda pool: Stage([CV(c.conv2), dict(kind='lif')] + ([MP()] if pool else []), dev)
    s_conv3 = lambda: Stage([CV(c.conv3), dict(kind='lif')], dev)
    s_conv4 = lambda: Stage([CV(c.conv4), dict(kind='lif')], dev)
    p1 = [s_conv1(), s_conv2(False)]
    p2 = [s_conv1(), s_conv2(True), s_conv3(), s_conv4()]
    p3 = [s_conv1(), s_conv2(True), s_conv3(), s_conv4(),
          Stage([CV(c.conv5), dict(kind='lif'), MP(), dict(kind='flatten')], dev),
          Stage([LN(c.fc1), dict(kind='lif')], dev),
          Stage([LN(c.fc2), dict(kind='lif')], dev),
          Stage([LN(c.fc3)], dev)]                          # readout (no lif)
    return {1: (p1, 'exit1'), 2: (p2, 'exit2'), 3: (p3, None)}


def run_path(stages, head_mod, spike_train, delay):
    """Fixed-point forward through one path; accumulate exit/final logits (decay 0.9).
    head_mod: FP32 exit-head module applied to the last LIF stage's spikes each tick
    (paths 1,2); None -> the last stage is the fc3 readout (path 3)."""
    T, B = spike_train.shape[0], spike_train.shape[1]
    for st in stages:
        st.V = None
    n = len(stages)
    ac_tot = 0.0
    nclass = 10                          # class logits (head output / fc3), not stage width
    acc = torch.zeros(B, nclass, dtype=torch.float64, device=stages[0].dev)
    t_run = (T + n + 3) if delay else T
    prev_out = [None] * n
    prev_in = None
    for t in range(t_run):
        xt = spike_train[t] if t < T else torch.zeros_like(spike_train[0])
        new = [None] * n
        for k, st in enumerate(stages):
            src = (prev_in if k == 0 else prev_out[k - 1]) if delay else \
                  (xt if k == 0 else new[k - 1])
            if src is None:
                continue
            out, rinc, ac = st.forward(src)
            ac_tot += ac
            if out is not None:
                new[k] = out
            if rinc is not None:                 # path-3 readout stage
                acc = acc * DECAY + rinc
        # exit head hangs off the last LIF stage (+1 tick like a real layer)
        if head_mod is not None:
            src = prev_out[n - 1] if delay else new[n - 1]
            if src is not None:
                acc = acc * DECAY + head_mod(src.float()).double()
        prev_out = new
        prev_in = xt
    return acc, ac_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--cfg', type=float, nargs=2, default=[0.5, 0.5],
                    help='exit thresholds (exit1, exit2)')
    ap.add_argument('--delay', action='store_true')
    ap.add_argument('--reset', choices=['hard', 'subtract'], default='hard')
    ap.add_argument('--ckpt', default='results/branchy_quantized_snn_qat_clean.pth')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from config import config
    from models.branchy_quantized_snn import BranchyQuantizedSNN, reset_exit_temperatures
    from models.quantized_snn_alexnet import encode_poisson
    T = config['num_steps']
    torch.backends.quantized.engine = 'onednn'
    afp.RESET = args.reset

    # fake-quant model: OFF-CHIP gate (exit routing) + FP32 exit-head modules
    fq = BranchyQuantizedSNN(num_classes=10, time_window=T, beta=config['beta'])
    fq.train(); fq.fuse_model()
    fq.qconfig = tq.QConfig(activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
                            weight=tq.MinMaxObserver.with_args(dtype=torch.qint8,
                                                               qscheme=torch.per_tensor_symmetric))
    tq.prepare_qat(fq, inplace=True)
    fq.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    reset_exit_temperatures(fq)
    fq.prepare_for_inference(confidence_thresholds=list(args.cfg))
    fq.eval().to(args.device)

    # backbone int8 hand-rolled from the fake-quant model (convert() collapses the QDSNN)
    paths = build_paths(fq, args.device)
    print(f"[qdsnn] Cfg {args.cfg}  delay={'ON' if args.delay else 'OFF'}  T={T}  dev={args.device}")

    import torchvision, torchvision.transforms as TT
    tfm = TT.Compose([TT.ToTensor(), TT.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2470, 0.2435, 0.2616))])
    ds = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))

    correct = 0; ref_correct = 0
    dist = {1: 0, 2: 0, 3: 0}
    ac_by = {1: 0.0, 2: 0.0, 3: 0.0}
    ok_by = {1: 0, 2: 0, 3: 0}
    with torch.no_grad():
        for i in range(N):
            img, lab = ds[i]
            x = img.unsqueeze(0).to(args.device)
            eo, fo, ep = fq(x)                      # off-chip gate + composed prediction
            ep = int(ep.item()) if torch.is_tensor(ep) else int(ep)
            ref_pred = (eo[0] if ep == 1 else eo[1] if ep == 2 else fo).argmax(1).item()
            ref_correct += int(ref_pred == lab)     # off-chip Ref, SAME sample
            spk = encode_poisson(x, T).to(args.device)
            stages, head_name = paths[ep]
            head_mod = getattr(fq, head_name) if head_name else None
            logits, ac = run_path(stages, head_mod, spk, args.delay)
            pred = int(logits.argmax(1).item())
            correct += (pred == lab); dist[ep] += 1
            ac_by[ep] += ac; ok_by[ep] += (pred == lab)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{N}  Loihi={correct/(i+1)*100:.2f}%  ref={ref_correct/(i+1)*100:.2f}%  dist={dist}")
    print(f"\n[qdsnn] N={N}  Cfg {args.cfg}  Loihi composed={correct/N*100:.2f}%  "
          f"REF(off-chip,same samples)={ref_correct/N*100:.2f}%  delay={'ON' if args.delay else 'OFF'}")
    tot = sum(dist.values())
    print(f"[qdsnn] exit dist: " + " ".join(f"e{k}={dist[k]/tot*100:.1f}%" for k in (1, 2, 3)))
    print(f"[qdsnn] mean ACs/sample = {sum(ac_by.values())/N:.4e}")
    for k in (1, 2, 3):
        if dist[k]:
            print(f"   exit{k}: n={dist[k]:3d} acc={ok_by[k]/dist[k]*100:5.1f}% "
                  f"meanACs={ac_by[k]/dist[k]:.3e}")


if __name__ == '__main__':
    main()
