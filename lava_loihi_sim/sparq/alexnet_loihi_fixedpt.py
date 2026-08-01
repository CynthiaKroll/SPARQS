"""
AlexNet on Loihi-2: BIT-ACCURATE FIXED-POINT SIMULATION (torch replica).

WHY a replica instead of Lava Dense-unroll: AlexNet's conv layers unroll to
**2.9 BILLION** synapses (conv3 alone = 1.2 B) vs the MLP's 32 M / LeNet's 11 M.
A Lava Dense run at that scale is ~2.9 GB of matrices and ~18 min/sample -- hours
per model, intractable. So we run AlexNet through the SAME fixed-point Loihi-2
neuron model (`PyLifModelBitAcc`) Lava uses, implemented as a validated torch
replica (matches Lava within a few pp on MLP/LeNet -- audit §4.3), but with **conv
as an actual convolution** (torch conv2d on int8 weights -> exact integer currents),
MaxPool as a native OR of binary spikes, and per-CHANNEL vth (AlexNet uses
per-channel weight quantization, unlike LeNet/MLP per-tensor).

Report as "bit-accurate fixed-point simulation of Loihi-2 dynamics," NOT a Lava run.

Fixed-point LIF: du=4095 (u=a<<6), dv=round((1-β)·4096)=205, membrane clip ±2**23,
hard reset, per-channel vth=round(1/scale)<<6, bias folded round(64·b/scale).
Delay model: each conv/linear LAYER buffers 1 tick (Loihi dendritic delay); stage k
at tick t consumes stage k-1's output from t-1. Readout (fc3, no LIF) sums the real
logit over T (QSNN plain-sum/mean readout). ACs counted EXACTLY via conv/linear with
a weight-nonzero mask (spike-driven accumulates).

Usage (from AlexNet/):
  PYTHONPATH=. ../venv/Scripts/python.exe \
     ../lava_loihi_sim/sparq/alexnet_loihi_fixedpt.py --n 200 [--delay]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.ao.quantization as tq
import torch.ao.nn.quantized as nnq
import snntorch

BETA, THR, ASHIFT, MEMCLIP = 0.95, 1.0, 6, 2 ** 23
DV = int(round((1 - BETA) * 4096))          # 205
RESET = 'hard'                              # 'hard' (Lava stock) or 'subtract' (model-trained)


def extract_stages(conv_model):
    """Walk the converted-INT8 QSNN -> ordered op list -> STAGES (each starts at a
    conv/linear, includes its LIF + trailing maxpool/flatten). Last stage = readout."""
    ops = []
    for seq in (conv_model.features_seq, conv_model.classifier_seq):
        for m in seq:
            if isinstance(m, (nnq.Conv2d, nnq.Linear)):
                w = m.weight()
                W = w.int_repr().cpu().numpy().astype(np.int32)
                per_ch = w.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine)
                sc = (w.q_per_channel_scales().cpu().numpy() if per_ch
                      else np.full(W.shape[0], float(w.q_scale()))).astype(np.float64)
                b = (m.bias().detach().cpu().numpy() if m.bias() is not None
                     else np.zeros(W.shape[0])).astype(np.float64)
                d = dict(kind='conv' if isinstance(m, nnq.Conv2d) else 'linear',
                         W=W, scale=sc, bias=b)
                if d['kind'] == 'conv':
                    d.update(stride=int(m.stride[0]), pad=int(m.padding[0]))
                ops.append(d)
            elif isinstance(m, snntorch.Leaky):
                ops.append(dict(kind='lif'))
            elif isinstance(m, torch.nn.MaxPool2d):
                ops.append(dict(kind='maxpool', k=int(m.kernel_size), s=int(m.stride)))
            elif isinstance(m, torch.nn.Flatten):
                ops.append(dict(kind='flatten'))
    # group into stages (each begins at a weight op)
    stages, cur = [], []
    for o in ops:
        if o['kind'] in ('conv', 'linear') and cur:
            stages.append(cur); cur = [o]
        else:
            cur.append(o)
    stages.append(cur)
    return stages


class Stage:
    """One pipeline stage: a weight op (+ optional LIF + trailing pool/flatten)."""
    def __init__(self, ops, device):
        self.ops = ops
        self.wop = ops[0]
        self.has_lif = any(o['kind'] == 'lif' for o in ops)
        self.dev = device
        W = self.wop['W']
        self.Wt = torch.tensor(W, dtype=torch.float64, device=device)
        self.mask = torch.tensor((W != 0), dtype=torch.float64, device=device)
        self.scale = torch.tensor(self.wop['scale'], dtype=torch.float64, device=device)
        self.bias = torch.tensor(self.wop['bias'], dtype=torch.float64, device=device)
        vth = np.maximum(1, np.round(THR / self.wop['scale']).astype(np.int64))
        eb = np.round(64.0 * self.wop['bias'] / self.wop['scale']).astype(np.int64)
        self.vth = torch.tensor(vth, device=device)
        self.eb = torch.tensor(eb, device=device)
        self.V = None                       # membrane, lazily shaped

    def _bshape(self, a):
        return [1, a.shape[1]] + [1] * (a.dim() - 2)

    def _apply(self, x, W):
        if self.wop['kind'] == 'conv':
            return F.conv2d(x, W, None, self.wop['stride'], self.wop['pad'])
        return F.linear(x, W)

    def forward(self, x):
        """x: binary spikes [B,...]. Returns (out_spikes_or_None, readout_inc_or_None,
        ac_count)."""
        a = self._apply(x.double(), self.Wt)                 # integer-valued current
        ac = float(self._apply((x > 0).double(), self.mask).sum().item())
        if not self.has_lif:                                 # readout stage
            real = a * self.scale.view(self._bshape(a)) + self.bias.view(self._bshape(a))
            return None, real, ac
        vth = self.vth.view(self._bshape(a)); eb = self.eb.view(self._bshape(a))
        if self.V is None or self.V.shape != a.shape:
            self.V = torch.zeros_like(a, dtype=torch.long)
        u = a.long() << ASHIFT
        v = torch.sign(self.V) * ((self.V.abs() * (4096 - DV)) >> 12) + u + eb
        v = v.clamp(-MEMCLIP, MEMCLIP)
        vt = vth << ASHIFT
        s = (v > vt).to(torch.long)
        self.V = (v - s * vt) if RESET == 'subtract' else (v * (1 - s))
        out = s.double()
        for o in self.ops[1:]:
            if o['kind'] == 'maxpool':
                out = F.max_pool2d(out, o['k'], o['s'])
            elif o['kind'] == 'flatten':
                out = out.flatten(1)
        return out, None, ac


def run(stages, spike_train, delay):
    T, B = spike_train.shape[0], spike_train.shape[1]
    for st in stages:
        st.V = None
    n = len(stages)
    ro = torch.zeros(B, stages[-1].wop['W'].shape[0], dtype=torch.float64,
                     device=stages[-1].dev)
    ac_tot = [0.0] * n
    spk_tot = [0] * n
    depth = n
    t_run = (T + depth + 3) if delay else T
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
            ac_tot[k] += ac
            if out is not None:
                new[k] = out
                spk_tot[k] += int((out > 0).sum().item())
            if rinc is not None:
                ro += rinc
        prev_out = new
        prev_in = xt
    return ro, dict(ac=ac_tot, spikes=spk_tot, t_run=t_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--delay', action='store_true')
    ap.add_argument('--reset', choices=['hard', 'subtract'], default='hard')
    ap.add_argument('--ckpt', default='results/quantized_snn_alexnet_qat_int8_clean.pth')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from config import config
    from models.quantized_snn_alexnet import (QuantizedSNNAlexNet, get_quant_engine,
                                              prepare_quantized_snn_alexnet_for_qat,
                                              encode_poisson)
    T = config['num_steps']
    global RESET
    RESET = args.reset
    torch.backends.quantized.engine = get_quant_engine()
    m = QuantizedSNNAlexNet(num_classes=10, beta=config['beta'], time_steps=T)
    m = prepare_quantized_snn_alexnet_for_qat(m).eval()
    m = tq.convert(m, inplace=False)
    m.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    m.eval()
    stage_ops = extract_stages(m)
    stages = [Stage(o, args.device) for o in stage_ops]
    print(f"[alexnet-fixedpt] {len(stages)} stages "
          f"({sum(s.has_lif for s in stages)} LIF + 1 readout), "
          f"delay={'ON' if args.delay else 'OFF'}, T={T}, dev={args.device}")

    import torchvision, torchvision.transforms as TT
    tfm = TT.Compose([TT.ToTensor(), TT.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2470, 0.2435, 0.2616))])
    ds = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=tfm)
    N = min(args.n, len(ds))
    correct = 0; ref_correct = 0; ac_run = 0.0; sp_run = None
    m_cpu = m.cpu()
    with torch.no_grad():
        for i in range(N):
            img, lab = ds[i]
            spk = encode_poisson(img.unsqueeze(0), T).to(args.device)
            logits, st = run(stages, spk, args.delay)
            correct += int(logits.argmax(1).item() == lab)
            oc = m_cpu(img.unsqueeze(0))                      # off-chip INT8, SAME sample
            oc = oc[0] if isinstance(oc, tuple) else oc
            ref_correct += int(oc.reshape(-1, 10).sum(0).argmax().item() == lab)
            ac_run += sum(st['ac'])
            sp_run = st['spikes'] if sp_run is None else [a + b for a, b in zip(sp_run, st['spikes'])]
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{N}  Loihi={correct/(i+1)*100:.2f}%  ref={ref_correct/(i+1)*100:.2f}%  avg_ACs={ac_run/(i+1):.3e}")
    print(f"\n[alexnet-fixedpt] N={N}  Loihi_acc={correct/N*100:.2f}%  REF(int8,same samples)={ref_correct/N*100:.2f}%  "
          f"delay={'ON' if args.delay else 'OFF'}  reset={RESET}  t_run={st['t_run']}")
    print(f"[alexnet-fixedpt] mean ACs/sample = {ac_run/N:.4e}")
    print(f"[alexnet-fixedpt] mean spikes/stage = {[int(s/N) for s in sp_run]}")


if __name__ == '__main__':
    main()
