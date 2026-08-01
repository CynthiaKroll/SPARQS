"""
Build + run the SPARQ MLP-QSNN on Lava's fixed-point Loihi-2 simulation.

Runs in `venv-lava` (py3.10). Consumes the .npz from export_mlp_qsnn.py.

Fixed-point mapping (derived from lava PyDenseModelBitAcc + PyLifModelBitAcc,
verified empirically by --calibrate):
  Dense: weights = w_int_k (int8, [-127,127]); num_message_bits=0 so binary
         input -> a_out = w_int @ spike exactly. NB Dense buffers 1 step:
         a_out[t] = w_int @ spike[t-1]  => 1-timestep delay per layer.
  LIF  : du=4095 (u = a_in<<6, no current memory = single-state like snn.Leaky),
         dv=205  ((4096-205)/4096 = 0.94995 ~= beta 0.95),
         vth_k = round(1/scale_k)  (effective_vth = vth<<6 matches thr=1.0),
         bias via (bias_mant<<bias_exp) = round(64*bias_k/scale_k),
         reset = hard (stock LIF).  Subtract-reset is the Phase-2 variant.

Because Loihi adds 1 delay/layer, we run T_RUN = T + n_layers + pad steps and
feed input spikes only for the first T steps (zeros after) to flush the pipeline.

Usage:
  venv-lava\\Scripts\\python.exe lava_loihi_sim\\sparq\\lava_build_mlp.py \
      --npz lava_loihi_sim\\exports\\mlp_qsnn.npz --n 20 --calibrate
"""
import argparse
import numpy as np

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeSource
from lava.proc.io.sink import RingBuffer as SpikeSink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

DECAY_UNITY = 4096
BETA_TARGET = 0.95


def beta_to_dv(beta):
    """dv such that (4096-dv)/4096 == beta (dm_offset=0)."""
    return int(round((1.0 - beta) * DECAY_UNITY))


def bias_to_mant_exp(effective_bias):
    """Split target effective_bias into (mant, exp) with mant in int13 range."""
    eb = int(round(effective_bias))
    if eb == 0:
        return 0, 0
    exp = 0
    while abs(eb >> exp) > 4095 and exp < 7:
        exp += 1
    return int(np.sign(eb) * (abs(eb) >> exp)), exp


def round_to_even(w):
    """Pre-round int8 weights to nearest EVEN before Loihi stores them.

    Lava's PyDenseModelBitAcc truncates mixed-sign 8-bit weights to 7 magnitude
    bits by flooring the LSB (w>>1<<1) -- a *biased* rounding that, across ~32M
    weights, systematically shifts every layer and collapses accuracy
    (98.5%->69.5% on the MLP). Pre-rounding to nearest even makes the truncation
    a no-op AND removes the bias (errors cancel), recovering ~96%."""
    return np.clip(np.round(w / 2.0) * 2, -126, 126).astype(np.int32)


def load_layers(npz):
    d = np.load(npz, allow_pickle=True)
    n = int(d['n_layers'])
    layers = []
    for k in range(n):
        layers.append(dict(
            w=round_to_even(d[f'w{k}'].astype(np.int32)),   # [out, in], Loihi 7-bit even
            scale=float(d[f'scale{k}']),
            bias=d[f'bias{k}'].astype(np.float64),
        ))
    keys = set(d.files)
    # readout: 'integrator' (QSNN: final Linear has no LIF, accumulate its output)
    #       or 'spike_count' (FP32 SNN: every Linear has a LIF, classify by output
    #          spike counts -- rate-coded readout)
    readout = str(d['readout']) if 'readout' in keys else 'integrator'
    thr = (d['thresholds'].astype(float).tolist() if 'thresholds' in keys
           else [float(d['threshold'])] * int(d['n_lif']))
    # ac_fanout: mean NONZERO weights per input unit. For a true dense layer this
    # equals out_features, but for a conv unrolled into a dense matrix (LeNet) the
    # matrix is mostly zeros -- using out_features would overcount ACs ~35x.
    fanout = (d['ac_fanout'].astype(float).tolist() if 'ac_fanout' in keys else None)
    # input_mode 'spikes' (Poisson) or 'bias' (DIRECT/constant-current encoding:
    # layer 0's current is injected as a constant LIF bias, so layer 0 performs no
    # synaptic ops -- see lenet_snn).
    input_mode = str(d['input_mode']) if 'input_mode' in keys else 'spikes'
    input_current = d['input_current'] if 'input_current' in keys else None
    meta = dict(
        ac_fanout=fanout, input_mode=input_mode, input_current=input_current,
        beta=float(d['beta']), threshold=float(d['threshold']),
        thresholds=thr, readout=readout,
        n_lif=int(d['n_lif']), T=int(d['T']),
        spikes=d['spikes'],                         # [N, T, in0] uint8
        labels=d['labels'], golden_pred=d['golden_pred'],
        golden_fire=d['golden_fire'], int8_acc=float(d['int8_acc']),
        golden_acc=float(d['golden_acc']),
    )
    return layers, meta


def run_sample(layers, meta, spike_in, t_run, inp_current=None):
    """Run ONE sample through the fixed-point Lava net.

    spike_in: [T, in0] uint8. Returns (logits[10], per_lif_spike_counts, per_layer_input_spikes).
    """
    n_lif = meta['n_lif']
    T = spike_in.shape[0]
    in0 = spike_in.shape[1]

    dv = beta_to_dv(meta['beta'])
    bias_mode = meta.get('input_mode') == 'bias'
    procs = []
    lifs = []
    readout = None

    if bias_mode:
        # DIRECT encoding: no input spikes. Layer 0's constant current becomes the
        # first LIF's bias; a zero source only exists to drive the runtime.
        L0 = layers[0]
        n0 = L0['w'].shape[0]
        src = SpikeSource(data=np.zeros((n0, t_run), dtype=np.int32))
        eb = 64.0 * np.asarray(inp_current, np.float64) / L0['scale']
        mant = np.zeros(n0, np.int32); exp = np.zeros(n0, np.int32)
        for j, v in enumerate(eb):
            m, e = bias_to_mant_exp(v); mant[j] = m; exp[j] = e
        vth0 = max(1, int(round(meta['thresholds'][0] / L0['scale'])))
        lif0 = LIF(shape=(n0,), du=4095, dv=dv, vth=vth0, bias_mant=mant, bias_exp=exp)
        src.s_out.connect(lif0.a_in)
        sink0 = SpikeSink(shape=(n0,), buffer=t_run)
        lif0.s_out.connect(sink0.a_in)
        procs += [src, lif0]; lifs.append((lif0, sink0))
        prev = lif0
        start_k = 1
    else:
        # Pad the input spike train out to t_run steps (zeros flush the pipeline).
        data = np.zeros((in0, t_run), dtype=np.int32)
        data[:, :T] = spike_in.T.astype(np.int32)
        src = SpikeSource(data=data)
        procs = [src]
        prev = src
        start_k = 0

    for k, L in enumerate(layers):
        if k < start_k:
            continue
        dense = Dense(weights=L['w'])               # int8 weights, num_message_bits=0
        procs.append(dense)
        prev.s_out.connect(dense.s_in)
        if k < n_lif:
            vth = max(1, int(round(meta['thresholds'][k] / L['scale'])))
            eff_bias = 64.0 * L['bias'] / L['scale']   # act_shift=6 -> input scaled by 64
            mant = np.zeros(L['w'].shape[0], np.int32)
            exp = np.zeros(L['w'].shape[0], np.int32)
            for j, eb in enumerate(eff_bias):
                m, e = bias_to_mant_exp(eb)
                mant[j] = m; exp[j] = e
            lif = LIF(shape=(L['w'].shape[0],), du=4095, dv=dv, vth=vth,
                      bias_mant=mant, bias_exp=exp)
            dense.a_out.connect(lif.a_in)
            sink = SpikeSink(shape=(L['w'].shape[0],), buffer=t_run)
            lif.s_out.connect(sink.a_in)
            procs.append(lif); lifs.append((lif, sink))
            prev = lif
        else:
            # final readout: non-spiking integrator (dv=0 = plain sum, matching the
            # QSNN's sum-over-T). Fold the final-layer bias INTO the neuron bias so
            # v = 64/scale * sum_t(W@spk + b) exactly. (Adding T*bias post-hoc was
            # unit-inconsistent -- v is in 64/scale units -- though harmless while
            # the final bias is ~0.)
            eb = 64.0 * L['bias'] / L['scale']
            rm = np.zeros(L['w'].shape[0], np.int32); re = np.zeros(L['w'].shape[0], np.int32)
            for j, v in enumerate(eb):
                m, e = bias_to_mant_exp(v); rm[j] = m; re[j] = e
            ro = LIF(shape=(L['w'].shape[0],), du=4095, dv=0, vth=2**20,
                     bias_mant=rm, bias_exp=re)
            procs.append(ro)
            dense.a_out.connect(ro.a_in)
            readout = ro

    run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
    try:
        src.run(condition=RunSteps(num_steps=t_run), run_cfg=run_cfg)
        if readout is not None:
            logits = np.asarray(readout.v.get()).astype(np.float64).copy()
        else:
            logits = None
        lif_spikes = [s.data.get().sum(axis=1).astype(np.int64) for (_, s) in lifs]  # per-neuron totals
    finally:
        src.stop()

    if readout is not None:
        # bias is now folded into the readout neuron (see final-readout branch)
        pass
    else:
        # spike-count readout: class votes = output-layer spike totals
        logits = lif_spikes[-1].astype(np.float64)
    return logits, lif_spikes


def calibrate(layers, meta, spike_in):
    """Single-layer check: does Lava fixed-point layer 0 reproduce the golden
    LIF spike behaviour (hard reset) for one sample?"""
    L = layers[0]; T = spike_in.shape[0]
    x = spike_in.astype(np.float64)                            # [T, in0]
    W = L['w'].astype(np.float64) * L['scale']                 # [out, in]
    b = L['bias'].astype(np.float64)
    thr = meta['threshold']; beta = meta['beta']
    # golden with HARD reset (to match stock Lava LIF)
    mem = np.zeros(W.shape[0]); gold = np.zeros((T, W.shape[0]), np.int64)
    for t in range(T):
        mem = beta*mem + (W @ x[t] + b)
        spk = (mem >= thr).astype(np.float64)
        mem = mem * (1 - spk)                                    # HARD reset
        gold[t] = spk.astype(np.int64)

    # lava single layer, t_run = T+2
    t_run = T + 2
    data = np.zeros((spike_in.shape[1], t_run), np.int32); data[:, :T] = spike_in.T
    src = SpikeSource(data=data); dense = Dense(weights=L['w'])
    vth = max(1, int(round(1.0/L['scale'])))
    eff_bias = 64.0*L['bias']/L['scale']
    mant = np.zeros(W.shape[0], np.int32); exp = np.zeros(W.shape[0], np.int32)
    for j, eb in enumerate(eff_bias):
        m, e = bias_to_mant_exp(eb); mant[j]=m; exp[j]=e
    lif = LIF(shape=(W.shape[0],), du=4095, dv=beta_to_dv(beta), vth=vth, bias_mant=mant, bias_exp=exp)
    sink = SpikeSink(shape=(W.shape[0],), buffer=t_run)
    src.s_out.connect(dense.s_in); dense.a_out.connect(lif.a_in); lif.s_out.connect(sink.a_in)
    try:
        src.run(condition=RunSteps(num_steps=t_run), run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
        lava = sink.data.get().astype(np.int64)                 # [n_out, t_run]
    finally:
        src.stop()
    # Dense buffers 1 step -> lava spikes appear shifted by +1 (dense) [+ maybe src]
    print(f"[calib] golden total spikes = {gold.sum()}, lava total = {lava.sum()}")
    for shift in range(0, 4):
        lv = lava[:, shift:shift+T].T if shift+T <= lava.shape[1] else None
        if lv is None: continue
        match = (lv == gold).mean()*100 if lv.shape == gold.shape else -1
        print(f"   shift={shift}: per-neuron-per-step match = {match:.1f}%  "
              f"(lava window spikes={lv.sum()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='lava_loihi_sim/exports/mlp_qsnn.npz')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--pad', type=int, default=4, help='extra flush steps beyond T+depth')
    ap.add_argument('--calibrate', action='store_true')
    args = ap.parse_args()

    layers, meta = load_layers(args.npz)
    print(f"[lava] {len(layers)} layers, n_lif={meta['n_lif']}, T={meta['T']}, "
          f"int8_acc={meta['int8_acc']*100:.2f}%, golden_acc={meta['golden_acc']*100:.2f}%")
    spikes = meta['spikes']; labels = meta['labels']; gpred = meta['golden_pred']

    if args.calibrate:
        calibrate(layers, meta, spikes[0])
        return

    t_run = meta['T'] + len(layers) + args.pad
    N = min(args.n, spikes.shape[0])
    n_lif = meta['n_lif']
    out_feats = [L['w'].shape[0] for L in layers]
    # per-spike synaptic fan-out used for AC counting (see load_layers)
    fanout = meta['ac_fanout'] if meta.get('ac_fanout') else [float(v) for v in out_feats]
    correct = 0; gmatch = 0
    spk_sum = np.zeros(n_lif)                 # total output spikes/layer (summed over samples)
    ac_sum = np.zeros(len(layers))           # total ACs/layer
    for i in range(N):
        cur = (meta['input_current'][i] if meta.get('input_mode') == 'bias' else None)
        logits, lif_spikes = run_sample(layers, meta, spikes[i], t_run, cur)
        pred = int(np.argmax(logits))
        correct += (pred == labels[i]); gmatch += (pred == gpred[i])
        # AC ops: dense layer k does (active input spikes) x (out neurons) accumulates.
        in_spk = int(spikes[i].sum())        # input spikes to layer 0 (encoded image)
        for k in range(len(layers)):
            ac_sum[k] += in_spk * fanout[k]
            if k < n_lif:
                s = int(lif_spikes[k].sum())
                spk_sum[k] += s
                in_spk = s                   # feeds next layer
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{N}  lava_acc={correct/(i+1)*100:.1f}%  match_golden={gmatch/(i+1)*100:.1f}%")

    total_ac = ac_sum.sum() / N
    # --- persist a machine-readable artifact for the power/latency report ---
    import json, os
    art = dict(
        model=os.path.splitext(os.path.basename(args.npz))[0],
        readout=meta['readout'],
        n_samples=int(N), T=int(meta['T']), t_run=int(t_run),
        n_layers=len(layers), n_lif=int(n_lif),
        accuracy=float(correct / N), golden_acc=float(meta['golden_acc']),
        int8_acc=float(meta['int8_acc']), agreement_golden=float(gmatch / N),
        layer_names=[f'fc{k+1}' for k in range(len(layers))],
        out_features=[int(v) for v in out_feats],
        ac_fanout=[float(v) for v in fanout],
        in_features=[int(L['w'].shape[1]) for L in layers],
        acs_per_sample=[float(ac_sum[k] / N) for k in range(len(layers))],
        total_acs_per_sample=float(total_ac),
        spikes_per_sample=[float(spk_sum[k] / N) for k in range(n_lif)],
        neurons=[int(L['w'].shape[0]) for L in layers],
    )
    out_json = os.path.splitext(args.npz)[0] + f'_lava_n{N}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(art, f, indent=2)
    print(f"[lava fixed-pt] wrote artifact -> {out_json}")

    print(f"\n[lava fixed-pt] N={N}  acc={correct/N*100:.2f}%  "
          f"(golden {meta['golden_acc']*100:.2f}%, int8 {meta['int8_acc']*100:.2f}%)")
    print(f"[lava fixed-pt] pred agreement w/ golden = {gmatch/N*100:.2f}%")
    print(f"[lava fixed-pt] avg spikes/sample per LIF layer = "
          f"{[int(spk_sum[k]/N) for k in range(n_lif)]}")
    print(f"[lava fixed-pt] avg ACs/sample per layer = "
          f"{[f'{ac_sum[k]/N:.3e}' for k in range(len(layers))]}")
    print(f"[lava fixed-pt] TOTAL ACs/sample = {total_ac:.3e}  (over T={meta['T']} input steps)")


if __name__ == '__main__':
    main()
