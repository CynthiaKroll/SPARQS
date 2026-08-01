"""
Run the SPARQ QDSNN on Lava's fixed-point Loihi-2 simulation by PATH DECOMPOSITION.

Runs in `venv-lava` (py3.10). Consumes exports/mlp_qdsnn.npz.

Loihi runs a STATIC graph -- there is no "stop early when confident". So:
  * PyTorch already decided the exits; the npz carries the per-sample
    (exit_point, exit_timestep) measured from the real trained model.
  * Here, each sample is simulated on Loihi through ONLY the sub-network its
    exit selected, for ONLY the ticks it actually used.
  * Reported as: per-path hardware-simulated cost weighted by the measured exit
    distribution -- NOT an end-to-end hardware run.

Paths (see _forward_inference in branchy_quantized_snn_mlp.py):
  exit 1 -> fc1 -> lif1 -> [exit1 head] -> integrator readout
  exit 2 -> fc1..fc3 -> lif1..lif3 -> [exit2 head] -> integrator readout
  exit 3 -> fc1..fc8 (full backbone) -> integrator readout on fc8

Cost accounting matches the source exactly: every active tick a sample pays
fc1 + exit1; if it did not exit at 1 it also pays fc2 + fc3 + exit2; if it did
not exit at 2 it also pays fc4..fc8. The exit heads that are computed but not
used as the prediction (e.g. exit1 on an exit-2 sample) are cost-only, so their
ACs are added analytically (= driving-layer spikes x 10) instead of building
extra Lava processes -- exact, and keeps the graph small.

Exit logits accumulate with decay d=0.9 (NOT the LIF beta) -> readout integrator
uses dv = round((1-0.9)*4096) = 410.

Usage:
  venv-lava\\Scripts\\python.exe lava_loihi_sim\\sparq\\lava_build_qdsnn.py \
      --npz lava_loihi_sim\\exports\\mlp_qdsnn.npz --n 100
"""
import argparse
import json
import os
import numpy as np

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeSource
from lava.proc.io.sink import RingBuffer as SpikeSink
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava_build_mlp import beta_to_dv, bias_to_mant_exp, round_to_even

# Readout integrator must NEVER spike. Lava computes effective_vth = vth << 6 in
# int32, and the membrane is clipped to +/-2**23 (8388608). So vth must satisfy
#   (vth << 6) > 2**23   AND   (vth << 6) < 2**31
# 2**20 -> 67,108,864: safely above the clip, well inside int32.
# DO NOT raise this to 2**26 -- (2**26 << 6) == 2**32 overflows int32 to ZERO,
# which makes the integrator spike every tick and reset, destroying the logits
# (this produced 51% accuracy / 0-of-7 on exit1 before it was caught).
READOUT_VTH = 2 ** 20


def load_qdsnn(npz):
    d = np.load(npz, allow_pickle=True)
    n = int(d['n_layers'])
    layers = [dict(w=round_to_even(d[f'w{k}'].astype(np.int32)),
                   scale=float(d[f'scale{k}']),
                   bias=d[f'bias{k}'].astype(np.float64)) for k in range(n)]
    exits = [dict(w=round_to_even(d[f'ew{j}'].astype(np.int32)),
                  scale=float(d[f'escale{j}']),
                  bias=d[f'ebias{j}'].astype(np.float64)) for j in range(int(d['n_exits']))]
    keys = set(d.files)
    meta = dict(
        # AC fan-out: mean NONZERO weights per input (conv-unrolled-to-dense is
        # mostly zeros, so out_features would massively overcount -- see LeNet).
        ac_fanout=(d['ac_fanout'].astype(float).tolist() if 'ac_fanout' in keys else None),
        exit_fanout=(d['exit_fanout'].astype(float).tolist() if 'exit_fanout' in keys else None),
        beta=float(d['beta']), threshold=float(d['threshold']),
        n_lif=int(d['n_lif']), T=int(d['T']), exit_decay=float(d['exit_decay']),
        exit_after=d['exit_after'].astype(int).tolist(),
        spikes=d['spikes'], labels=d['labels'],
        exit_points=d['exit_points'], exit_timesteps=d['exit_timesteps'],
        golden_pred=d['golden_pred'], golden_acc=float(d['golden_acc']),
    )
    return layers, exits, meta


def path_spec(exit_point, layers, exits, meta):
    """Return (backbone_idx, head, cost_heads) for the taken exit.

    Generic over topology via meta['exit_after'] -- exit j attaches after backbone
    layer index exit_after[j]. MLP: [0,2] (exit1 after fc1, exit2 after fc3).
    LeNet: [0,1] (exit1 after conv1, exit2 after conv2).

    backbone_idx : backbone layers this path executes
    head         : exit head used for the PREDICTION (None -> final-layer readout)
    cost_heads   : (head_idx, driving_backbone_idx) heads computed but not used as
                   the prediction -- cost-only, ACs added analytically
    """
    ea = meta['exit_after']
    if exit_point == 1:
        return list(range(ea[0] + 1)), exits[0], [(0, ea[0])]
    if exit_point == 2:
        return list(range(ea[1] + 1)), exits[1], [(0, ea[0]), (1, ea[1])]
    return list(range(len(layers))), None, [(0, ea[0]), (1, ea[1])]


def _split_bias(bias, scale):
    """Per-neuron (mant, exp) for effective bias = 64*bias/scale (act_shift=6)."""
    eff = 64.0 * np.asarray(bias, np.float64) / scale
    mant = np.zeros(len(eff), np.int32); exp = np.zeros(len(eff), np.int32)
    for j, eb in enumerate(eff):
        m, e = bias_to_mant_exp(eb); mant[j] = m; exp[j] = e
    return mant, exp


def make_lif(n_out, scale, bias, thr, dv):
    vth = max(1, int(round(thr / scale)))
    mant, exp = _split_bias(bias, scale)
    return LIF(shape=(n_out,), du=4095, dv=dv, vth=vth, bias_mant=mant, bias_exp=exp)


def make_readout(scale, bias, dv):
    """Non-spiking integrator with the layer bias folded in (correct units).

    v = decay*v + 64*(W@spk) + 64*bias/scale  ==  64/scale * (decay*.. + W@spk + b),
    matching the model's `acc = acc*d + (W@spk + b)`. dv sets the decay (dv_exit for
    d=0.9 accumulators; 0 for a plain sum)."""
    n = len(bias)
    mant, exp = _split_bias(bias, scale)
    return LIF(shape=(n,), du=4095, dv=dv, vth=READOUT_VTH, bias_mant=mant, bias_exp=exp)


def run_sample(layers, exits, meta, spike_in, ts, exit_point, feed_ts=None):
    """Simulate ONE sample through its own exit path.

    feed_ts = number of INPUT ticks to drive the sub-network for the PREDICTION.
    Loihi runs a static graph (no dynamic mid-graph halt -- that is exactly why we
    do path decomposition), so the correct per-path accuracy is the sub-network's
    full-T output; the dynamic exit timestep `ts` governs only latency/AC accounting
    (done by the caller), NOT the prediction. Passing feed_ts=ts (the old behaviour)
    starves the decay-0.9 exit readout under the per-layer delay and collapses the
    exit paths. Default None -> full T."""
    if feed_ts is None:
        feed_ts = int(meta['T'])
    feed_ts = min(feed_ts, spike_in.shape[0])
    bidx, head, _ = path_spec(exit_point, layers, exits, meta)
    dv = beta_to_dv(meta['beta'])
    dv_exit = int(round((1.0 - meta['exit_decay']) * 4096))     # 410 for d=0.9
    n_lif = meta['n_lif']

    # Read the readout at its PEAK: the last input tick (feed_ts-1) reaches the
    # readout after `depth` layer-delays, i.e. at tick feed_ts+depth-1. Running past
    # that only drains the pipeline (zero-input flush ticks) while the decay-0.9
    # readout keeps leaking -- reading late cost ~15pp in the fixed-point replica.
    depth = len(bidx) + (1 if head is not None else 0)
    t_run = feed_ts + depth
    in0 = spike_in.shape[1]
    data = np.zeros((in0, t_run), dtype=np.int32)
    data[:, :feed_ts] = spike_in[:feed_ts].T.astype(np.int32)
    src = SpikeSource(data=data)

    prev = src
    lifs = []
    readout = None
    for k in bidx:
        L = layers[k]
        dense = Dense(weights=L['w'])
        prev.s_out.connect(dense.s_in)
        if k < n_lif:                       # fc1..fc7 have a LIF; fc8 does not
            lif = make_lif(L['w'].shape[0], L['scale'], L['bias'], meta['threshold'], dv)
            dense.a_out.connect(lif.a_in)
            sink = SpikeSink(shape=(L['w'].shape[0],), buffer=t_run)
            lif.s_out.connect(sink.a_in)
            lifs.append((k, sink))
            prev = lif
        else:                               # final backbone Linear -> integrator
            # The model accumulates the FINAL exit with decay d=0.9 too
            # (fa = fa*0.9 + f), NOT a plain sum -- so this readout must leak at
            # dv_exit, exactly like the exit heads below, AND carry the final-layer
            # bias. (dv=0 + no bias here was a bug: harmless for the MLP QDSNN where
            # 0 samples reach exit3, but it corrupted every exit-3 prediction on
            # LeNet, where ~98% do.)
            ro = make_readout(L['scale'], L['bias'], dv_exit)
            dense.a_out.connect(ro.a_in)
            readout = ro

    if head is not None:                    # exit head -> leaky integrator (d=0.9)
        hd = Dense(weights=head['w'])
        prev.s_out.connect(hd.s_in)
        ro = make_readout(head['scale'], head['bias'], dv_exit)
        hd.a_out.connect(ro.a_in)
        readout = ro

    # ACs/energy reflect the DYNAMIC exit: the sample would have halted at `ts`, so
    # count spikes only within its active window (ts input ticks + pipeline fill),
    # even though the PREDICTION integrates the full feed_ts window above.
    acc_cut = min(t_run, int(ts) + depth + 1)
    try:
        src.run(condition=RunSteps(num_steps=t_run), run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
        logits = np.asarray(readout.v.get()).astype(np.float64).copy()
        spikes_by_layer = {k: s.data.get()[:, :acc_cut].sum(axis=1).astype(np.int64).sum()
                           for (k, s) in lifs}
    finally:
        src.stop()
    return logits, spikes_by_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='lava_loihi_sim/exports/mlp_qdsnn.npz')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--feed', choices=['full', 'exit'], default='full',
                    help="input ticks driving the PREDICTION: 'full'=T (correct for "
                         "a static Loihi sub-network) or 'exit'=exit-timestep (old, "
                         "starves the decay-0.9 exit readout under delay). Latency/AC "
                         "accounting always uses the real exit timestep either way.")
    args = ap.parse_args()

    layers, exits, meta = load_qdsnn(args.npz)
    meta['exit_after'] = meta.get('exit_after', [0, 2])
    spikes = meta['spikes']; labels = meta['labels']
    epts = meta['exit_points']; ets = meta['exit_timesteps']
    N = min(args.n, spikes.shape[0])
    out_feats = [L['w'].shape[0] for L in layers]
    fanout = meta['ac_fanout'] if meta.get('ac_fanout') else [float(v) for v in out_feats]
    efan = (meta['exit_fanout'] if meta.get('exit_fanout')
            else [float(e['w'].shape[0]) for e in exits])

    print(f"[qdsnn] paths from measured exits; T={meta['T']}, "
          f"golden(composed) acc={meta['golden_acc']*100:.2f}%")
    dist = {int(k): int((epts[:N] == k).sum()) for k in (1, 2, 3)}
    print(f"[qdsnn] exit distribution over N={N}: {dist}")

    correct = 0; gmatch = 0
    ac_total = 0.0; ticks_total = 0.0
    trun_total = 0.0        # ACTUAL simulated ticks (input + pipeline fill + pad)
                            # -- this, not the input-tick count, is the latency.
    per_exit = {1: dict(n=0, ac=0.0, ok=0), 2: dict(n=0, ac=0.0, ok=0), 3: dict(n=0, ac=0.0, ok=0)}

    feed_ts_val = int(meta['T']) if args.feed == 'full' else None
    for i in range(N):
        ep = int(epts[i]); ts = int(ets[i])
        logits, spk = run_sample(layers, exits, meta, spikes[i], ts, ep,
                                 feed_ts=(feed_ts_val if args.feed == 'full' else ts))
        pred = int(np.argmax(logits))
        correct += (pred == labels[i]); gmatch += (pred == meta['golden_pred'][i])

        # --- ACs: backbone + cost-only exit heads ---
        bidx, head, cost_heads = path_spec(ep, layers, exits, meta)
        trun_total += ts + len(bidx) + (1 if head is not None else 0) + 3
        in_spk = int(spikes[i, :ts].sum())
        ac = 0.0
        for k in bidx:
            ac += in_spk * fanout[k]
            in_spk = int(spk.get(k, 0)) if k in spk else in_spk
        for (hj, drive_k) in cost_heads:
            ac += int(spk.get(drive_k, 0)) * efan[hj]

        ac_total += ac; ticks_total += ts
        pe = per_exit[ep]; pe['n'] += 1; pe['ac'] += ac; pe['ok'] += int(pred == labels[i])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}  acc={correct/(i+1)*100:.1f}%  "
                  f"avg_ACs={ac_total/(i+1):.3e}  avg_ticks={ticks_total/(i+1):.2f}")

    print(f"\n[qdsnn lava] N={N}  acc={correct/N*100:.2f}%  "
          f"(golden composed {meta['golden_acc']*100:.2f}%)")
    print(f"[qdsnn lava] agreement w/ golden = {gmatch/N*100:.2f}%")
    print(f"[qdsnn lava] MEAN ACs/sample = {ac_total/N:.4e}")
    print(f"[qdsnn lava] MEAN ticks/sample = {ticks_total/N:.2f} / {meta['T']}")
    for k in (1, 2, 3):
        pe = per_exit[k]
        if pe['n']:
            print(f"   exit{k}: n={pe['n']:<4} acc={pe['ok']/pe['n']*100:5.1f}%  "
                  f"mean ACs={pe['ac']/pe['n']:.3e}")

    art = dict(
        model=os.path.splitext(os.path.basename(args.npz))[0],
        method='path_decomposition',
        n_samples=int(N), T=int(meta['T']),
        accuracy=float(correct / N), golden_acc=float(meta['golden_acc']),
        int8_acc=float(meta['golden_acc']), agreement_golden=float(gmatch / N),
        total_acs_per_sample=float(ac_total / N),
        mean_ticks_per_sample=float(ticks_total / N),   # INPUT ticks used (the exit metric)
        t_run=float(trun_total / N),                    # ACTUAL ticks simulated = latency
        exit_distribution={str(k): per_exit[k]['n'] for k in (1, 2, 3)},
        acs_by_exit={str(k): (per_exit[k]['ac'] / per_exit[k]['n'] if per_exit[k]['n'] else 0.0)
                     for k in (1, 2, 3)},
        layer_names=[f'fc{k+1}' for k in range(len(layers))],
        in_features=[int(L['w'].shape[1]) for L in layers],
        out_features=[int(v) for v in out_feats],
        neurons=[int(L['w'].shape[0]) for L in layers],
        acs_per_sample=[], spikes_per_sample=[], n_layers=len(layers), n_lif=meta['n_lif'],
    )
    out_json = os.path.splitext(args.npz)[0] + f'_lava_n{N}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(art, f, indent=2)
    print(f"[qdsnn lava] wrote artifact -> {out_json}")


if __name__ == '__main__':
    main()
