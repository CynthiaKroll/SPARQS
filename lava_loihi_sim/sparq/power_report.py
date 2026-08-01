"""
SPARQ Loihi-2 power / energy / latency report.

Consumes the JSON artifact written by lava_build_mlp.py (real spike + AC counts
measured in Lava's bit-accurate fixed-point Loihi-2 simulation) and turns it into
energy, power and latency.

DELIBERATELY NOT a port of the BLADE `power_estimation.py`: that file hardcodes
another paper's rows (Ahmed hybrid 0.97e9 ACs / 3.1 mJ) and a project-specific
"reporting rule" banner. Only its *methodology* is reused here.

Reporting stance (SPARQ):
  - SPARQ's own analytical model stays the paper HEADLINE.
  - Loihi numbers here are an INDEPENDENT CROSS-CHECK, used mainly for the
    RELATIVE claim (QDSNN vs QSNN on ACs / timesteps). Absolute mJ differ because
    the two models measure different things -- do not force them to agree.

Energy constants -- three columns, clearly sourced:
  Loihi 1 (PUBLISHED)   23.6 pJ/synop, 81 pJ/neuron-update
                        Davies et al., IEEE Micro 2018, Table 2. Conservative
                        upper bound; the only officially published per-op figures.
  Loihi 2 (ESTIMATED)    8.0 pJ/synop, 30 pJ/neuron-update
                        Intel has NOT published per-op energy for Loihi 2.
                        Derived from Hala Point (380 TSOP/s @ 2600 W ~ 6.8
                        pJ/synop system-level) + Isik et al. 2024 (~10-12 pJ/op).
  SPARQ analytical      0.01 pJ/AC (config.E_AC_NEUROMORPHIC_J)
                        SPARQ's own constant. NOTE: this is the raw 8-bit ALU add
                        cost, NOT a full synaptic operation (which also pays a
                        weight memory read, routing and neuron state update).
                        It is ~800-2400x more optimistic than published Loihi
                        per-synop energy -- surfaced here so the paper does not
                        rest on it silently.

Latency: Loihi is clocked per timestep. The network needs T input ticks PLUS one
tick per layer to fill the pipeline (Loihi's Dense adds 1 tick of dendritic
delay), so wall-clock latency = t_run * time_per_timestep.

Usage:
  venv-lava\\Scripts\\python.exe lava_loihi_sim\\sparq\\power_report.py \
      --json lava_loihi_sim\\exports\\mlp_qsnn_lava_n100.json
"""
import argparse
import json
from datetime import datetime

# --- per-operation energy constants (J) ---
E_SYNOP = {
    'loihi1': 23.6e-12,   # published
    'loihi2': 8.0e-12,    # estimated
    'sparq': 0.01e-12,    # SPARQ config E_AC_NEUROMORPHIC_J
}
E_NEURON = {
    'loihi1': 81.0e-12,
    'loihi2': 30.0e-12,
    'sparq': 0.0,         # SPARQ's model has no separate neuron-update term
}
SOURCE = {
    'loihi1': 'Davies et al., IEEE Micro 2018, Tab.2 (PUBLISHED)',
    'loihi2': 'Hala Point + Isik et al. 2024 (ESTIMATED, not Intel-published)',
    'sparq':  'SPARQ config.E_AC_NEUROMORPHIC_J (raw ALU add, NOT a full synop)',
}

# Loihi 2 timestep: algorithmic timestep is typically ~10 us for modest networks.
# Reported explicitly so the assumption is visible and adjustable.
TIME_PER_TIMESTEP_S = 10e-6
NEURONS_PER_CORE = 8192
CORES_PER_CHIP = 128


def active_neurons(art):
    """Neurons actually evaluated per inference.

    For a normal (static) run that is every neuron. For a path-decomposition
    (QDSNN) run only the layers the taken exit executes are live, so counting
    all 8 layers would badly overstate neuron energy -- weight by the measured
    exit distribution instead.
    """
    if art.get('method') != 'path_decomposition':
        return sum(art['neurons'])
    of = art['out_features']
    dist = art.get('exit_distribution', {})
    total = sum(dist.values()) or 1
    # exit1 -> fc1 (+1 head); exit2 -> fc1..fc3 (+2 heads); exit3 -> full (+2 heads)
    paths = {'1': (of[:1], 1), '2': (of[:3], 2), '3': (of, 2)}
    acc = 0.0
    for k, cnt in dist.items():
        if not cnt:
            continue
        layer_out, n_heads = paths[k]
        acc += cnt * (sum(layer_out) + 10 * n_heads)
    return acc / total


def energy_breakdown(art, gen):
    """Energy in J for one inference under generation `gen`."""
    syn = art['total_acs_per_sample'] * E_SYNOP[gen]
    # neuron updates: every neuron is evaluated every tick it is live
    neu = active_neurons(art) * art['t_run'] * E_NEURON[gen]
    return syn, neu, syn + neu


def fmt_j(x):
    """Human-friendly energy string."""
    if x >= 1e-3:
        return f"{x*1e3:.4f} mJ"
    if x >= 1e-6:
        return f"{x*1e6:.4f} uJ"
    return f"{x*1e9:.4f} nJ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--time-per-step-us', type=float, default=TIME_PER_TIMESTEP_S * 1e6)
    ap.add_argument('--out', default=None, help='optional path to save the report')
    args = ap.parse_args()

    with open(args.json, encoding='utf-8') as f:
        art = json.load(f)

    t_step = args.time_per_step_us * 1e-6
    latency_s = art['t_run'] * t_step
    n_neurons = active_neurons(art)
    n_synapses = sum(i * o for i, o in zip(art['in_features'], art['out_features']))
    cores = -(-n_neurons // NEURONS_PER_CORE)      # ceil
    chips = -(-cores // CORES_PER_CHIP)

    L = []
    w = L.append
    w("=" * 78)
    w(f" SPARQ {art['model'].upper()} -- Loihi 2 simulation: energy / power / latency")
    w(f" Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}   "
      f"samples={art['n_samples']}  T={art['T']}  t_run={art['t_run']}")
    w("=" * 78)
    w("")
    w(" Spike activity + ACs measured in Lava's bit-accurate fixed-point Loihi-2")
    w(" simulation. Energy/power/latency are ESTIMATES = measured ops x per-op")
    w(" constants; they are NOT wattmeter readings from silicon.")
    w("")

    # --- accuracy ---
    w(" ACCURACY")
    w(" " + "-" * 66)
    w(f"   Loihi fixed-point (simulated) : {art['accuracy']*100:.2f}%")
    w(f"   Reference (real-valued LIF)   : {art['golden_acc']*100:.2f}%")
    w(f"   PyTorch INT8 model            : {art['int8_acc']*100:.2f}%")
    w(f"   Agreement with reference      : {art['agreement_golden']*100:.2f}%")
    w(f"   Gap vs reference              : "
      f"{(art['golden_acc']-art['accuracy'])*100:.2f} pp "
      f"(8-layer pipeline delay: Loihi adds 1 tick/layer)")
    w("")

    # --- compute ---
    w(" COMPUTE (per inference)")
    w(" " + "-" * 66)
    w(f"   Total ACs (synaptic ops)      : {art['total_acs_per_sample']:.4e}")
    w(f"   Total spikes                  : {sum(art['spikes_per_sample']):.4e}")
    w(f"   Neurons                       : {n_neurons:,}")
    w(f"   Synapses                      : {n_synapses:,}")
    w(f"   NOTE: Loihi performs ACs, not MACs. MAC counts are the GPU/ANN")
    w(f"         baseline and are reported separately by SPARQ's own counters.")
    w("")

    w(f" {'Layer':<8} {'shape':<16} {'ACs':<14} {'spikes':<12} {'neurons':<10}")
    w(" " + "-" * 66)
    for k, name in enumerate(art['layer_names'] if art.get('acs_per_sample') else []):
        sp = f"{art['spikes_per_sample'][k]:.3e}" if k < art['n_lif'] else "-(readout)"
        w(f" {name:<8} {str(art['in_features'][k])+'->'+str(art['out_features'][k]):<16} "
          f"{art['acs_per_sample'][k]:<14.3e} {sp:<12} {art['neurons'][k]:<10}")
    w("")

    # --- latency ---
    w(" LATENCY")
    w(" " + "-" * 66)
    w(f"   Timestep assumption           : {args.time_per_step_us:.2f} us/tick")
    w(f"   Ticks per inference           : {art['t_run']}  "
      f"(= T {art['T']} input + {art['n_layers']} pipeline-fill + pad)")
    w(f"   Latency per inference         : {latency_s*1e3:.4f} ms")
    w(f"   Throughput                    : {1.0/latency_s:.1f} inferences/s")
    w("")

    # --- energy / power ---
    w(" ENERGY & POWER (per inference)")
    w(" " + "-" * 74)
    w(f" {'Model':<16} {'Synaptic':<14} {'Neuron':<14} {'TOTAL':<14} {'Power':<12}")
    w(" " + "-" * 74)
    for gen in ('loihi1', 'loihi2', 'sparq'):
        syn, neu, tot = energy_breakdown(art, gen)
        power_w = tot / latency_s
        label = {'loihi1': 'Loihi 1 (pub.)', 'loihi2': 'Loihi 2 (est.)',
                 'sparq': 'SPARQ analytic'}[gen]
        w(f" {label:<16} {fmt_j(syn):<14} {fmt_j(neu):<14} {fmt_j(tot):<14} "
          f"{power_w*1e3:.3f} mW")
    w(" " + "-" * 74)
    w("")
    for gen in ('loihi1', 'loihi2', 'sparq'):
        w(f"   {gen:<8} {E_SYNOP[gen]*1e12:>6.2f} pJ/synop  "
          f"{E_NEURON[gen]*1e12:>5.1f} pJ/neuron   {SOURCE[gen]}")
    w("")

    # --- the constant-discrepancy warning ---
    _, _, e_l2 = energy_breakdown(art, 'loihi2')
    _, _, e_sp = energy_breakdown(art, 'sparq')
    ratio = e_l2 / e_sp if e_sp > 0 else float('inf')
    w(" !! CONSTANT DISCREPANCY")
    w(" " + "-" * 66)
    w(f"   SPARQ's analytical neuromorphic energy is {ratio:.0f}x LOWER than the")
    w(f"   Loihi 2 estimate for the SAME measured operation count.")
    w(f"   Cause: SPARQ uses 0.01 pJ/AC = the raw 8-bit ALU add. A real synaptic")
    w(f"   operation also pays weight memory read + routing + neuron state update,")
    w(f"   which is why published Loihi figures are 8-23.6 pJ/synop.")
    w(f"   => Do not present SPARQ's absolute neuromorphic mJ as hardware-realistic.")
    w(f"      Use it for RELATIVE comparisons only, or restate with Loihi constants.")
    w("")

    # --- mapping (indicative only) ---
    w(" HARDWARE MAPPING (indicative)")
    w(" " + "-" * 66)
    w(f"   Neurocores (8192 neurons/core): {cores}")
    w(f"   Chips (128 cores/chip)        : {chips}")
    w(f"   Weight memory @1 B/synapse    : {n_synapses/1024/1024:.2f} MiB")
    w(f"   NOTE: Lava's simulator does not enforce per-core fan-in limits. The")
    w(f"         widest layer alone would need splitting across many cores on real")
    w(f"         silicon, so treat this as indicative, not a placed-and-routed map.")
    w("")
    w("=" * 78)

    report = "\n".join(L)
    print(report)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nsaved -> {args.out}")


if __name__ == '__main__':
    main()
