"""
Power and Energy Estimation from Lava Simulation Results

Paper: Ahmed et al. (arXiv:2403.10173v4), Section 5, Tables 5 & 6
Hardware target: Intel Loihi 2

Estimation methodology:
- AC (Accumulate) operations counted from actual spike activity
- Energy estimated using two sets of per-operation constants:

  1. Loihi 1 (PUBLISHED — conservative upper bound):
     - 23.6 pJ/SynOp, 81 pJ/neuron update (active)
     - Source: Davies et al., "Loihi: A Neuromorphic Manycore Processor
       with On-Chip Learning," IEEE Micro, 2018, Table 2

  2. Loihi 2 (ESTIMATED — derived from system-level data):
     - ~8 pJ/SynOp (midpoint of 5-12 pJ range)
     - Derived from Hala Point: 380 TSOP/s at 2600 W → ~6.8 pJ/SynOp system-level
     - Application-level: ~10-12 pJ/OP (Isik et al., arXiv:2408.16096, 2024)
     - Intel has NOT published official per-operation energy for Loihi 2

NOTE: Table 5 of Ahmed et al. reports real Loihi 2 power/time for the
4-layer SNN block only. The often-cited 3.1 mJ number comes from the
paper's comparative hybrid-energy analysis in Sec. 5.2 and combines
1.63e9 MACs + 0.97e9 ACs under a 7 nm-style model; it is not a direct
measured Loihi 2 SNN energy. Our approach estimates from simulation +
published constants, so absolute values remain approximate.

Paper reference numbers:
- Real Loihi 2 SNN block: 1.73 ± 0.10 W, 2.06 ms/step
- Ahmed hybrid complexity: 1.63e9 MACs, 0.97e9 ACs
- Ahmed comparative hybrid energy: 3.1 mJ
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime


# ============================================================================
# Hardware Constants
# ============================================================================

# --- Loihi 1: PUBLISHED (Davies et al., IEEE Micro 2018, Table 2) ---
# Conservative upper bound for Loihi 2 estimation
LOIHI1_ENERGY_PER_SYNOP_PJ = 23.6   # min energy per synaptic spike op (0.75V)
LOIHI1_ENERGY_PER_NEURON_PJ = 81.0  # energy per active neuron update
LOIHI1_SOURCE = "Davies et al., IEEE Micro 2018, Table 2"

# --- Loihi 2: ESTIMATED (derived, NOT officially published by Intel) ---
# Midpoint estimate from:
#   - Hala Point system: 380 TSOP/s @ 2600W → ~6.8 pJ/SynOp (system-level)
#   - Isik et al. 2024 (arXiv:2408.16096): ~10-12 pJ/OP (application-level)
#   - Loihi 2 is Intel 4 process (vs Loihi 1 Intel 14nm), ~2-4x improvement
LOIHI2_ENERGY_PER_SYNOP_PJ = 8.0    # estimated midpoint of 5-12 pJ range
LOIHI2_ENERGY_PER_NEURON_PJ = 30.0  # estimated (~2.5x improvement over Loihi 1)
LOIHI2_SOURCE = "Estimated from Hala Point system data + Isik et al. 2024"

# --- Loihi 2 hardware specs (from Intel documentation) ---
NEURONS_PER_CORE = 8192  # max neurons per neurocore
CORES_PER_CHIP = 128     # neurocores per Loihi 2 chip
STATIC_POWER_PER_CHIP_W = 0.05  # ~50 mW idle per chip (estimated)

# Memory
WEIGHT_BYTES_INT8 = 1    # int8 = 1 byte per weight
NEURON_STATE_BYTES = 2   # 16-bit membrane potential


# ============================================================================
# AC Calculator
# ============================================================================

def compute_acs_per_layer(results: Dict) -> Dict[str, Dict]:
    """
    Compute Accumulate operations (ACs) for each SNN layer.

    For spike-based computation:
        ACs_layer = Σ_t (input_spikes[t]) × k² × C_out

    Unlike MACs, ACs only count operations where the input is a spike (1),
    since multiplying by 0 (no spike) is skipped on neuromorphic hardware.

    Args:
        results: Per-layer simulation results from LavaSNNSimulator

    Returns:
        ac_info: Per-layer AC counts and breakdowns
    """
    ac_info = {}

    for name, layer in results.items():
        k = layer['kernel_size']
        c_out = layer['channels']
        acs_per_step = layer['acs_per_step']
        total_acs = layer['total_acs']

        total_neurons = layer['total_neurons']
        max_acs_per_step = total_neurons * k * k * c_out

        ac_info[name] = {
            'total_acs': total_acs,
            'acs_per_step': acs_per_step,
            'sparsity': 1.0 - layer['firing_rate'],
            'theoretical_max_acs': max_acs_per_step * len(acs_per_step),
        }

    return ac_info


# ============================================================================
# Energy Estimator (dual Loihi 1 / Loihi 2 estimates)
# ============================================================================

def estimate_energy_mj(
    results: Dict,
    num_time_steps: int = 10,
    chip_generation: str = "both",
) -> Dict:
    """
    Estimate energy consumption from simulation results.

    Computes energy using both Loihi 1 (published) and Loihi 2 (estimated)
    per-operation constants.

    Energy components:
    1. Synaptic (AC) energy: ACs × energy_per_SynOp
    2. Neuron update energy: neurons × T × energy_per_neuron
    3. Static energy: chips × time × static_power

    Args:
        results: Per-layer simulation results
        num_time_steps: T (default 10)
        chip_generation: "loihi1", "loihi2", or "both"

    Returns:
        energy: Breakdown per layer and total, in millijoules
    """
    configs = {}
    if chip_generation in ("loihi1", "both"):
        configs["loihi1"] = (LOIHI1_ENERGY_PER_SYNOP_PJ, LOIHI1_ENERGY_PER_NEURON_PJ)
    if chip_generation in ("loihi2", "both"):
        configs["loihi2"] = (LOIHI2_ENERGY_PER_SYNOP_PJ, LOIHI2_ENERGY_PER_NEURON_PJ)

    hw = estimate_hardware_requirements(results)
    num_chips = hw['total']['chips']
    time_per_inference_s = estimate_time_per_step_ms(results) * 1e-3

    all_energy = {}

    for gen_name, (synop_pj, neuron_pj) in configs.items():
        energy = {}
        total_ac_energy_j = 0.0
        total_neuron_energy_j = 0.0

        for name, layer in results.items():
            ac_energy_j = layer['total_acs'] * synop_pj * 1e-12
            total_ac_energy_j += ac_energy_j

            n_neurons = layer['total_neurons']
            neuron_energy_j = n_neurons * num_time_steps * neuron_pj * 1e-12
            total_neuron_energy_j += neuron_energy_j

            layer_energy_j = ac_energy_j + neuron_energy_j
            energy[name] = {
                'ac_energy_mj': ac_energy_j * 1e3,
                'neuron_energy_mj': neuron_energy_j * 1e3,
                'total_energy_mj': layer_energy_j * 1e3,
            }

        static_energy_j = num_chips * STATIC_POWER_PER_CHIP_W * time_per_inference_s
        total_energy_j = total_ac_energy_j + total_neuron_energy_j + static_energy_j

        energy['total'] = {
            'ac_energy_mj': total_ac_energy_j * 1e3,
            'neuron_energy_mj': total_neuron_energy_j * 1e3,
            'static_energy_mj': static_energy_j * 1e3,
            'total_energy_mj': total_energy_j * 1e3,
        }

        all_energy[gen_name] = energy

    if chip_generation == "both":
        return all_energy
    else:
        return all_energy[chip_generation]


def estimate_power_w(
    results: Dict,
    num_time_steps: int = 10,
    chip_generation: str = "both",
) -> Dict:
    """
    Estimate power consumption in Watts.

    Power = Energy / inference_time

    Args:
        results: Per-layer simulation results
        num_time_steps: T (default 10)
        chip_generation: "loihi1", "loihi2", or "both"

    Returns:
        power: Dict with power in Watts and breakdown
    """
    energy_all = estimate_energy_mj(results, num_time_steps, chip_generation="both")
    time_ms = estimate_time_per_step_ms(results)
    time_s = time_ms * 1e-3

    power = {'time_per_step_ms': time_ms}

    for gen_name, energy in energy_all.items():
        total_energy_j = energy['total']['total_energy_mj'] * 1e-3
        power_w = total_energy_j / time_s if time_s > 0 else 0.0
        power[gen_name] = {
            'power_w': power_w,
            'total_energy_mj': energy['total']['total_energy_mj'],
        }

    return power


def estimate_time_per_step_ms(results: Dict) -> float:
    """
    Estimate inference time per step on Loihi 2.

    Based on empirical scaling from the paper's measurement:
    Paper: ~0.96M SNN params → 2.06 ms/step on Loihi 2

    Args:
        results: Per-layer simulation results

    Returns:
        time_ms: Estimated time per inference step in milliseconds
    """
    total_neurons = sum(layer['total_neurons'] for layer in results.values())

    # Empirical scaling from paper:
    # 0.96M params, 2.06 ms → ~2.15 μs per 1K neurons per timestep
    T = 10
    time_per_step_us = (total_neurons / 1000.0) * 2.15
    time_ms = time_per_step_us * T / 1000.0

    # Clamp to reasonable range
    time_ms = max(0.5, min(time_ms, 10.0))

    return time_ms


# ============================================================================
# Hardware Requirements Estimator
# ============================================================================

def estimate_hardware_requirements(results: Dict) -> Dict:
    """
    Estimate Loihi 2 hardware requirements.

    Computes:
    - Weight memory (int8 quantization)
    - Neuron state memory (16-bit membrane potential)
    - Neurocores needed (8192 neurons/core)
    - Chips needed (128 cores/chip)

    Args:
        results: Per-layer simulation results

    Returns:
        hw: Per-layer and total hardware requirements
    """
    hw = {}
    total_weight_bytes = 0
    total_neuron_bytes = 0
    total_cores = 0

    for name, layer in results.items():
        k = layer['kernel_size']
        c_in = _get_input_channels(name, results)
        c_out = layer['channels']
        n_neurons = layer['total_neurons']

        weight_bytes = c_out * c_in * k * k * WEIGHT_BYTES_INT8
        total_weight_bytes += weight_bytes

        neuron_bytes = n_neurons * NEURON_STATE_BYTES
        total_neuron_bytes += neuron_bytes

        cores = int(np.ceil(n_neurons / NEURONS_PER_CORE))
        total_cores += cores

        hw[name] = {
            'weight_bytes': weight_bytes,
            'weight_kb': weight_bytes / 1024,
            'neuron_state_bytes': neuron_bytes,
            'neuron_state_kb': neuron_bytes / 1024,
            'neurocores': cores,
            'neurons': n_neurons,
        }

    total_chips = int(np.ceil(total_cores / CORES_PER_CHIP))

    hw['total'] = {
        'weight_bytes': total_weight_bytes,
        'weight_kb': total_weight_bytes / 1024,
        'neuron_state_bytes': total_neuron_bytes,
        'neuron_state_kb': total_neuron_bytes / 1024,
        'neurocores': total_cores,
        'chips': total_chips,
    }

    return hw


def _get_input_channels(block_name: str, results: Optional[Dict] = None) -> int:
    """Get input channels for a given SNN block.

    For the base model (block1-4), uses a hardcoded channel map.
    For NAS models (snn_block_0, snn_block_1, ...), infers from the
    results dict by looking at the previous layer's output channels,
    or falls back to 256 if unavailable.

    Args:
        block_name: Layer name (e.g., 'block1' or 'snn_block_0')
        results: Optional simulation results dict for NAS inference
    """
    # Base model hardcoded map
    ch_map = {
        'block1': 2,
        'block2': 64,
        'block3': 128,
        'block4': 256,
    }
    if block_name in ch_map:
        return ch_map[block_name]

    # NAS model: infer from results ordering
    if results is not None and block_name.startswith('snn_block_'):
        idx = int(block_name.split('_')[-1])
        if idx == 0:
            return 2  # First SNN block always takes 2-channel event tensor
        # Look up previous block's output channels
        prev_name = f'snn_block_{idx - 1}'
        if prev_name in results:
            return results[prev_name]['channels']

    return 256


# ============================================================================
# Verification: Compare with SpikingJelly model
# ============================================================================

def verify_spike_agreement(
    lava_results: Dict,
    spikingjelly_results: Dict,
    tolerance: float = 0.10,
) -> Dict:
    """
    Compare firing rates between Lava simulation and SpikingJelly model.

    After int8 quantization, firing rates should be within 5-10%.

    Args:
        lava_results: Results from Lava/PyTorch LIF simulation
        spikingjelly_results: Results from SpikingJelly forward pass
        tolerance: Maximum acceptable relative difference (default 10%)

    Returns:
        comparison: Per-layer firing rate comparison
    """
    comparison = {}

    for name in lava_results:
        if name not in spikingjelly_results:
            continue

        lava_rate = lava_results[name]['firing_rate']
        sj_rate = spikingjelly_results[name]['firing_rate']

        if sj_rate > 0:
            rel_diff = abs(lava_rate - sj_rate) / sj_rate
        else:
            rel_diff = abs(lava_rate - sj_rate)

        comparison[name] = {
            'lava_firing_rate': lava_rate,
            'spikingjelly_firing_rate': sj_rate,
            'relative_difference': rel_diff,
            'within_tolerance': rel_diff <= tolerance,
        }

    return comparison


# ============================================================================
# Report Generator
# ============================================================================

def generate_power_report(
    results: Dict,
    num_samples: int = 100,
    num_time_steps: int = 10,
) -> str:
    """
    Generate a comprehensive power/energy report matching paper Tables 5 & 6.

    Reports both Loihi 1 (published upper bound) and Loihi 2 (estimated)
    energy numbers with clear source attribution.

    Args:
        results: Per-layer simulation results from LavaSNNSimulator
        num_samples: Number of samples used in simulation
        num_time_steps: T (default 10)

    Returns:
        report: Formatted report string
    """
    energy_all = estimate_energy_mj(results, num_time_steps, chip_generation="both")
    power = estimate_power_w(results, num_time_steps)
    hw = estimate_hardware_requirements(results)

    total_acs = sum(results[n]['total_acs'] for n in results)

    # Get layer names dynamically (works for both base and NAS models)
    layer_names = [n for n in results.keys()]
    num_snn_layers = len(layer_names)

    lines = []
    w = lines.append

    w("=" * 76)
    w(f" SNN BACKBONE -- Loihi 2 Neuromorphic Simulation Results")
    w(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f" Samples: {num_samples}, Timesteps: {num_time_steps}")
    w(f" SNN Layers: {num_snn_layers}")
    w("=" * 76)
    w("")
    w(" !!! PAPER REPORTING RULE -- LOIHI 1 COLUMNS ARE LEGACY -- DO NOT CITE !!!")
    w(" Per CLAUDE.md energy-reporting rule: the journal/paper reports a SINGLE")
    w(" heterogeneous Loihi 2 (2 pJ/AC) + GPU (4.6 pJ/MAC) number per config.")
    w(" Loihi 1 columns/rows below are retained only for historical comparison")
    w(" against earlier ablations; they MUST NOT appear in any paper figure or")
    w(" table. Use the 'Loihi 2 (est.)' rows only.")
    w("")
    w(" METHODOLOGY:")
    w("   Spike activity measured via fixed-point LIF simulation (Loihi2SimCfg-")
    w("   equivalent). Energy estimated using per-operation constants -- see")
    w("   source notes below each table. The paper measured on real Loihi 2")
    w("   hardware; our results are simulation-based estimates.")

    # Table A: Power and Time (cf. Paper Table 5)
    w("")
    w(" Table A: Power and Time (cf. Paper Table 5)")
    w(" " + "-" * 68)
    w(f" {'Estimate':<20} {'# Chips':<10} {'Power [W]':<15} {'Time/Step [ms]':<15}")
    w(" " + "-" * 68)
    w(f" {'Loihi 1 (upper)':<20} {hw['total']['chips']:<10} "
      f"{power['loihi1']['power_w']:<15.2f} {power['time_per_step_ms']:<15.2f}")
    w(f" {'Loihi 2 (est.)':<20} {hw['total']['chips']:<10} "
      f"{power['loihi2']['power_w']:<15.2f} {power['time_per_step_ms']:<15.2f}")
    w(f" {'Paper (Table 5)':<20} {'6':<10} {'1.73 +/- 0.10':<15} {'2.06':<15}")
    w(" " + "-" * 68)

    # Table B: Computational Complexity (cf. Paper Table 6)
    w("")
    w(" Table B: Computational Complexity (cf. Paper Table 6)")
    w(" " + "-" * 72)
    w(f" {'Component':<15} {'ACs':<15} {'Energy [mJ]':<16} {'Energy [mJ]':<16}")
    w(f" {'':<15} {'':<15} {'(Loihi 1 pub.)':<16} {'(Loihi 2 est.)':<16}")
    w(" " + "-" * 72)

    for name in layer_names:
        acs = results[name]['total_acs']
        e1 = energy_all['loihi1'][name]['total_energy_mj']
        e2 = energy_all['loihi2'][name]['total_energy_mj']
        w(f" SNN {name:<11} {_fmt_sci(acs):<15} {e1:<16.3f} {e2:<16.3f}")

    w(" " + "-" * 72)
    e1_total = energy_all['loihi1']['total']['total_energy_mj']
    e2_total = energy_all['loihi2']['total']['total_energy_mj']
    w(f" {'SNN Total':<15} {_fmt_sci(total_acs):<15} {e1_total:<16.3f} {e2_total:<16.3f}")
    w(f" {'Ahmed hybrid*':<15} {'0.97e9':<15} {'3.1 mJ':<16} {'3.1 mJ':<16}")
    w(" " + "-" * 72)

    # Table C: Per-Layer Spike Activity
    w("")
    w(" Table C: Per-Layer Spike Activity (from simulation)")
    w(" " + "-" * 68)
    w(f" {'Layer':<14} {'Firing Rate':<14} {'Spikes/step':<14} "
      f"{'ACs/step':<14} {'dv (from tau)':<12}")
    w(" " + "-" * 68)

    for name in layer_names:
        layer = results[name]
        mean_spikes_per_step = np.mean(layer['spikes_per_step'])
        mean_acs_per_step = np.mean(layer['acs_per_step'])
        w(f" {name:<14} {layer['firing_rate']:<14.4f} "
          f"{_fmt_sci(mean_spikes_per_step):<14} "
          f"{_fmt_sci(mean_acs_per_step):<14} "
          f"{layer['dv']:<12.4f}")

    w(" " + "-" * 68)

    # Table D: Hardware Requirements
    w("")
    w(" Table D: Hardware Requirements")
    w(" " + "-" * 68)
    w(f" {'Layer':<14} {'Weights(KB)':<14} {'Neuron(KB)':<14} "
      f"{'Neurocores':<12} {'Chips':<8}")
    w(" " + "-" * 68)

    for name in layer_names:
        if name not in hw:
            continue
        h = hw[name]
        w(f" {name:<14} {h['weight_kb']:<14.1f} {h['neuron_state_kb']:<14.1f} "
          f"{h['neurocores']:<12} {'--':<8}")

    w(" " + "-" * 68)
    h_total = hw['total']
    w(f" {'Total':<14} {h_total['weight_kb']:<14.1f} "
      f"{h_total['neuron_state_kb']:<14.1f} "
      f"{h_total['neurocores']:<12} {h_total['chips']:<8}")
    w(" " + "-" * 68)

    # Energy breakdown
    w("")
    w(" Table E: Energy Breakdown")
    w(" " + "-" * 56)
    w(f" {'Component':<25} {'Loihi 1 [mJ]':<15} {'Loihi 2 [mJ]':<15}")
    w(" " + "-" * 56)
    e1 = energy_all['loihi1']['total']
    e2 = energy_all['loihi2']['total']
    w(f" {'Synaptic (ACs)':<25} {e1['ac_energy_mj']:<15.4f} {e2['ac_energy_mj']:<15.4f}")
    w(f" {'Neuron updates':<25} {e1['neuron_energy_mj']:<15.4f} {e2['neuron_energy_mj']:<15.4f}")
    w(f" {'Static (leakage)':<25} {e1['static_energy_mj']:<15.4f} {e2['static_energy_mj']:<15.4f}")
    w(f" {'Total':<25} {e1['total_energy_mj']:<15.4f} {e2['total_energy_mj']:<15.4f}")
    w(f" {'* Ahmed hybrid total: 3.1 mJ':<25}")
    w(" " + "-" * 56)

    # Sources
    w("")
    w(" ENERGY CONSTANT SOURCES:")
    w(f"   Loihi 1 (published, conservative upper bound):")
    w(f"     {LOIHI1_ENERGY_PER_SYNOP_PJ} pJ/SynOp, "
      f"{LOIHI1_ENERGY_PER_NEURON_PJ} pJ/neuron update")
    w(f"     Source: {LOIHI1_SOURCE}")
    w(f"   Loihi 2 (estimated, NOT officially published by Intel):")
    w(f"     ~{LOIHI2_ENERGY_PER_SYNOP_PJ} pJ/SynOp (range: 5-12 pJ), "
      f"~{LOIHI2_ENERGY_PER_NEURON_PJ} pJ/neuron update")
    w(f"     Derived from: {LOIHI2_SOURCE}")

    # Verification notes
    w("")
    w(" VERIFICATION:")
    w(f"   Total ACs: {_fmt_sci(total_acs)} (Ahmed hybrid: 0.97e9)")
    acs_ratio = total_acs / 0.97e9 if total_acs > 0 else 0
    w(f"   ACs ratio vs paper: {acs_ratio:.2f}x")
    w(f"   Power (Loihi 1): {power['loihi1']['power_w']:.2f} W | "
      f"Power (Loihi 2 est.): {power['loihi2']['power_w']:.2f} W | "
      f"Paper Table 5: 1.73 W")
    w(f"   Expected firing rate range: 5-30%, decreasing in deeper layers")
    for name in layer_names:
        fr = results[name]['firing_rate']
        status = "OK" if 0.01 <= fr <= 0.50 else "UNUSUAL"
        w(f"   {name} firing rate: {fr:.2%} [{status}]")

    w("")
    w("=" * 76)

    return "\n".join(lines)


def _fmt_sci(value: float) -> str:
    """Format number in scientific notation (e.g., 2.34e8)."""
    if abs(value) < 1:
        return f"{value:.2f}"
    exp = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    mantissa = value / (10 ** exp) if exp != 0 else value
    return f"{mantissa:.2f}e{exp}"


# ============================================================================
# Standalone usage
# ============================================================================

if __name__ == "__main__":
    print("Power Estimation Module - Demo with synthetic data")
    print("=" * 60)

    # Synthetic results mimicking expected SNN backbone activity
    demo_results = {
        'block1': {
            'total_spikes': 1.5e6,
            'spikes_per_step': [1.5e5] * 10,
            'firing_rate': 0.22,
            'total_acs': 3.5e8,
            'acs_per_step': [3.5e7] * 10,
            'total_neurons': 120 * 152 * 64,
            'tau': 2.0,
            'dv': 0.5,
            'spatial': (120, 152),
            'channels': 64,
            'kernel_size': 3,
            'stride': 2,
        },
        'block2': {
            'total_spikes': 5.0e5,
            'spikes_per_step': [5.0e4] * 10,
            'firing_rate': 0.15,
            'total_acs': 2.8e8,
            'acs_per_step': [2.8e7] * 10,
            'total_neurons': 60 * 76 * 128,
            'tau': 2.1,
            'dv': 0.524,
            'spatial': (60, 76),
            'channels': 128,
            'kernel_size': 3,
            'stride': 2,
        },
        'block3': {
            'total_spikes': 2.0e5,
            'spikes_per_step': [2.0e4] * 10,
            'firing_rate': 0.10,
            'total_acs': 2.0e8,
            'acs_per_step': [2.0e7] * 10,
            'total_neurons': 30 * 38 * 256,
            'tau': 2.2,
            'dv': 0.545,
            'spatial': (30, 38),
            'channels': 256,
            'kernel_size': 3,
            'stride': 2,
        },
        'block4': {
            'total_spikes': 1.0e5,
            'spikes_per_step': [1.0e4] * 10,
            'firing_rate': 0.07,
            'total_acs': 1.5e8,
            'acs_per_step': [1.5e7] * 10,
            'total_neurons': 30 * 38 * 256,
            'tau': 2.3,
            'dv': 0.565,
            'spatial': (30, 38),
            'channels': 256,
            'kernel_size': 3,
            'stride': 1,
        },
    }

    report = generate_power_report(demo_results, num_samples=100)
    print(report)
