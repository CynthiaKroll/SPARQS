"""
Threat-aware energy / latency accounting.

Per the locked scope, nothing is re-simulated in Lava. Instead the *inputs* to
the already-calibrated Loihi-2 cost model are re-measured under threat (spike
counts, exit distribution, exit ticks) and the same equation is re-evaluated:

    E   = N_AC_hw · E_synop + N_neurons · t_run · E_neuron       (main.tex Eq. energy_loihi)
    N_AC_hw = Σ_l  (spikes entering layer l) · fanout_l
    t_run   = T_used + pipeline_depth
    latency = t_run · t_tick ;   P = E / latency

Constants and per-layer `ac_fanout` are taken from the existing artifacts
(`lava_loihi_sim/sparq/power_report.py`, `lava_loihi_sim/exports/*.npz`) so the
threat numbers are on exactly the same footing as the published clean ones.

The metric consumes only the RATIO E_lambda/E_0, so a common scale factor applied
to both terms cancels exactly. The synop/neuron *balance* does not: E is a sum of
two terms carrying different constants, so changing their relative value shifts
the ratio. That effect is second-order here because the synaptic term dominates,
but the defensible claim is "insensitive to a common rescaling", not "independent
of the constants". Absolutes are reported as estimates, never as measurements.

Nothing in this module runs on neuromorphic hardware or in Lava. It re-evaluates
the calibrated cost model on quantities measured by ordinary PyTorch inference.
"""
import os

import numpy as np
import snntorch as snn
import torch

# --- Loihi-2 constants (mirror of power_report.py; see main.tex Tab. loihi_constants)
E_SYNOP = 8.0e-12      # J per synaptic operation   (ESTIMATED, not Intel-published)
E_NEURON = 30.0e-12    # J per neuron state update  (ESTIMATED)
T_TICK = 10e-6         # s per algorithmic timestep

_EXPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "lava_loihi_sim", "exports")

# npz carrying the topology (fanout, exit points, layer dims) for each arch
_TOPOLOGY = {
    ("lenet", "qdsnn"): "lenet_qdsnn_delay_hr_T16.npz",
    ("lenet", "qsnn"):  "lenet_qsnn_delay_T16.npz",
    ("mlp", "qdsnn"):   "mlp_qdsnn.npz",
    ("mlp", "qsnn"):    "mlp_qsnn.npz",
}


class Topology:
    """Per-layer fan-out and exit geometry, read from the Loihi export."""

    def __init__(self, fanout, out_dims, exit_after, n_layers):
        self.fanout = np.asarray(fanout, float)
        self.out_dims = np.asarray(out_dims, int)
        self.exit_after = np.asarray(exit_after, int) if exit_after is not None else None
        self.n_layers = int(n_layers)
        # spiking neurons = every layer output except the non-spiking readout
        self.n_neurons = int(self.out_dims[:-1].sum())

    def depth_of_exit(self, exit_idx, n_exits):
        """Number of backbone layers executed by the path taken at `exit_idx`."""
        if self.exit_after is None or exit_idx > n_exits:
            return self.n_layers
        return int(self.exit_after[exit_idx - 1]) + 1


def derive_topology(model, sample_input, exit_after=None, T=1):
    """Build a Topology by profiling the model, for architectures with no Loihi export.

    Mean fan-out of a layer is its MAC count divided by the number of input
    elements it consumes: for a Linear that is exactly the output width, and for
    a Conv it is the number of output accumulations a single input spike drives,
    which is the quantity the synaptic-operation count needs. Shapes are read
    from a single forward pass rather than assumed.
    """
    import torch.nn as nn

    recs = []

    def hook(m, inp, out):
        x = inp[0]
        if not torch.is_tensor(x) or not torch.is_tensor(out):
            return
        in_elem = x[0].numel() if x.dim() > 1 else x.numel()
        out_elem = out[0].numel() if out.dim() > 1 else out.numel()
        if isinstance(m, nn.Conv2d):
            macs = (m.kernel_size[0] * m.kernel_size[1] * m.in_channels //
                    max(m.groups, 1)) * out_elem
        else:
            macs = in_elem * out_elem
        recs.append((macs / max(in_elem, 1), out_elem))

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    if not recs:
        raise RuntimeError("derive_topology: no Conv2d/Linear layers were executed")

    # A spiking forward loops over T timesteps, so every layer fires its hook T
    # times and the raw record list is the per-tick sequence repeated T times.
    # We want ONE entry per layer: spike counts are already summed over time by
    # SpikeCounter, and t_run must be T + n_layers, not T + T*n_layers.
    if T > 1 and len(recs) % T == 0:
        recs = recs[: len(recs) // T]

    fanout = [r[0] for r in recs]
    out_dims = [r[1] for r in recs]
    return Topology(fanout, out_dims, exit_after, len(recs))


def load_topology(arch, variant):
    key = _TOPOLOGY.get((arch, variant)) or _TOPOLOGY.get((arch, "qdsnn"))
    if key is None:
        return None          # caller falls back to derive_topology
    path = os.path.join(_EXPORTS, key)
    d = np.load(path, allow_pickle=True)
    n_layers = int(d["n_layers"])
    out_dims = [d[f"w{k}"].shape[0] for k in range(n_layers)]
    # Dense layers carry no `ac_fanout`; for them fan-out IS the output width.
    # For unrolled convolutions it must be the mean NONZERO fan-out, or the
    # count is overestimated by more than an order of magnitude.
    fanout = (d["ac_fanout"].astype(float) if "ac_fanout" in d.files
              else np.asarray(out_dims, float))
    exit_after = d["exit_after"].astype(int) if "exit_after" in d.files else None
    return Topology(fanout, out_dims, exit_after, n_layers)


# --------------------------------------------------------------------------
class SpikeCounter:
    """Context manager: per-LIF emitted-spike counts over a whole forward pass.

    Counts `out[0]` (spikes), never the membrane -- reading `out[1]` was the
    original SPARQ counter bug and made the sparsity claim unmeasurable.
    Gated (early-exited) samples contribute zero spikes, so a batch total is
    already exit-weighted.
    """

    def __init__(self, model):
        self.model = model
        self.counts = None
        self._hooks = []

    def __enter__(self):
        lifs = [m for m in self.model.modules() if isinstance(m, snn.Leaky)]
        self.counts = [0.0] * len(lifs)

        def make(i):
            def hook(_m, _inp, out):
                spk = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(spk):
                    self.counts[i] += float((spk > 0).sum().item())
            return hook

        self._hooks = [m.register_forward_hook(make(i)) for i, m in enumerate(lifs)]
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        return False


# --------------------------------------------------------------------------
def synaptic_ops(topo, input_spikes, lif_spikes):
    """N_AC_hw = Σ_l (spikes entering layer l) · fanout_l.

    Layer 0 is driven by the encoded input spikes; layer k>0 by LIF k-1. This
    counts spikes weighted by fan-out (what the chip accumulates), NOT spike
    events -- the two differ by orders of magnitude for an unrolled convolution.
    """
    entering = [float(input_spikes)] + [float(s) for s in lif_spikes]
    n = min(len(entering), len(topo.fanout))
    return float(np.sum(np.asarray(entering[:n]) * topo.fanout[:n]))


def run_cost(topo, input_spikes, lif_spikes, n_samples, exit_pts=None,
             exit_ts=None, T=None, n_exits=2):
    """Per-sample energy / latency / power under whatever condition was just run."""
    acs = synaptic_ops(topo, input_spikes, lif_spikes) / max(n_samples, 1)

    if exit_pts is not None and exit_ts is not None:
        # Path decomposition: a sample is charged only for the layers and ticks
        # it actually used (main.tex Eq. pathdecomp), with the exit distribution
        # measured UNDER THREAT rather than clean.
        depths = np.array([topo.depth_of_exit(int(e), n_exits)
                           for e in np.asarray(exit_pts)], float)
        ticks = np.asarray(exit_ts, float) + depths
        t_run = float(ticks.mean())
        frac = depths / topo.n_layers
        neurons = float(topo.n_neurons * frac.mean())
    else:
        t_run = float((T or 0) + topo.n_layers)
        neurons = float(topo.n_neurons)

    e_syn = acs * E_SYNOP
    e_neu = neurons * t_run * E_NEURON
    energy = e_syn + e_neu
    latency = t_run * T_TICK
    return dict(acs_per_sample=acs, t_run=t_run, n_neurons_live=neurons,
                energy_j=energy, latency_s=latency,
                power_w=(energy / latency if latency > 0 else float("nan")))


def exposure(topo, exit_pts, n_params, bits_per_weight, n_exits=2):
    """Fault exposure X(c) = n_params · b · phi, phi = exit-weighted exercised fraction.

    phi is approximated by the fraction of backbone LAYERS a sample executes.
    That understates the QDSNN's advantage whenever the skipped layers are the
    parameter-heavy ones (true for the MLP, whose parameters concentrate in the
    deep layers), so the reported figure is conservative.
    """
    if exit_pts is None:
        return float(n_params * bits_per_weight), 1.0
    depths = np.array([topo.depth_of_exit(int(e), n_exits)
                       for e in np.asarray(exit_pts)], float)
    phi = float((depths / topo.n_layers).mean())
    return float(n_params * bits_per_weight * phi), phi
