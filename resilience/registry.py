"""
One place that knows how to load every (architecture, variant) pair used by the
resilience campaign, plus the metadata the metric needs.

Each architecture lives in its own directory with its own `config.py`, so this
module must be imported with that directory on PYTHONPATH and with the process
CWD set to it (the same convention lava_loihi_sim/sparq/* already uses).

Variants
--------
  ann    static FP32 network  -- b=32 bits/weight, no exits, no time axis
  qsnn   INT8 spiking network -- b=8,  no exits
  qdsnn  INT8 spiking + early exits and an RL exit policy -- b=8
"""
import os

import torch


class Spec:
    """A loaded model plus everything the metric needs to know about it."""

    def __init__(self, model, arch, variant, is_int8, is_spiking, is_dynamic,
                 T, n_params, n_bits, thresholds=None):
        self.model = model
        self.arch = arch
        self.variant = variant
        self.is_int8 = is_int8
        self.is_spiking = is_spiking
        self.is_dynamic = is_dynamic
        self.T = T
        self.n_params = n_params
        self.n_bits = n_bits
        self.thresholds = thresholds

    def __repr__(self):
        return (f"<{self.arch}/{self.variant} T={self.T} "
                f"{'INT8' if self.is_int8 else 'FP32'} "
                f"{self.n_params:,}w = {self.n_bits:,} bits>")


# --------------------------------------------------------------------------
CATALOG = {
    "mlp": {
        "ann":   dict(name="standard_mlp",              ckpt="results/standard_mlp.pth"),
        "qsnn":  dict(name="quantized_snn_mlp",         ckpt="results/quantized_snn_mlp.pth", T=4),
        "qdsnn": dict(name="branchy_quantized_snn_mlp", ckpt="results/branchy_quantized_snn_mlp.pth",
                      agent="results/branchy_quantized_snn_mlp_agent.pkl", T=4),
    },
    "lenet": {
        "ann":   dict(name="lenet",                  ckpt="results/standard_lenet.pth"),
        # The delay-aware retrains are the checkpoints the paper reports (Sec. 7).
        "qsnn":  dict(name="quantized_snn_lenet",    ckpt="results/quantized_snn_lenet_delay_int8.pth", T=16),
        "qdsnn": dict(name="branchy_quantized_snn",  ckpt="results/branchy_quantized_snn_delay_hr_int8.pth",
                      agent="results/branchy_quantized_snn_delay_hr_int8_agent.pkl",
                      T=16, reset="zero"),
    },
    "alexnet": {
        "ann":   dict(name="standard_alexnet", ckpt="results_willi/standard_alexnet.pth"),
        # AlexNet has no converted-INT8 QDSNN checkpoint, only the repaired
        # fake-quant one, so BOTH spiking variants use the fake-quant weights and
        # the simulated-INT8 fault path. Treating them identically matters more
        # than using the converted QSNN for one arm and not the other.
        "qsnn":  dict(name="quantized_snn_alexnet",
                      ckpt="results/quantized_snn_alexnet_qat_clean.pth",
                      T=32, sim_int8=True),
        # exit_after: AlexNet's heads attach after conv blocks 2 and 4, i.e.
        # backbone layer indices 1 and 3 of conv1..conv5,fc1..fc3.
        "qdsnn": dict(name="branchy_quantized_snn",
                      ckpt="results/branchy_quantized_snn_qat_clean.pth",
                      agent="results_willi/rl_agent_qat.pkl",
                      T=32, sim_int8=True, exit_after=[1, 3]),
    },
}


def _bit_budget(model, is_int8):
    from fault_inject import bit_budget
    return bit_budget(model, is_int8)


def prepare_input(spec, images):
    """Shape a batch of images the way this architecture's forward expects.

    The MLP models flatten internally only when handed a 2-D tensor; a 4-D
    image batch is mistaken for an already-encoded spike train.
    """
    if spec.arch == "mlp":
        return images.view(images.size(0), -1)
    return images


# --------------------------------------------------------------------------
def _load_lenet_int8(entry, variant, T):
    """Explicit INT8 skeleton rebuild.

    Not routed through LeNet's `evaluate.load_model` because the delay-aware
    QDSNN was retrained under HARD reset to match Loihi's stock LIF, and
    snntorch only honours that when it is applied to the constructed module.
    """
    import torch.ao.quantization as tq
    import snntorch as snn
    from config import config

    if variant == "qdsnn":
        from models.branchy_quantized_snn import BranchyQuantizedSNN
        model = BranchyQuantizedSNN(num_classes=config["num_classes"],
                                    time_window=T, beta=config["snn"]["beta"])
        model.fuse_model()
    else:
        from models.quantized_snn_lenet import (QuantizedSNNLeNet,
                                                prepare_quantized_snn_lenet_for_qat)
        model = QuantizedSNNLeNet(num_classes=config["num_classes"],
                                  beta=config["snn"]["beta"], time_steps=T)
        model = prepare_quantized_snn_lenet_for_qat(model)
        model.eval().cpu()
        model = tq.convert(model, inplace=False)
        model.load_state_dict(torch.load(entry["ckpt"], map_location="cpu"))
        return model.eval().cpu()

    from models.quantized_snn_lenet import get_quant_engine
    torch.backends.quantized.engine = get_quant_engine()
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
    model.load_state_dict(torch.load(entry["ckpt"], map_location="cpu"))  # strict
    return model.eval().cpu()


def _load_alexnet(entry, variant, T):
    """AlexNet's `evaluate.load_model` has no branch for the plain QSNN, and both
    spiking variants here are fake-quant rather than converted INT8, so the QAT
    skeletons are rebuilt directly (mirroring `evaluate_quantized_pair.py`)."""
    from config import config

    if variant == "qsnn":
        from models.quantized_snn_alexnet import (QuantizedSNNAlexNet,
                                                  prepare_quantized_snn_alexnet_for_qat)
        model = QuantizedSNNAlexNet(num_classes=config["num_classes"],
                                    beta=config["beta"], time_steps=T)
        model = prepare_quantized_snn_alexnet_for_qat(model)
    else:
        from models.branchy_quantized_snn import BranchyQuantizedSNN
        from models.quantized_snn_alexnet import get_quant_engine
        import torch.ao.quantization as tq
        import snntorch as snn
        model = BranchyQuantizedSNN(num_classes=config["num_classes"],
                                    beta=config["beta"], time_window=T)
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

    model.load_state_dict(torch.load(entry["ckpt"], map_location="cpu"))  # strict
    return model.eval().cpu()


def load(arch, variant, device="cpu", T=None):
    entry = CATALOG[arch][variant]
    T = T or entry.get("T")
    is_int8 = variant in ("qsnn", "qdsnn")
    sim_int8 = bool(entry.get("sim_int8"))
    is_spiking = is_int8
    is_dynamic = variant == "qdsnn"

    if not os.path.exists(entry["ckpt"]):
        raise FileNotFoundError(f"{arch}/{variant}: missing checkpoint {entry['ckpt']}")

    if arch == "lenet" and is_int8:
        model = _load_lenet_int8(entry, variant, T)
    elif arch == "alexnet" and is_int8:
        model = _load_alexnet(entry, variant, T)
    else:
        from evaluate import load_model
        try:                     # MLP signature
            model = load_model(entry["name"], entry["ckpt"], device, time_steps=T)
        except TypeError:        # LeNet / AlexNet signature
            model = load_model(entry["name"], entry["ckpt"], device, time_steps_from_args=T)
    model.eval()

    # Hard reset where the checkpoint was trained under it (Loihi-native).
    if entry.get("reset") == "zero":
        import snntorch as snn
        for _, m in model.named_modules():
            if isinstance(m, snn.Leaky):
                m.reset_mechanism = "zero"

    thresholds = None
    if is_dynamic:
        agent_path = entry.get("agent")
        if agent_path and os.path.exists(agent_path):
            from models.rl_environment import QLearningAgent
            agent = QLearningAgent(n_exits=model.n_exits)
            agent.load(agent_path)
            model.prepare_for_inference(rl_agent=agent)
        thresholds = list(model.confidence_thresholds)

    # A sim-INT8 model stores FP32 tensors but DEPLOYS as INT8, so its bit budget
    # must be counted at 8 bits/weight even though `bit_budget(is_int8=True)`
    # would look for converted quantized modules and find none.
    if sim_int8:
        from fault_inject import fp32_weight_tensors, quantize_weights_inplace
        quantize_weights_inplace(model)
        n_params = sum(p.numel() for _, p in fp32_weight_tensors(model))
        n_bits = n_params * 8
    else:
        n_params, n_bits = _bit_budget(model, is_int8)

    spec = Spec(model, arch, variant, is_int8, is_spiking, is_dynamic,
                T, n_params, n_bits, thresholds)
    spec.sim_int8 = sim_int8
    return spec
