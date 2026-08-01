import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import snntorch as snn
from snntorch import surrogate

from config import config

spike_grad = surrogate.atan()


def get_quant_engine():
    """Pick a quantization engine that this machine actually supports.

    'fbgemm' was hardcoded (in config and in the old prepare/convert helpers),
    but it is not available on every build -- this box only exposes 'onednn',
    which made prepare_qat/convert fail at runtime."""
    supported = list(torch.backends.quantized.supported_engines)
    for preferred in ('fbgemm', 'qnnpack', 'onednn'):
        if preferred in supported:
            return preferred
    return supported[0]


def encode_poisson(x, num_steps, gain=None):
    """Canonical Poisson RATE encoding, shared by the QSNN and the QDSNN and
    matched by the AC/spike/memory counters (same formula as the 5-layer-MLP
    module's utils.encode_poisson_rate).

    Per-timestep spike probability p = clamp(x_norm * gain, 0, 1), where x_norm
    is min-max normalized to [0, 1]. Expected spikes per pixel = T * p, so more
    timesteps give a better rate estimate (standard rate coding).

    Do NOT divide p by num_steps (as the previous version did): that makes the
    expected TOTAL spike count <= gain*x regardless of T, so T becomes
    meaningless and the rate code is destroyed. Min-max normalizing first also
    handles the mean/std-standardized (partly negative) MNIST input, which a
    bare clamp(x, 0, 1) would truncate."""
    if gain is None:
        gain = config.get("poisson_gain", 1.0)
    x = x.float()
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    p = torch.clamp(x_norm * gain, 0, 1)
    p = p.unsqueeze(0).expand(num_steps, *x.shape)
    return (torch.rand_like(p) < p).float()


class QuantizedSNNLeNet(nn.Module):
    """
    Hybrid Quantized SNN LeNet: Conv/Linear layers quantized to INT8 via QAT,
    LIF (snn.Leaky) spiking layers kept in FP32 with explicit membrane state.

    A QuantStub is inserted before EVERY quantized layer. The old version had a
    single shared QuantStub at the input only, so after conversion the INT8
    Linear layers would receive the FP32 (dequantized) LIF output and error out
    -- i.e. the model could not actually run once quantized. Order is
    conv/linear -> BN -> (dequant) -> LIF, matching the QDSNN backbone exactly so
    the QSNN vs QDSNN comparison is like-for-like.
    """

    def __init__(self, num_classes=10, beta=0.95, thresholds=None, time_steps=None):
        super().__init__()
        if time_steps is None:
            time_steps = config["num_steps"]
        self.beta = beta
        self.time_steps = time_steps

        def make_lif(key):
            th = thresholds.get(key, 1.0) if thresholds else 1.0
            return snn.Leaky(beta=beta, threshold=th, spike_grad=spike_grad,
                             init_hidden=False, reset_mechanism='subtract')

        self.quant = QuantStub()      # model entry (encoded input spikes)
        self.dequant = DeQuantStub()  # model exit

        # --- Feature backbone: conv -> BN -> LIF -> AvgPool, QuantStub before each conv ---
        self.features_seq = nn.ModuleList([
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),   # 0
            nn.BatchNorm2d(6),                                     # 1
            make_lif('features.2'),                                # 2 -> FP32 out
            nn.AvgPool2d(2, 2),                                    # 3
            QuantStub(),                                           # 4 (quantize before conv2)
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 5
            nn.BatchNorm2d(16),                                    # 6
            make_lif('features.6'),                                # 7 -> FP32 out
            nn.AvgPool2d(2, 2),                                    # 8
            QuantStub(),                                           # 9 (quantize before fc1)
        ])

        self.classifier_seq = nn.ModuleList([
            nn.Flatten(),                                          # 0
            nn.Linear(16 * 5 * 5, 120),                            # 1
            nn.BatchNorm1d(120),                                   # 2
            make_lif('classifier.3'),                              # 3 -> FP32 out
            QuantStub(),                                           # 4 (quantize before fc2)
            nn.Linear(120, 84),                                    # 5
            nn.BatchNorm1d(84),                                    # 6
            make_lif('classifier.6'),                              # 7 -> FP32 out
            QuantStub(),                                           # 8 (quantize before fc3)
            nn.Linear(84, num_classes),                            # 9
        ])

    # ------------------------------------------------------------------ utils
    def fuse_model(self):
        """Fuse each Conv/Linear + BatchNorm pair by module name (required for QAT).

        In TRAIN mode fuse_modules_qat keeps BatchNorm LIVE (intrinsic ConvBn2d/
        LinearBn1d, folded only at convert); eval-mode fusion before training
        would fold the freshly-initialized BN away and train without
        normalization. Eval mode (INT8 skeleton rebuild) keeps static folding."""
        pairs = [['features_seq.0', 'features_seq.1'],
                 ['features_seq.5', 'features_seq.6'],
                 ['classifier_seq.1', 'classifier_seq.2'],
                 ['classifier_seq.5', 'classifier_seq.6']]
        if self.training:
            tq.fuse_modules_qat(self, pairs, inplace=True)
        else:
            tq.fuse_modules(self, pairs, inplace=True)
        return self

    @staticmethod
    def _to_fp32(x):
        """LIF neurons run in FP32; dequantize the (possibly INT8) activation."""
        if hasattr(x, 'is_quantized') and x.is_quantized:
            x = x.dequantize()
        if x.dtype != torch.float32:
            x = x.float()
        return x

    def _run_seq(self, x, seq, mems):
        """Run one timestep through a ModuleList, threading LIF membrane state."""
        snn_idx = 0
        for layer in seq:
            if isinstance(layer, snn.Leaky):
                x = self._to_fp32(x)
                mem = mems[snn_idx]
                if mem is None or mem.shape != x.shape:
                    mem = torch.zeros_like(x)
                x, mems[snn_idx] = layer(x, mem)
                snn_idx += 1
            else:
                x = layer(x)
        return x

    def forward(self, x):
        """x: [Time, Batch, C, H, W] spike train, or [Batch, C, H, W] to encode.

        Returns the per-timestep output stack [Time, Batch, num_classes]."""
        if x.dim() == 4:  # raw images -> Poisson spike train (same encoder as QDSNN)
            x = encode_poisson(x, self.time_steps)

        feature_mems = [None] * sum(1 for m in self.features_seq if isinstance(m, snn.Leaky))
        classifier_mems = [None] * sum(1 for m in self.classifier_seq if isinstance(m, snn.Leaky))

        out_rec = []
        for t in range(self.time_steps):
            xt = self.quant(x[t])
            xt = self._run_seq(xt, self.features_seq, feature_mems)
            xt = self._run_seq(xt, self.classifier_seq, classifier_mems)
            out_rec.append(self.dequant(xt))

        return torch.stack(out_rec, dim=0)


# --- QAT helpers ---
def prepare_quantized_snn_lenet_for_qat(model):
    """Fuse, exclude LIF layers from quantization, and insert fake-quant modules."""
    torch.backends.quantized.engine = get_quant_engine()
    model.fuse_model()
    model.qconfig = tq.QConfig(
        activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=tq.MinMaxObserver.with_args(dtype=torch.qint8,
                                           qscheme=torch.per_tensor_symmetric),
    )
    for _, module in model.named_modules():
        if isinstance(module, snn.Leaky):
            module.qconfig = None
    tq.prepare_qat(model, inplace=True)
    return model


def convert_quantized_snn_lenet(model):
    """Convert a QAT-trained model to a real INT8 model for inference."""
    torch.backends.quantized.engine = get_quant_engine()
    return tq.convert(model.eval().cpu(), inplace=False)
