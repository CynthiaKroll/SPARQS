import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import snntorch as snn
from snntorch import surrogate
import copy

from utils import encode_poisson_rate, DEFAULT_SPIKE_GAIN

# Define the surrogate gradient function (arctan as specified)
spike_grad = surrogate.atan()


def get_quant_engine():
    """Pick a quantization engine that this machine actually supports.

    The paper specifies FBGEMM (x86), but not every torch build ships it -- this
    box only exposes 'onednn', so a hardcoded 'fbgemm' made prepare_qat/convert
    fail at runtime. Preference order keeps FBGEMM first for paper fidelity."""
    supported = list(torch.backends.quantized.supported_engines)
    for preferred in ('fbgemm', 'qnnpack', 'onednn'):
        if preferred in supported:
            return preferred
    return supported[0]


class QuantizedSNNMLP(nn.Module):
    """
    Hybrid Quantized SNN MLP: Linear layers quantized to INT8 via QAT, LIF
    (snn.Leaky) spiking layers kept in FP32 with explicit membrane state
    (init_hidden=False).

    A QuantStub sits before every quantized Linear; LIF inputs are dequantized
    on the fly. Linear+BatchNorm pairs MUST be fused (fuse_model()) before
    prepare_qat -- otherwise the converted INT8 model contains standalone FP32
    BatchNorm1d layers that receive quantized tensors and crash at inference.
    """
    def __init__(self, num_classes=10, beta=0.95, thresholds=None, time_steps=4, input_size=1*28*28):
        super().__init__()
        self.beta = beta
        self.time_steps = time_steps
        self.input_size = input_size

        self.quant = QuantStub()     # Overall model entry
        self.dequant = DeQuantStub() # Overall model exit

        # --- Features - WITH intermediate QuantStubs ---
        self.features_seq = nn.ModuleList([
            # Layer 1
            nn.Linear(input_size, 4096),                                                # 0
            nn.BatchNorm1d(4096),                                                       # 1
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn1', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 2
            nn.Dropout(0.25),                                                           # 3
            QuantStub(),  # 4 - quantize before next Linear

            # Layer 2
            nn.Linear(4096, 2048),                                                      # 5
            nn.BatchNorm1d(2048),                                                       # 6
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn2', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 7
            nn.Dropout(0.25),                                                           # 8
            QuantStub(),  # 9 - quantize before next Linear

            # Layer 3
            nn.Linear(2048, 1024),                                                      # 10
            nn.BatchNorm1d(1024),                                                       # 11
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn3', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 12
            QuantStub(),  # 13 - quantize before next Linear

            # Layer 4
            nn.Linear(1024, 512),                                                       # 14
            nn.BatchNorm1d(512),                                                        # 15
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn4', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 16
            QuantStub(),  # 17 - quantize before next Linear

            # Layer 5
            nn.Linear(512, 256),                                                        # 18
            nn.BatchNorm1d(256),                                                        # 19
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn5', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 20
            nn.Dropout(0.25),                                                           # 21
            QuantStub(),  # 22 - quantize before Classifier
        ])

        self.classifier_seq = nn.ModuleList([
            # Layer 6
            nn.Linear(256, 4096),                                                       # 0
            nn.BatchNorm1d(4096),                                                       # 1
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn6', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 2
            QuantStub(),  # 3 - quantize before next Linear

            # Layer 7
            nn.Linear(4096, 4096),                                                      # 4
            nn.BatchNorm1d(4096),                                                       # 5
            snn.Leaky(beta=beta, threshold=(thresholds.get('snn7', 1.0) if thresholds else 1.0), spike_grad=spike_grad, init_hidden=False, reset_mechanism='subtract'),  # 6
            QuantStub(),  # 7 - quantize before final Linear

            # Output Layer
            nn.Linear(4096, num_classes),                                               # 8
        ])

    # ------------------------------------------------------------------ utils
    def fuse_model(self):
        """Fuse each Linear+BatchNorm pair BY MODULE NAME (required before QAT).

        Without this, prepare_qat leaves the BatchNorm1d layers as standalone
        FP32 modules; after convert() they receive quantized tensors and the
        forward pass raises NotImplementedError (aten::native_batch_norm has no
        QuantizedCPU kernel). The saved 'INT8' checkpoint could then never run."""
        pairs = [['features_seq.0', 'features_seq.1'],
                 ['features_seq.5', 'features_seq.6'],
                 ['features_seq.10', 'features_seq.11'],
                 ['features_seq.14', 'features_seq.15'],
                 ['features_seq.18', 'features_seq.19'],
                 ['classifier_seq.0', 'classifier_seq.1'],
                 ['classifier_seq.4', 'classifier_seq.5']]
        # TRAIN mode: fuse_modules_qat produces intrinsic LinearBn1d modules that
        # keep BatchNorm LIVE during QAT training (convert folds it at the end).
        # Fusing in eval mode before training folds the freshly-initialized BN
        # (an identity) into the Linear and deletes it, so the deep SNN trains
        # with no normalization at all and collapses to ~10% accuracy.
        # EVAL mode (INT8 skeleton rebuild in evaluate.py): static folding is
        # correct, the weights come from the checkpoint.
        if self.training:
            tq.fuse_modules_qat(self, pairs, inplace=True)
        else:
            tq.fuse_modules(self, pairs, inplace=True)
        return self

    def _encode_poisson(self, x, num_steps, gain=DEFAULT_SPIKE_GAIN):
        """Canonical Poisson rate encoding (see utils.encode_poisson_rate)."""
        return encode_poisson_rate(x, num_steps, gain)

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

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        """x: [T, B, features] spike train, or [B, features] to Poisson-encode.

        Returns the per-timestep output stack [T, B, num_classes]."""
        if x.dim() == 2:      # raw images -> Poisson spike train (canonical encoder)
            x = self._encode_poisson(x.view(x.size(0), -1), self.time_steps)
        elif x.dim() != 3:
            raise ValueError(f"Invalid input dimension: {x.dim()}. Expected 2D or 3D tensor.")

        feature_mems = [None] * sum(1 for m in self.features_seq if isinstance(m, snn.Leaky))
        classifier_mems = [None] * sum(1 for m in self.classifier_seq if isinstance(m, snn.Leaky))

        out_rec = []
        for t in range(x.size(0)):
            xt = self.quant(x[t])
            xt = self._run_seq(xt, self.features_seq, feature_mems)
            xt = self._run_seq(xt, self.classifier_seq, classifier_mems)
            # Dequantize per timestep: stacking quantized tensors is not
            # generally supported, and downstream code expects FP32 logits.
            out_rec.append(self.dequant(xt))

        return torch.stack(out_rec, dim=0)


# --- QAT helpers (single source of truth; train.py and evaluate.py use these) ---
def prepare_quantized_snn_mlp_for_qat(model):
    """Fuse Linear+BN, exclude LIF layers, and insert fake-quant modules.

    The previous version passed TYPE NAMES (['Linear', 'BatchNorm1d']) to
    fuse_modules, which expects module names and would raise AttributeError,
    and hardcoded the fbgemm engine/qconfig."""
    torch.backends.quantized.engine = get_quant_engine()
    model.fuse_model()
    model.qconfig = tq.QConfig(
        activation=tq.MinMaxObserver.with_args(dtype=torch.qint8),
        weight=tq.MinMaxObserver.with_args(dtype=torch.qint8,
                                           qscheme=torch.per_tensor_symmetric),
    )
    for _, module in model.named_modules():
        if isinstance(module, snn.Leaky):
            module.qconfig = None
    tq.prepare_qat(model, inplace=True)
    return model


def convert_quantized_snn_mlp(model_qat_trained):
    """Convert a QAT-trained (fake-quant) model to a real INT8 model."""
    torch.backends.quantized.engine = get_quant_engine()
    # snntorch LIF neurons retain non-leaf membrane/spike tensors after a
    # forward pass, which block deepcopy; detach them first.
    for mod in model_qat_trained.modules():
        if isinstance(mod, snn.Leaky):
            for attr in ('mem', 'spk'):
                v = getattr(mod, attr, None)
                if torch.is_tensor(v):
                    setattr(mod, attr, v.detach())
    model = copy.deepcopy(model_qat_trained).eval().cpu()
    return tq.convert(model, inplace=False)
