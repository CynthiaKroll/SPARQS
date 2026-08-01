import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import snntorch as snn
from snntorch import surrogate

from config import config
from models.quantized_snn_lenet import encode_poisson, get_quant_engine  # noqa: F401

spike_grad = surrogate.atan()


class ExitClassifier(nn.Module):
    """Early-exit head: global-average-pool the spike map, then a linear layer.

    Kept in FP32 (not quantized) because it gates control flow and is tiny
    relative to the backbone (6*10 / 16*10 weights vs 62k)."""

    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, num_classes)
        self.fc.qconfig = None  # keep exit heads in FP32

    def forward(self, x):
        return self.fc(self.flatten(self.pool(x)))


class BranchyQuantizedSNN(nn.Module):
    """
    Quantized Dynamic SNN (QDSNN) LeNet.

    Built on the SAME backbone as QuantizedSNNLeNet
    (conv1(1->6) -> conv2(6->16) -> fc1(400->120) -> fc2(120->84) -> fc3(84->10)),
    with two early-exit heads inserted after conv block 1 and conv block 2. The
    ONLY difference from the QSNN is the dynamic early exits, so any power/delay
    improvement is attributable to early exiting rather than to a different net.
    (Backbone weights == QuantizedSNNLeNet's, 62,158 params.)

    Hybrid quantization (identical scheme to QuantizedSNNLeNet):
      - Conv/Linear layers are quantized to INT8 via QAT (QuantStub before each).
      - LIF (snn.Leaky) neurons run in FP32 with explicit membrane state
        (init_hidden=False); their inputs are dequantized on the fly.
      - Exit heads stay FP32.

    The previous version was a standalone net whose _forward_inference explicitly
    skipped the exit heads ("Continue to main path only") -- it ran the full
    network for every sample at every timestep and never exited early, so the
    entire dynamic-inference contribution was inert.
    """

    def __init__(self, num_classes=10, time_window=None, beta=0.95, thresholds=None):
        super().__init__()
        if time_window is None:
            time_window = config["num_steps"]
        self.num_classes = num_classes
        self.beta = beta
        self.time_window = time_window
        self.time_steps = time_window  # alias so getattr(model, 'time_steps') works
        self.n_exits = 2
        self.exit_loss_weights = [0.3, 0.3, 1.0]  # [exit1, exit2, final]

        # RL agent + thresholds (RL wiring handled by the training script)
        self.rl_agent = None
        self.confidence_thresholds = [0.9, 0.9]

        # Temperature scaling for the two early exits
        self.temperature1 = nn.Parameter(torch.ones(1))
        self.temperature2 = nn.Parameter(torch.ones(1))

        def make_lif(key):
            th = thresholds.get(key, 1.0) if thresholds else 1.0
            return snn.Leaky(beta=beta, threshold=th, spike_grad=spike_grad,
                             init_hidden=False, reset_mechanism='subtract')

        # Quantization stubs: ONE PER TENSOR (the old code shared one QuantStub
        # across tensors of very different ranges).
        self.quant = QuantStub()   # model entry (encoded input spikes)
        self.q1 = QuantStub()      # before conv2
        self.q2 = QuantStub()      # before fc1
        self.q3 = QuantStub()      # before fc2
        self.q4 = QuantStub()      # before fc3
        self.dequant_exit1 = DeQuantStub()
        self.dequant_exit2 = DeQuantStub()
        self.dequant_final = DeQuantStub()

        # --- Feature backbone (mirrors QuantizedSNNLeNet.features_seq) ---
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.lif1 = make_lif('features.2')
        self.pool1 = nn.AvgPool2d(2, 2)                     # 28 -> 14
        self.exit1 = ExitClassifier(6, num_classes)         # <-- Exit 1

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.lif2 = make_lif('features.6')
        self.pool2 = nn.AvgPool2d(2, 2)                     # 10 -> 5
        self.exit2 = ExitClassifier(16, num_classes)        # <-- Exit 2

        # --- Classifier backbone (mirrors QuantizedSNNLeNet.classifier_seq) ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.lif_fc1 = make_lif('classifier.3')

        self.fc2 = nn.Linear(120, 84)
        self.bn_fc2 = nn.BatchNorm1d(84)
        self.lif_fc2 = make_lif('classifier.6')

        self.fc3 = nn.Linear(84, num_classes)  # final exit (Exit 3)

        # Per-inference diagnostics, used by the exit-aware delay metric.
        self.last_exit_points = None
        self.last_exit_timesteps = None
        self.last_avg_timesteps = float('nan')
        self.last_avg_depth = float('nan')

    # ------------------------------------------------------------------ utils
    def fuse_model(self):
        """Fuse each Conv/Linear + BatchNorm pair by module name (required for QAT).

        In TRAIN mode fuse_modules_qat keeps BatchNorm LIVE (intrinsic ConvBn2d/
        LinearBn1d, folded only at convert); eval-mode fusion before training
        would fold the freshly-initialized BN away and train without
        normalization."""
        pairs = [['conv1', 'bn1'], ['conv2', 'bn2'],
                 ['fc1', 'bn_fc1'], ['fc2', 'bn_fc2']]
        if self.training:
            tq.fuse_modules_qat(self, pairs, inplace=True)
        else:
            tq.fuse_modules(self, pairs, inplace=True)
        return self

    @staticmethod
    def _maybe_dequant(t):
        return t.dequantize() if (hasattr(t, 'is_quantized') and t.is_quantized) else t

    def _spiking_block(self, x, conv, bn, lif, mem):
        """conv/fc -> bn -> (dequant) -> LIF. Returns (spk[FP32], mem[FP32])."""
        x = conv(x)
        x = bn(x)
        x = self._maybe_dequant(x)
        if x.dtype != torch.float32:
            x = x.float()
        if mem is None or mem.shape != x.shape:
            mem = torch.zeros_like(x)
        return lif(x, mem)

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        if x.dim() == 4:      # [batch, C, H, W] -> encode to spike train
            spike_train = encode_poisson(x, self.time_window)
        else:                 # already [time, batch, C, H, W]
            spike_train = x
        if self.training:
            return self._forward_training(spike_train)
        return self._forward_inference(spike_train)

    def _forward_training(self, spike_train):
        """All exits always run; returns (exit_outputs, final_output, None)."""
        T, B = spike_train.shape[0], spike_train.shape[1]
        dev = spike_train.device
        C = self.num_classes
        mems = [None] * 4
        e1a = torch.zeros(B, C, device=dev)
        e2a = torch.zeros(B, C, device=dev)
        fa = torch.zeros(B, C, device=dev)
        d = 0.9  # accumulator decay

        for t in range(T):
            xt = self.quant(spike_train[t])

            spk1, mems[0] = self._spiking_block(xt, self.conv1, self.bn1, self.lif1, mems[0])
            p1 = self.pool1(spk1)
            e1a = e1a * d + self.exit1(p1)
            x = self.q1(p1)

            spk2, mems[1] = self._spiking_block(x, self.conv2, self.bn2, self.lif2, mems[1])
            p2 = self.pool2(spk2)
            e2a = e2a * d + self.exit2(p2)
            x = self.q2(self.flatten(p2))

            spk_fc1, mems[2] = self._spiking_block(x, self.fc1, self.bn_fc1, self.lif_fc1, mems[2])
            x = self.q3(spk_fc1)

            spk_fc2, mems[3] = self._spiking_block(x, self.fc2, self.bn_fc2, self.lif_fc2, mems[3])
            x = self.q4(spk_fc2)

            fa = fa * d + self._maybe_dequant(self.fc3(x))

        exit_outputs = [self.dequant_exit1(e1a), self.dequant_exit2(e2a)]
        return exit_outputs, self.dequant_final(fa), None

    def _forward_inference(self, spike_train):
        """Per-timestep early exit. Exited samples stop propagating spikes to
        deeper layers (real AC savings) and the time loop breaks once every
        sample has exited (real timestep/latency savings).

        The old implementation never computed the exit heads at all here, so it
        reported no exit statistics and saved no compute."""
        T, B = spike_train.shape[0], spike_train.shape[1]
        dev = spike_train.device
        C = self.num_classes
        mems = [None] * 4
        e1a = torch.zeros(B, C, device=dev)
        e2a = torch.zeros(B, C, device=dev)
        fa = torch.zeros(B, C, device=dev)
        d = 0.9

        active = torch.ones(B, dtype=torch.bool, device=dev)
        exit_pts = torch.full((B,), 3, dtype=torch.long, device=dev)  # default: final exit
        exit_ts = torch.full((B,), T, dtype=torch.long, device=dev)   # steps used (default full T)

        def gate(t_, mask):
            """Zero out the feature maps of samples that have already exited."""
            return t_ * mask.float().view(-1, *([1] * (t_.dim() - 1)))

        for t in range(T):
            if not active.any():
                break
            xt = self.quant(gate(spike_train[t], active))

            # --- Conv block 1 + Exit 1 ---
            spk1, mems[0] = self._spiking_block(xt, self.conv1, self.bn1, self.lif1, mems[0])
            p1 = self.pool1(spk1)
            e1 = self.exit1(p1)
            e1a = torch.where(active.unsqueeze(1), e1a * d + e1, e1a)
            # Exit decision uses the ACCUMULATED exit logits (e1a) -- the tensor
            # returned as this exit's prediction and the statistic the RL agent's
            # confidence states are built from (paper Sec. III-B: "the confidence
            # of that branch's prediction"). Deciding on the raw single-timestep
            # logits e1 applied thresholds learned on full-T confidences to
            # 1-step spike noise, firing exits at t=1 with near-random outputs.
            conf1 = torch.softmax(e1a / self.temperature1, dim=1).max(dim=1)[0]
            ex1 = (conf1 > self.confidence_thresholds[0]) & active
            exit_pts[ex1] = 1
            exit_ts[ex1] = t + 1
            active = active & ~ex1
            if not active.any():
                # Every sample has exited: genuinely SKIP the deeper layers for
                # this timestep (otherwise the memory-access counter, which
                # dominates the energy model, would show no saving at all).
                continue
            x = self.q1(gate(p1, active))  # only still-active samples continue

            # --- Conv block 2 + Exit 2 ---
            spk2, mems[1] = self._spiking_block(x, self.conv2, self.bn2, self.lif2, mems[1])
            p2 = self.pool2(spk2)
            e2 = self.exit2(p2)
            e2a = torch.where(active.unsqueeze(1), e2a * d + e2, e2a)
            conf2 = torch.softmax(e2a / self.temperature2, dim=1).max(dim=1)[0]
            ex2 = (conf2 > self.confidence_thresholds[1]) & active
            exit_pts[ex2] = 2
            exit_ts[ex2] = t + 1
            active = active & ~ex2
            if not active.any():
                continue  # skip the classifier entirely for this timestep
            x = self.q2(self.flatten(gate(p2, active)))

            # --- Classifier (final exit) ---
            spk_fc1, mems[2] = self._spiking_block(x, self.fc1, self.bn_fc1, self.lif_fc1, mems[2])
            x = self.q3(spk_fc1)

            spk_fc2, mems[3] = self._spiking_block(x, self.fc2, self.bn_fc2, self.lif_fc2, mems[3])
            x = self.q4(spk_fc2)

            f = self._maybe_dequant(self.fc3(x))
            fa = torch.where(active.unsqueeze(1), fa * d + f, fa)

        # Diagnostics for the exit-aware delay metric
        self.last_exit_points = exit_pts.cpu()
        self.last_exit_timesteps = exit_ts.cpu()
        self.last_avg_timesteps = exit_ts.float().mean().item()
        self.last_avg_depth = exit_pts.float().mean().item()

        exit_outputs = [self.dequant_exit1(e1a), self.dequant_exit2(e2a)]
        return exit_outputs, self.dequant_final(fa), exit_pts.cpu()

    def prepare_for_inference(self, confidence_thresholds=None, rl_agent=None):
        """Set exit thresholds directly, or from an RL agent's learned policy."""
        if confidence_thresholds is not None:
            self.confidence_thresholds = list(confidence_thresholds)
            self.rl_agent = None
        elif rl_agent is not None:
            self.rl_agent = rl_agent
            learned = rl_agent.get_confidence_thresholds()
            if learned and len(learned) == self.n_exits:
                self.confidence_thresholds = list(learned)
        return self
