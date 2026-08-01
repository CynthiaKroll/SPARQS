import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import snntorch as snn
from snntorch import surrogate

spike_grad = surrogate.atan()

# Same hard bound on the LIF membrane as the QSNN backbone
# (models.quantized_snn_alexnet.MEMBRANE_CLAMP). Without it, the membrane
# accumulates toward +/-inf over T=32 steps and NaNs the loss at ~epoch 12,
# halting training. Threshold is 1.0, so a clamp at 8.0 leaves healthy
# dynamics untouched. Both models MUST clamp identically or the QSNN-vs-QDSNN
# comparison is no longer like-for-like.
MEMBRANE_CLAMP = 8.0


def _clamp_membrane(mem):
    mem = torch.nan_to_num(mem, nan=0.0, posinf=MEMBRANE_CLAMP, neginf=-MEMBRANE_CLAMP)
    return torch.clamp(mem, -MEMBRANE_CLAMP, MEMBRANE_CLAMP)


# NOTE: an earlier _accum_norm() (decay-sum normalization of the exit
# accumulator) and a post-hoc temperature calibration were removed. Together they
# made the model exit every sample at t=1 with near-random predictions: the
# normalization erased the natural "confidence grows with evidence" behaviour and
# the calibrated temperature (T2~0.03) saturated confidence to 1.0. Raw
# accumulated confidence + a threshold sweep (the LeNet-proven approach) replaces
# them. temperature1/2 are kept for checkpoint compatibility but stay at 1.0.


class ExitClassifier(nn.Module):
    """Early-exit head: pool the spike map to a coarse grid, normalize, classify.

    The previous version global-average-pooled to 1x1, collapsing a 16x16 spike
    map to ONE number per channel (192 features). Measured consequence: exit 2
    reached 52.7% standalone accuracy but its logits averaged |0.12| (vs 0.72
    for the final head), so its softmax sat near-uniform and its max confidence
    over the whole test set was 0.197 -- it could never cross ANY usable
    threshold, and the more accurate of the two heads contributed 0% of exits.

    Pooling to 4x4 keeps coarse spatial structure (16x the features) and the
    BatchNorm rescales the sparse spike rates, so the linear layer can emit
    well-scaled logits.

    Still tiny next to the backbone: 3072*10 / 4096*10 weights vs ~35.9M.
    Kept in FP32 (not quantized) because it gates control flow."""

    POOL = 4

    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((self.POOL, self.POOL))
        self.flatten = nn.Flatten()
        n_feat = in_channels * self.POOL * self.POOL
        self.norm = nn.BatchNorm1d(n_feat)
        self.fc = nn.Linear(n_feat, num_classes)
        # Keep the whole head in FP32 (it gates control flow, not bulk compute).
        self.norm.qconfig = None
        self.fc.qconfig = None

    def forward(self, x):
        return self.fc(self.norm(self.flatten(self.pool(x))))


class BranchyQuantizedSNN(nn.Module):
    """
    Quantized Dynamic SNN (QDSNN) AlexNet.

    Built on the SAME backbone as QuantizedSNNAlexNet (conv1..conv5 -> fc1..fc3,
    3->64->192->384->256->256 -> 4096->4096->10), with two early-exit heads
    inserted after conv block 2 and conv block 4. The ONLY difference from the
    QSNN is the dynamic early exits, so any power/delay improvement is
    attributable to early exiting rather than to a smaller network.

    (The previous version of this class was a standalone 2-conv/2-fc net with
    6.4M params, versus the QSNN's 35.9M -- an invalid baseline comparison.)

    Hybrid quantization (identical scheme to QuantizedSNNAlexNet):
      - Conv/Linear layers are quantized to INT8 via QAT (QuantStub before each).
      - LIF (snn.Leaky) neurons run in FP32 with explicit membrane state
        (init_hidden=False); their inputs are dequantized on the fly.
      - Exit heads stay FP32.
    """

    def __init__(self, num_classes=10, time_window=32, beta=0.95, thresholds=None):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.time_window = time_window
        self.time_steps = time_window  # alias so getattr(model, 'time_steps') works
        self.n_exits = 2
        # Raised from [0.3, 0.3, 1.0]: at 0.3 the exit heads got weak gradient
        # pressure and stalled at 34.7% / 52.7% standalone accuracy, far too low
        # to exit on without a large accuracy cost. The final exit keeps the
        # dominant weight so the full-depth path stays the most accurate.
        self.exit_loss_weights = [0.5, 0.5, 1.0]  # [exit1, exit2, final]
        # Decay of the per-timestep exit-logit accumulator. Named (was a local
        # `d = 0.9` in both forward paths) so the temperature calibration can
        # normalize by exactly the same factor the forward pass uses.
        self.accum_decay = 0.9

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

        # Quantization stubs: ONE PER TENSOR. Sharing a single QuantStub across
        # the input image and several spike tensors (as the old version did)
        # forces one scale/zero-point onto tensors with very different ranges.
        self.quant = QuantStub()   # model entry (encoded input spikes)
        self.q1 = QuantStub()      # before conv2
        self.q2 = QuantStub()      # before conv3
        self.q3 = QuantStub()      # before conv4
        self.q4 = QuantStub()      # before conv5
        self.q5 = QuantStub()      # before fc1
        self.q6 = QuantStub()      # before fc2
        self.q7 = QuantStub()      # before fc3
        self.dequant_exit1 = DeQuantStub()
        self.dequant_exit2 = DeQuantStub()
        self.dequant_final = DeQuantStub()

        # --- Feature backbone (mirrors QuantizedSNNAlexNet.features_seq) ---
        # Order is conv -> BN -> LIF. The old code ran conv -> LIF -> BN and fed
        # the *un-normalized* spikes downstream, so BN only ever affected the
        # exit branch while fuse_bn() still folded it into the conv weights.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = make_lif('snn1')
        self.pool1 = nn.MaxPool2d(2, 2)                     # 32 -> 16

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.lif2 = make_lif('snn2')
        self.exit1 = ExitClassifier(192, num_classes)       # <-- Exit 1
        self.pool2 = nn.MaxPool2d(2, 2)                     # 16 -> 8

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.lif3 = make_lif('snn3')

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = make_lif('snn4')
        self.exit2 = ExitClassifier(256, num_classes)       # <-- Exit 2

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.lif5 = make_lif('snn5')
        self.pool5 = nn.MaxPool2d(2, 2)                     # 8 -> 4

        # --- Classifier backbone (mirrors QuantizedSNNAlexNet.classifier_seq) ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.lif6 = make_lif('snn6')

        self.fc2 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.lif7 = make_lif('snn7')

        self.fc3 = nn.Linear(4096, num_classes)  # final exit (Exit 3)

        # Per-inference diagnostics, used by the exit-aware delay metric.
        self.last_exit_points = None
        self.last_exit_timesteps = None
        self.last_avg_timesteps = float('nan')
        self.last_avg_depth = float('nan')

    # ------------------------------------------------------------------ utils
    def fuse_model(self):
        """Fuse each Conv/Linear + BatchNorm pair BY MODULE NAME (required for QAT).

        In TRAIN mode fuse_modules_qat keeps BatchNorm LIVE (intrinsic ConvBn2d/
        LinearBn1d, folded only at convert); eval-mode fusion before training
        would fold the freshly-initialized BN away and train without
        normalization."""
        pairs = [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],
                 ['conv4', 'bn4'], ['conv5', 'bn5'],
                 ['fc1', 'bn6'], ['fc2', 'bn7']]
        if self.training:
            tq.fuse_modules_qat(self, pairs, inplace=True)
        else:
            tq.fuse_modules(self, pairs, inplace=True)
        return self

    def _encode_poisson(self, x, num_steps, gain=None):
        """Canonical Poisson rate encoding -- the SAME function the QSNN and the
        AC/spike counters use (models.quantized_snn_alexnet.encode_poisson).
        The previous private copy divided by num_steps and clamped standardized
        inputs, silently diverging from the shared encoder."""
        from models.quantized_snn_alexnet import encode_poisson
        return encode_poisson(x, num_steps, gain)

    @staticmethod
    def _maybe_dequant(t):
        return t.dequantize() if (hasattr(t, 'is_quantized') and t.is_quantized) else t

    def _spiking_block(self, x, conv, bn, lif, mem):
        """conv/fc -> bn -> (dequant) -> LIF. Returns (spk[FP32], mem[FP32])."""
        x = conv(x)
        x = bn(x)
        # LIF must run in FP32; dequantize the (possibly INT8) activation first.
        x = self._maybe_dequant(x)
        if x.dtype != torch.float32:
            x = x.float()
        if mem is None or mem.shape != x.shape:
            mem = torch.zeros_like(x)
        spk, mem = lif(x, mem)
        return spk, _clamp_membrane(mem)  # cap the T=32 runaway

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        if x.dim() == 4:      # [batch, C, H, W] -> encode to spike train
            spike_train = self._encode_poisson(x, self.time_window)
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
        mems = [None] * 7
        e1a = torch.zeros(B, C, device=dev)
        e2a = torch.zeros(B, C, device=dev)
        fa = torch.zeros(B, C, device=dev)
        d = self.accum_decay

        for t in range(T):
            xt = self.quant(spike_train[t])

            spk1, mems[0] = self._spiking_block(xt, self.conv1, self.bn1, self.lif1, mems[0])
            x = self.q1(self.pool1(spk1))

            spk2, mems[1] = self._spiking_block(x, self.conv2, self.bn2, self.lif2, mems[1])
            e1a = e1a * d + self.exit1(spk2)
            x = self.q2(self.pool2(spk2))

            spk3, mems[2] = self._spiking_block(x, self.conv3, self.bn3, self.lif3, mems[2])
            x = self.q3(spk3)

            spk4, mems[3] = self._spiking_block(x, self.conv4, self.bn4, self.lif4, mems[3])
            e2a = e2a * d + self.exit2(spk4)
            x = self.q4(spk4)

            spk5, mems[4] = self._spiking_block(x, self.conv5, self.bn5, self.lif5, mems[4])
            x = self.q5(self.flatten(self.pool5(spk5)))

            spk6, mems[5] = self._spiking_block(x, self.fc1, self.bn6, self.lif6, mems[5])
            x = self.q6(spk6)

            spk7, mems[6] = self._spiking_block(x, self.fc2, self.bn7, self.lif7, mems[6])
            x = self.q7(spk7)

            fa = fa * d + self._maybe_dequant(self.fc3(x))

        exit_outputs = [self.dequant_exit1(e1a), self.dequant_exit2(e2a)]
        return exit_outputs, self.dequant_final(fa), None

    def _forward_inference(self, spike_train):
        """Per-timestep early exit. Exited samples stop propagating spikes to
        deeper layers (real AC savings) and the time loop breaks once every
        sample has exited (real timestep/latency savings).

        The old implementation masked only the output accumulators, so every
        conv and fc still ran for every sample at every timestep -- i.e. it
        reported early exits while saving no compute at all."""
        T, B = spike_train.shape[0], spike_train.shape[1]
        dev = spike_train.device
        C = self.num_classes
        mems = [None] * 7
        e1a = torch.zeros(B, C, device=dev)
        e2a = torch.zeros(B, C, device=dev)
        fa = torch.zeros(B, C, device=dev)
        d = self.accum_decay

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

            # --- Conv blocks 1-2 + Exit 1 ---
            spk1, mems[0] = self._spiking_block(xt, self.conv1, self.bn1, self.lif1, mems[0])
            x = self.q1(self.pool1(spk1))

            spk2, mems[1] = self._spiking_block(x, self.conv2, self.bn2, self.lif2, mems[1])
            e1 = self.exit1(spk2)
            e1a = torch.where(active.unsqueeze(1), e1a * d + e1, e1a)
            # Exit decision uses the ACCUMULATED exit logits (e1a), NOT the raw
            # single-timestep logits (which are 1-step spike noise). The raw
            # accumulator is intentionally small early and grows as spikes
            # accumulate, so confidence is naturally low at t=1 and rises with
            # evidence -- exits fire once there is enough evidence, not on tick 1.
            # (An earlier _accum_norm normalization + temperature calibration
            # removed that safeguard and made the model dump everything at t=1
            # with near-random predictions; both were reverted. temperature=1.0.)
            conf1 = torch.softmax(e1a / self.temperature1, dim=1).max(dim=1)[0]
            ex1 = (conf1 > self.confidence_thresholds[0]) & active
            exit_pts[ex1] = 1
            exit_ts[ex1] = t + 1
            active = active & ~ex1
            if not active.any():
                # Every sample has exited: genuinely SKIP the deeper layers for
                # this timestep. Without this the conv/fc layers would still be
                # invoked on all-zero activations, so the memory-access counter
                # (which dominates the energy model) would show no saving at all.
                continue
            x = self.q2(self.pool2(gate(spk2, active)))  # only still-active samples continue

            # --- Conv blocks 3-4 + Exit 2 ---
            spk3, mems[2] = self._spiking_block(x, self.conv3, self.bn3, self.lif3, mems[2])
            x = self.q3(spk3)

            spk4, mems[3] = self._spiking_block(x, self.conv4, self.bn4, self.lif4, mems[3])
            e2 = self.exit2(spk4)
            e2a = torch.where(active.unsqueeze(1), e2a * d + e2, e2a)
            conf2 = torch.softmax(e2a / self.temperature2, dim=1).max(dim=1)[0]
            ex2 = (conf2 > self.confidence_thresholds[1]) & active
            exit_pts[ex2] = 2
            exit_ts[ex2] = t + 1
            active = active & ~ex2
            if not active.any():
                continue  # skip conv5 + classifier entirely for this timestep
            x = self.q4(gate(spk4, active))

            # --- Conv block 5 + classifier (final exit) ---
            spk5, mems[4] = self._spiking_block(x, self.conv5, self.bn5, self.lif5, mems[4])
            x = self.q5(self.flatten(self.pool5(spk5)))

            spk6, mems[5] = self._spiking_block(x, self.fc1, self.bn6, self.lif6, mems[5])
            x = self.q6(spk6)

            spk7, mems[6] = self._spiking_block(x, self.fc2, self.bn7, self.lif7, mems[6])
            x = self.q7(spk7)

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


def reset_exit_temperatures(model):
    """Force both exit temperatures back to 1.0 (raw accumulated confidence).

    Checkpoints trained while the (now-removed) calibration was wired in have
    over-sharpened temperatures baked into their state_dict (e.g. T2~0.03), which
    saturates confidence to 1.0 and makes the model exit everything at t=1. The
    evaluators call this after loading so the exit decision uses raw confidence
    regardless of what a legacy checkpoint stored. A no-op for fresh checkpoints
    (temperatures already 1.0)."""
    model.temperature1.data.fill_(1.0)
    model.temperature2.data.fill_(1.0)
    return model
