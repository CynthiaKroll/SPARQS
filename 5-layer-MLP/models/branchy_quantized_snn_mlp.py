import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
import snntorch as snn
from snntorch import surrogate

# Import RL agent for dynamic inference
from models.rl_environment import QLearningAgent
from utils import encode_poisson_rate, DEFAULT_SPIKE_GAIN

# Define the surrogate gradient function (arctan as specified)
spike_grad = surrogate.atan()


class ExitClassifier(nn.Module):
    """Simple early-exit classifier head. Kept in FP32 (not quantized) because it
    gates control flow and is tiny relative to the backbone."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.fc.qconfig = None  # keep exit heads in FP32

    def forward(self, x):
        return self.fc(x)


class BranchyQuantizedSNNMLP(nn.Module):
    """
    Quantized Dynamic SNN (QDSNN) built on the SAME backbone as QuantizedSNNMLP
    (784 -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> 4096 -> 4096 -> 10), with two
    early-exit heads inserted after feature layer 1 (4096) and feature layer 3
    (1024). The ONLY difference from the QSNN is the dynamic early exits, so any
    power/delay improvement is attributable to early exiting rather than a
    smaller network.

    Hybrid quantization (identical scheme to QuantizedSNNMLP):
      - Linear layers are quantized to INT8 via QAT (QuantStub before each Linear).
      - LIF (snn.Leaky) neurons run in FP32 with explicit membrane state
        (init_hidden=False); their inputs are dequantized on the fly.
      - Exit heads stay FP32.
    """

    # Backbone layer sizes (input -> ... -> num_classes), matching QuantizedSNNMLP.
    def __init__(self, num_classes=10, time_window=32, beta=0.95, input_size=784,
                 thresholds=None):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.beta = beta
        self.time_window = time_window
        self.time_steps = time_window  # alias so getattr(model, 'time_steps') works
        self.exit_loss_weights = [0.3, 0.3, 0.4]
        self.n_exits = 2

        # RL agent + thresholds (RL wiring handled by the training script)
        self.rl_agent = None
        self.confidence_thresholds = [0.7, 0.7]

        # Temperature scaling for the two early exits
        self.temperature1 = nn.Parameter(torch.ones(1))
        self.temperature2 = nn.Parameter(torch.ones(1))

        # Quantization stubs
        self.quant = QuantStub()              # model entry
        self.dequant_exit1 = DeQuantStub()
        self.dequant_exit2 = DeQuantStub()
        self.dequant_final = DeQuantStub()

        def make_lif(key):
            th = thresholds.get(key, 1.0) if thresholds else 1.0
            return snn.Leaky(beta=beta, threshold=th, spike_grad=spike_grad,
                             init_hidden=False, reset_mechanism='subtract')

        # --- Feature backbone ---
        # Layer 1 (+ Exit 1)
        self.fc1 = nn.Linear(input_size, 4096); self.bn1 = nn.BatchNorm1d(4096)
        self.lif1 = make_lif('snn1'); self.drop1 = nn.Dropout(0.25); self.q1 = QuantStub()
        self.exit1 = ExitClassifier(4096, num_classes)

        # Layer 2
        self.fc2 = nn.Linear(4096, 2048); self.bn2 = nn.BatchNorm1d(2048)
        self.lif2 = make_lif('snn2'); self.drop2 = nn.Dropout(0.25); self.q2 = QuantStub()

        # Layer 3 (+ Exit 2)
        self.fc3 = nn.Linear(2048, 1024); self.bn3 = nn.BatchNorm1d(1024)
        self.lif3 = make_lif('snn3'); self.q3 = QuantStub()
        self.exit2 = ExitClassifier(1024, num_classes)

        # Layer 4
        self.fc4 = nn.Linear(1024, 512); self.bn4 = nn.BatchNorm1d(512)
        self.lif4 = make_lif('snn4'); self.q4 = QuantStub()

        # Layer 5
        self.fc5 = nn.Linear(512, 256); self.bn5 = nn.BatchNorm1d(256)
        self.lif5 = make_lif('snn5'); self.drop5 = nn.Dropout(0.25); self.q5 = QuantStub()

        # --- Classifier backbone ---
        self.fc6 = nn.Linear(256, 4096); self.bn6 = nn.BatchNorm1d(4096)
        self.lif6 = make_lif('snn6'); self.q6 = QuantStub()

        self.fc7 = nn.Linear(4096, 4096); self.bn7 = nn.BatchNorm1d(4096)
        self.lif7 = make_lif('snn7'); self.q7 = QuantStub()

        self.fc8 = nn.Linear(4096, num_classes)  # final exit (Exit 3)

        # Per-inference diagnostics (populated by _forward_inference), used by the
        # exit-aware delay metric.
        self.last_exit_points = None
        self.last_exit_timesteps = None
        self.last_avg_timesteps = float('nan')
        self.last_avg_depth = float('nan')

        self._initialize_weights()

    # ------------------------------------------------------------------ utils
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        """Fuse each Linear+BatchNorm pair by MODULE NAME (required for correct QAT).

        In TRAIN mode fuse_modules_qat keeps BatchNorm LIVE (intrinsic LinearBn1d,
        folded only at convert); eval-mode fusion before training would fold the
        freshly-initialized BN away and train the net with no normalization."""
        pairs = [['fc1', 'bn1'], ['fc2', 'bn2'], ['fc3', 'bn3'], ['fc4', 'bn4'],
                 ['fc5', 'bn5'], ['fc6', 'bn6'], ['fc7', 'bn7']]
        if self.training:
            tq.fuse_modules_qat(self, pairs, inplace=True)
        else:
            tq.fuse_modules(self, pairs, inplace=True)
        return self

    def _encode_poisson(self, x, num_steps, gain=DEFAULT_SPIKE_GAIN):
        """Canonical Poisson rate encoding (see utils.encode_poisson_rate)."""
        return encode_poisson_rate(x, num_steps, gain)

    def _spiking_block(self, x, fc, bn, lif, mem, drop=None):
        """fc -> bn -> (dequant) -> LIF. Returns (spk[FP32], mem[FP32])."""
        x = fc(x)
        x = bn(x)
        # LIF must run in FP32; dequantize the (possibly INT8) activation first.
        if hasattr(x, 'is_quantized') and x.is_quantized:
            x = x.dequantize()
        elif x.dtype != torch.float32:
            x = x.float()
        if mem is None or mem.shape != x.shape:
            mem = torch.zeros_like(x)
        spk, mem = lif(x, mem)
        if drop is not None:
            spk = drop(spk)
        return spk, mem

    @staticmethod
    def _maybe_dequant(t):
        return t.dequantize() if (hasattr(t, 'is_quantized') and t.is_quantized) else t

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        if x.dim() == 2:  # [batch, features] -> encode to spike train
            spike_train = self._encode_poisson(x.view(x.size(0), -1), self.time_window)
        else:             # already [time, batch, features]
            spike_train = x
        if self.training:
            return self._forward_training(spike_train)
        return self._forward_inference(spike_train)

    def _forward_training(self, spike_train):
        """All exits always run; returns (exit_outputs, final_output, None)."""
        T, B, _ = spike_train.shape
        dev = spike_train.device
        C = self.num_classes
        mems = [None] * 7
        e1a = torch.zeros(B, C, device=dev)
        e2a = torch.zeros(B, C, device=dev)
        fa = torch.zeros(B, C, device=dev)
        d = 0.9

        for t in range(T):
            xt = self.quant(spike_train[t])

            spk1, mems[0] = self._spiking_block(xt, self.fc1, self.bn1, self.lif1, mems[0], self.drop1)
            e1a = e1a * d + self.exit1(spk1)
            x = self.q1(spk1)

            spk2, mems[1] = self._spiking_block(x, self.fc2, self.bn2, self.lif2, mems[1], self.drop2)
            x = self.q2(spk2)

            spk3, mems[2] = self._spiking_block(x, self.fc3, self.bn3, self.lif3, mems[2])
            e2a = e2a * d + self.exit2(spk3)
            x = self.q3(spk3)

            spk4, mems[3] = self._spiking_block(x, self.fc4, self.bn4, self.lif4, mems[3])
            x = self.q4(spk4)

            spk5, mems[4] = self._spiking_block(x, self.fc5, self.bn5, self.lif5, mems[4], self.drop5)
            x = self.q5(spk5)

            spk6, mems[5] = self._spiking_block(x, self.fc6, self.bn6, self.lif6, mems[5])
            x = self.q6(spk6)

            spk7, mems[6] = self._spiking_block(x, self.fc7, self.bn7, self.lif7, mems[6])
            x = self.q7(spk7)

            fa = fa * d + self._maybe_dequant(self.fc8(x))

        exit_outputs = [self.dequant_exit1(e1a), self.dequant_exit2(e2a)]
        return exit_outputs, self.dequant_final(fa), None

    def _forward_inference(self, spike_train):
        """Per-timestep early exit. Exited samples stop propagating spikes to
        deeper layers (real AC savings) and the time loop breaks once every
        sample has exited (real timestep/latency savings)."""
        T, B, _ = spike_train.shape
        dev = spike_train.device
        C = self.num_classes
        mems = [None] * 7
        e1a = torch.zeros(B, C, device=dev)
        e2a = torch.zeros(B, C, device=dev)
        fa = torch.zeros(B, C, device=dev)
        d = 0.9

        active = torch.ones(B, dtype=torch.bool, device=dev)
        exit_pts = torch.full((B,), 3, dtype=torch.long, device=dev)   # default: final exit
        exit_ts = torch.full((B,), T, dtype=torch.long, device=dev)    # steps used (default full T)

        for t in range(T):
            if not active.any():
                break
            gate = active.float().unsqueeze(1)          # zero out already-exited samples
            xt = self.quant(spike_train[t] * gate)

            # --- Layer 1 + Exit 1 ---
            spk1, mems[0] = self._spiking_block(xt, self.fc1, self.bn1, self.lif1, mems[0], self.drop1)
            e1 = self.exit1(spk1)
            e1a = torch.where(active.unsqueeze(1), e1a * d + e1, e1a)
            # The exit decision uses the ACCUMULATED exit logits (e1a) -- the same
            # tensor returned as this exit's prediction, and the same statistic
            # the RL agent's confidence states are built from (paper Sec. III-B:
            # "the confidence of that branch's prediction"). Deciding on the raw
            # single-timestep logits e1 applied thresholds learned on full-T
            # confidences to 1-step spike noise, so exits fired at t=1 with
            # near-random predictions.
            conf1 = torch.softmax(e1a / self.temperature1, dim=1).max(dim=1)[0]
            ex1 = (conf1 > self.confidence_thresholds[0]) & active
            exit_pts[ex1] = 1; exit_ts[ex1] = t + 1
            active = active & ~ex1
            if not active.any():
                # Every sample has exited: genuinely SKIP the deeper layers for
                # this timestep instead of running them on all-zero activations.
                continue
            x = self.q1(spk1 * active.float().unsqueeze(1))  # only still-active samples continue

            # --- Layers 2-3 + Exit 2 ---
            spk2, mems[1] = self._spiking_block(x, self.fc2, self.bn2, self.lif2, mems[1], self.drop2)
            x = self.q2(spk2)
            spk3, mems[2] = self._spiking_block(x, self.fc3, self.bn3, self.lif3, mems[2])
            e2 = self.exit2(spk3)
            e2a = torch.where(active.unsqueeze(1), e2a * d + e2, e2a)
            conf2 = torch.softmax(e2a / self.temperature2, dim=1).max(dim=1)[0]
            ex2 = (conf2 > self.confidence_thresholds[1]) & active
            exit_pts[ex2] = 2; exit_ts[ex2] = t + 1
            active = active & ~ex2
            if not active.any():
                continue  # skip layers 4-8 entirely for this timestep
            x = self.q3(spk3 * active.float().unsqueeze(1))

            # --- Layers 4-8 (final exit) ---
            spk4, mems[3] = self._spiking_block(x, self.fc4, self.bn4, self.lif4, mems[3])
            x = self.q4(spk4)
            spk5, mems[4] = self._spiking_block(x, self.fc5, self.bn5, self.lif5, mems[4], self.drop5)
            x = self.q5(spk5)
            spk6, mems[5] = self._spiking_block(x, self.fc6, self.bn6, self.lif6, mems[5])
            x = self.q6(spk6)
            spk7, mems[6] = self._spiking_block(x, self.fc7, self.bn7, self.lif7, mems[6])
            x = self.q7(spk7)
            f = self._maybe_dequant(self.fc8(x))
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
            self.confidence_thresholds = confidence_thresholds
            self.rl_agent = None
        elif rl_agent is not None:
            self.rl_agent = rl_agent
            learned = rl_agent.get_confidence_thresholds()
            if learned and len(learned) == self.n_exits:
                self.confidence_thresholds = learned
        return self
