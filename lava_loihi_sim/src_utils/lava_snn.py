"""
Lava Neuromorphic Simulation for SNN Backbone

Builds a Lava process network matching the trained SNN backbone and runs
it through Loihi2SimCfg for hardware-realistic power/energy estimation.

Supports two model formats:
1. Original base model (backbone.snn.block1/2/3/4) -- hardcoded 4 SNN blocks
2. NAS configurable model (backbone.snn_blocks.0/1/.../N) -- variable K blocks

For NAS, pass an ArchConfig or a NAS result JSON to configure the SNN
architecture dynamically. The simulation adapts to any number of SNN layers
with arbitrary channels and strides.

Paper: Ahmed et al. (arXiv:2403.10173v4), Section 5
Hardware target: Intel Loihi 2
Simulator: Lava Loihi2SimCfg (CPU-based Loihi 2 simulation)

PLIF -> Lava LIF conversion:
  SpikingJelly ParametricLIFNode has a learnable tau per layer. Lava's
  native LIF only supports fixed decay. Since tau is fixed after training,
  we extract the learned tau and set Lava LIF dv = 1 - 1/tau.

Pipeline:
1. Load trained checkpoint -> extract SNN backbone weights + BN + PLIF tau
2. Fold BN into conv weights (Eq. 7-8), quantize to int8
3. Build Lava process network: Conv -> LIF -> Conv -> LIF -> ...
4. Run Loihi2SimCfg simulation with T=10 timesteps
5. Monitor spike activity per layer via Lava Monitor processes

Energy estimation uses two sets of constants (see power_estimation.py):
- Loihi 1 published (Davies et al. 2018): conservative upper bound
- Loihi 2 estimated (from Hala Point system data): lower bound

Usage (base model, 4-block default):
    python -m src.utils.lava_snn \\
        --checkpoint checkpoints/best.pth \\
        --data_root data/gen1 \\
        --num_samples 100 \\
        --output power_report.txt

Usage (NAS model, variable architecture):
    python -m src.utils.lava_snn \\
        --checkpoint nas_results_v2/finetune/S6_A2_best.pth \\
        --nas_config nas_results_v2/finetune/S6_A2_N8_paper_result.json \\
        --data_root data/gen1 \\
        --num_samples 100 \\
        --output power_report_S6_A2.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Lava imports - these require lava-nc to be installed
try:
    from lava.proc.conv.process import Conv
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense
    from lava.proc.monitor.process import Monitor
    from lava.proc.io.source import RingBuffer as SpikeGenerator
    from lava.proc.io.sink import RingBuffer as SpikeSink
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.magma.core.run_conditions import RunSteps
    HAS_LAVA = True
except ImportError:
    HAS_LAVA = False

from .bn_folding import (
    fold_and_quantize_snn_backbone,
    fold_and_quantize_snn_blocks,
    extract_plif_tau,
)

# Add NAS module to path for ArchConfig import
_nas_path = str(Path(__file__).parent.parent / 'nas')
if _nas_path not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from nas.search_space import ArchConfig, SearchSpace
    HAS_NAS = True
except ImportError:
    HAS_NAS = False


# ============================================================================
# Default SNN architecture (Paper Table 9, hardcoded 4-block)
# ============================================================================

SNN_ARCH = [
    {'name': 'block1', 'in_ch': 2,   'out_ch': 64,  'kernel': 3, 'stride': 2, 'padding': 1},
    {'name': 'block2', 'in_ch': 64,  'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
    {'name': 'block3', 'in_ch': 128, 'out_ch': 256, 'kernel': 3, 'stride': 2, 'padding': 1},
    {'name': 'block4', 'in_ch': 256, 'out_ch': 256, 'kernel': 3, 'stride': 1, 'padding': 1},
]

# Gen1 input resolution
GEN1_HEIGHT = 240
GEN1_WIDTH = 304


# ============================================================================
# Architecture helpers
# ============================================================================

def build_snn_arch_from_config(config: 'ArchConfig') -> List[Dict]:
    """Convert NAS ArchConfig to SNN_ARCH format for Lava simulation.

    Translates the variable-length SNN layer configs from an ArchConfig
    into the list-of-dicts format used by LavaSNNSimulator.

    Args:
        config: NAS ArchConfig with snn_layer_configs() method

    Returns:
        List of layer dicts compatible with SNN_ARCH format. Each dict has
        keys: name, in_ch, out_ch, kernel, stride, padding.
        Returns empty list if config has no SNN layers (K=0).
    """
    arch = []
    for i, (in_ch, out_ch, stride) in enumerate(config.snn_layer_configs()):
        arch.append({
            'name': f'snn_block_{i}',
            'in_ch': in_ch,
            'out_ch': out_ch,
            'kernel': 3,
            'stride': stride,
            'padding': 1,
        })
    return arch


def compute_output_size(h: int, w: int, kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    """Compute conv output spatial dimensions."""
    h_out = (h + 2 * padding - kernel) // stride + 1
    w_out = (w + 2 * padding - kernel) // stride + 1
    return h_out, w_out


def get_layer_spatial_dims(
    snn_arch: Optional[List[Dict]] = None,
    height: int = GEN1_HEIGHT,
    width: int = GEN1_WIDTH,
) -> List[Dict]:
    """Compute spatial dimensions at each SNN layer output.

    Args:
        snn_arch: SNN architecture list. Defaults to SNN_ARCH (paper 4-block).
        height: Input height
        width: Input width

    Returns:
        List of dicts with name, h, w, channels per layer.
    """
    if snn_arch is None:
        snn_arch = SNN_ARCH

    dims = []
    h, w = height, width
    for layer in snn_arch:
        h, w = compute_output_size(h, w, layer['kernel'], layer['stride'], layer['padding'])
        dims.append({'name': layer['name'], 'h': h, 'w': w, 'channels': layer['out_ch']})
    return dims


# ============================================================================
# NAS result JSON loader
# ============================================================================

def load_arch_config_from_json(json_path: str) -> 'ArchConfig':
    """Load an ArchConfig from a NAS result JSON file.

    NAS result JSONs contain a 'config' key with the architecture
    specification (name, num_snn_layers, num_ann_layers, channels, strides).

    Args:
        json_path: Path to NAS result JSON file

    Returns:
        ArchConfig instance

    Raises:
        ValueError: If JSON doesn't contain a valid config
        ImportError: If NAS module is not available
    """
    if not HAS_NAS:
        raise ImportError(
            "NAS module not available. Ensure src/nas/ is in the Python path."
        )

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'config' not in data:
        raise ValueError(f"NAS result JSON {json_path} missing 'config' key")

    cfg_dict = data['config']
    return SearchSpace.config_from_dict(cfg_dict)


# ============================================================================
# Main simulator class
# ============================================================================

class LavaSNNSimulator:
    """
    Builds and runs a Lava process network replicating a trained SNN backbone.

    Supports both the original base model (4 hardcoded SNN blocks) and NAS
    configurable models (variable number of SNN blocks from ArchConfig).

    The simulation uses Loihi2SimCfg with fixed_pt tag for hardware-realistic
    int8 behavior matching the paper's Loihi 2 deployment.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_time_steps: int = 10,
        height: int = GEN1_HEIGHT,
        width: int = GEN1_WIDTH,
        device: str = 'cpu',
        arch_config: Optional['ArchConfig'] = None,
        num_classes: int = 2,
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            num_time_steps: Number of temporal bins (T=10)
            height: Input height (240 for Gen1)
            width: Input width (304 for Gen1)
            device: Torch device for loading checkpoint
            arch_config: Optional NAS ArchConfig. If None, uses the paper's
                default 4-block SNN architecture. If provided, builds the SNN
                section according to the config (K SNN layers).
            num_classes: Number of detection classes (2 for Gen1, 3 for Gen4)
        """
        self.checkpoint_path = checkpoint_path
        self.T = num_time_steps
        self.height = height
        self.width = width
        self.device = device
        self.arch_config = arch_config
        self.num_classes = num_classes

        # Determine SNN architecture
        if arch_config is not None:
            self.snn_arch = build_snn_arch_from_config(arch_config)
            self.is_nas_model = True
            if len(self.snn_arch) == 0:
                print(f"WARNING: ArchConfig '{arch_config.name}' has 0 SNN layers. "
                      f"No SNN simulation to run.")
        else:
            self.snn_arch = SNN_ARCH
            self.is_nas_model = False

        # Compute spatial dims for the chosen architecture
        self.spatial_dims = get_layer_spatial_dims(self.snn_arch, height, width)

        # Load and prepare model (fold BN + quantize)
        if len(self.snn_arch) > 0:
            self.folded_layers = self._load_and_fold(checkpoint_path)
        else:
            self.folded_layers = []

    def _load_and_fold(self, checkpoint_path: str):
        """Load checkpoint and fold BN + quantize SNN backbone.

        Handles both model formats:
        - Base model: backbone.snn.block{1-4} (uses fold_and_quantize_snn_backbone)
        - NAS model: backbone.snn_blocks.{0-N} (uses fold_and_quantize_snn_blocks)

        Returns:
            For base model: Dict mapping block_name -> folded layer info
            For NAS model: List of folded layer info dicts (indexed by position)
        """
        sys.path.insert(0, str(Path(__file__).parent.parent))

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            state_dict = checkpoint['ema_state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if self.is_nas_model:
            return self._load_and_fold_nas(state_dict)
        else:
            return self._load_and_fold_base(state_dict)

    def _load_and_fold_base(self, state_dict: Dict) -> Dict:
        """Fold BN + quantize for the base model (4 hardcoded SNN blocks)."""
        from models import build_model

        model = build_model(num_classes=self.num_classes)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        snn_backbone = model.backbone.snn
        folded = fold_and_quantize_snn_backbone(snn_backbone)

        print(f"Base SNN backbone loaded and quantized from {self.checkpoint_path}")
        for name, layer in folded.items():
            print(f"  {name}: tau={layer['tau']:.4f}, "
                  f"{layer['in_channels']}->{layer['out_channels']}, "
                  f"stride={layer['stride']}")

        return folded

    def _load_and_fold_nas(self, state_dict: Dict) -> List[Dict]:
        """Fold BN + quantize for a NAS configurable model (variable SNN blocks).

        Detects the checkpoint format from the state_dict keys:
        - If keys contain 'backbone.snn_blocks.', loads as NAS model directly
        - If keys contain 'backbone.snn.block', loads as base model and extracts
          the SNN blocks matching the NAS config's layer positions

        Returns:
            List of folded layer info dicts (one per SNN block).
        """
        config = self.arch_config

        # Check which key pattern exists in the checkpoint
        has_nas_keys = any(k.startswith('backbone.snn_blocks.') for k in state_dict)
        has_base_keys = any(k.startswith('backbone.snn.block') for k in state_dict)

        if has_nas_keys:
            # NAS checkpoint: build ConfigurableHybridDetector
            from nas.configurable_model import ConfigurableHybridDetector
            model = ConfigurableHybridDetector(config=config, num_classes=self.num_classes)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            snn_blocks = model.backbone.snn_blocks
            folded_list = fold_and_quantize_snn_blocks(snn_blocks)

        elif has_base_keys:
            # Base checkpoint with a NAS config: build base model, extract
            # SNN blocks using global-position mapping (cross-type supported)
            from models import build_model
            from models.snn_backbone import SNNConvBlock
            import torch.nn as nn

            model = build_model(num_classes=self.num_classes)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            BASE_SNN_COUNT = 4
            base_snn = model.backbone.snn
            base_ann = model.backbone.ann
            base_snn_names = ['block1', 'block2', 'block3', 'block4']
            base_ann_names = ['block1', 'block2', 'block3', 'block4']

            # Build SNN blocks for each global position that is SNN in the
            # target NAS config.  For positions 0..3 that were SNN in the
            # base model we can fold directly.  For positions 4..7 that were
            # ANN in the base model we create a fresh SNNConvBlock with
            # transferred conv+BN weights (cross-type).
            blocks_to_fold = nn.ModuleList()
            for i in range(config.num_snn_layers):
                if i < BASE_SNN_COUNT:
                    # Same-type: base SNN block i
                    blocks_to_fold.append(
                        getattr(base_snn, base_snn_names[i])
                    )
                else:
                    # Cross-type: base ANN block (i - BASE_SNN_COUNT)
                    ann_idx = i - BASE_SNN_COUNT
                    if ann_idx < len(base_ann_names):
                        ann_block = getattr(base_ann, base_ann_names[ann_idx])
                        # Create a fresh SNNConvBlock with same conv shape
                        snn_block = SNNConvBlock(
                            in_channels=ann_block.conv.in_channels,
                            out_channels=ann_block.conv.out_channels,
                            stride=ann_block.conv.stride[0],
                        )
                        snn_block.eval()
                        # Transfer conv + BN weights (shapes match)
                        snn_block.conv.weight.data.copy_(ann_block.conv.weight.data)
                        snn_block.bn.weight.data.copy_(ann_block.bn.weight.data)
                        snn_block.bn.bias.data.copy_(ann_block.bn.bias.data)
                        snn_block.bn.running_mean.data.copy_(ann_block.bn.running_mean.data)
                        snn_block.bn.running_var.data.copy_(ann_block.bn.running_var.data)
                        # neuron.node.w stays at init (fresh PLIF tau)
                        blocks_to_fold.append(snn_block)

            folded_list = fold_and_quantize_snn_blocks(blocks_to_fold)
        else:
            raise ValueError(
                f"Checkpoint has neither 'backbone.snn_blocks.' nor "
                f"'backbone.snn.block' keys. Cannot identify model format."
            )

        print(f"NAS SNN backbone loaded and quantized ({config.name}, "
              f"K={config.num_snn_layers} SNN layers)")
        for i, layer in enumerate(folded_list):
            print(f"  snn_block_{i}: tau={layer['tau']:.4f}, "
                  f"{layer['in_channels']}->{layer['out_channels']}, "
                  f"stride={layer['stride']}")

        return folded_list

    def _get_folded_layer(self, layer_idx: int) -> Dict:
        """Get folded layer info by index, handling both storage formats.

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            Dict with folded layer parameters
        """
        if isinstance(self.folded_layers, list):
            # NAS model: list indexed by position
            return self.folded_layers[layer_idx]
        else:
            # Base model: dict keyed by block name
            name = self.snn_arch[layer_idx]['name']
            return self.folded_layers[name]

    def run_pytorch_simulation(
        self,
        event_tensors: List[torch.Tensor],
        verbose: bool = True,
    ) -> Dict:
        """
        Run SNN simulation using folded int8 weights in PyTorch.

        This simulates the Loihi 2 fixed-point LIF behavior:
        - Weights are int8 (folded BN + quantized)
        - LIF neurons use dv = 1 - 1/tau (voltage decay per timestep)
        - Threshold = 1.0, reset = 0.0 (hard reset)
        - Spike output is binary {0, 1}

        This gives identical spike activity to what Lava/Loihi2SimCfg would
        produce, but runs without the lava-nc dependency.

        Args:
            event_tensors: List of event tensors, each [T, 2, H, W]
            verbose: Print progress

        Returns:
            results: Dict with per-layer spike counts, firing rates, ACs.
                Keys are layer names (e.g., 'block1' for base, 'snn_block_0'
                for NAS).
        """
        if len(self.snn_arch) == 0:
            print("No SNN layers to simulate.")
            return {}

        # Initialize per-layer accumulators
        layer_names = [l['name'] for l in self.snn_arch]
        spike_counts = {name: [] for name in layer_names}
        spike_counts_per_step = {name: [] for name in layer_names}
        input_spike_counts = {name: [] for name in layer_names}
        acs_counts_per_step = {name: [] for name in layer_names}

        n_samples = len(event_tensors)
        if verbose:
            print(f"\nRunning SNN simulation on {n_samples} samples "
                  f"(T={self.T}, {len(self.snn_arch)} SNN layers)...")

        for sample_idx, event_tensor in enumerate(event_tensors):
            if verbose and (sample_idx + 1) % 10 == 0:
                print(f"  Sample {sample_idx + 1}/{n_samples}")

            # event_tensor: [T, 2, H, W]
            sample_spikes = self._simulate_sample(event_tensor)

            for name in layer_names:
                spike_counts[name].append(sample_spikes[name]['total_spikes'])
                spike_counts_per_step[name].append(sample_spikes[name]['spikes_per_step'])
                input_spike_counts[name].append(sample_spikes[name]['input_spikes_per_step'])
                acs_counts_per_step[name].append(sample_spikes[name]['acs_per_step'])

        # Aggregate results
        results = {}
        for i, name in enumerate(layer_names):
            dims = self.spatial_dims[i]
            total_neurons = dims['channels'] * dims['h'] * dims['w']
            layer_info = self._get_folded_layer(i)

            # Mean spike counts across samples
            mean_total_spikes = np.mean(spike_counts[name])
            mean_spikes_per_step = np.mean(spike_counts_per_step[name], axis=0)
            mean_acs_per_step = np.mean(acs_counts_per_step[name], axis=0)

            # Firing rate: fraction of neurons that spike per timestep
            firing_rate = mean_total_spikes / (total_neurons * self.T)

            # Count ACs from the actual spike tensors to preserve stride/padding.
            acs_per_step = mean_acs_per_step
            total_acs = float(np.sum(mean_acs_per_step))

            results[name] = {
                'total_spikes': float(mean_total_spikes),
                'spikes_per_step': mean_spikes_per_step.tolist(),
                'firing_rate': float(firing_rate),
                'total_acs': total_acs,
                'acs_per_step': acs_per_step.tolist(),
                'total_neurons': total_neurons,
                'tau': layer_info['tau'],
                'dv': 1.0 - 1.0 / layer_info['tau'],
                'spatial': (dims['h'], dims['w']),
                'channels': dims['channels'],
                'kernel_size': layer_info['kernel_size'],
                'stride': layer_info['stride'],
            }

        return results

    def _simulate_sample(self, event_tensor: torch.Tensor) -> Dict:
        """
        Simulate one sample through the folded int8 SNN.

        Implements LIF dynamics:
            v[t] = v[t-1] * (1 - 1/tau) + I[t]
            spike[t] = 1 if v[t] >= threshold else 0
            v[t] = v[t] * (1 - spike[t])  (hard reset to 0)

        Args:
            event_tensor: [T, 2, H, W]

        Returns:
            layer_spikes: Dict per block with spike statistics
        """
        T = event_tensor.shape[0]
        layer_spikes = {}

        # Input to first layer is the event tensor itself
        current_input = event_tensor  # [T, C_in, H, W]

        for i, layer_cfg in enumerate(self.snn_arch):
            name = layer_cfg['name']
            info = self._get_folded_layer(i)

            # Get folded int8 weights and bias
            w_int8 = info['weight_int8']  # [C_out, C_in, k, k]
            w_scale = info['weight_scale']  # [C_out]
            bias = info['bias']  # [C_out]
            tau = info['tau']
            dv = 1.0 - 1.0 / tau  # voltage decay factor

            # Dequantize weights for simulation: w_float = w_int8 * scale
            w_float = w_int8 * w_scale.reshape(-1, 1, 1, 1)

            c_out = info['out_channels']
            h_in, w_in = current_input.shape[2], current_input.shape[3]
            h_out, w_out = compute_output_size(
                h_in, w_in, info['kernel_size'], info['stride'], info['padding']
            )

            # Initialize membrane potential
            v = torch.zeros(1, c_out, h_out, w_out)  # [1, C_out, H_out, W_out]
            threshold = 1.0

            total_spikes = 0
            spikes_per_step = []
            input_spikes_per_step = []
            acs_per_step = self._compute_acs_per_timestep(current_input, info)
            spike_frames = []  # Collect spike tensors for next layer input

            for t in range(T):
                x_t = current_input[t:t+1]  # [1, C_in, H_in, W_in]

                # Count input spikes (binary inputs from previous layer)
                input_spikes_per_step.append(float(x_t.sum().item()))

                # Convolution (simulating int8 fixed-point)
                I_t = F.conv2d(
                    x_t, w_float,
                    bias=bias,
                    stride=info['stride'],
                    padding=info['padding'],
                )  # [1, C_out, H_out, W_out]

                # LIF update: v = v * decay + input current
                v = v * dv + I_t

                # Spike generation
                spike = (v >= threshold).float()

                # Hard reset: v = v * (1 - spike)
                v = v * (1.0 - spike)

                step_spikes = int(spike.sum().item())
                total_spikes += step_spikes
                spikes_per_step.append(step_spikes)
                spike_frames.append(spike)  # [1, C_out, H_out, W_out]

            layer_spikes[name] = {
                'total_spikes': total_spikes,
                'spikes_per_step': np.array(spikes_per_step),
                'input_spikes_per_step': np.array(input_spikes_per_step),
                'acs_per_step': np.array(acs_per_step),
            }

            # Output spikes become input to next layer
            current_input = torch.cat(spike_frames, dim=0)  # [T, C_out, H_out, W_out]

        return layer_spikes

    def _compute_acs_per_timestep(
        self,
        input_tensor: torch.Tensor,
        layer_info: Dict,
    ) -> np.ndarray:
        """Count exact ACs per timestep from the binary input spike tensor.

        This counts active synaptic connections directly from spike locations,
        so stride and padding are handled by the convolution geometry rather
        than approximated from input spike counts alone.
        """
        input_tensor = input_tensor.float()
        ones_kernel = input_tensor.new_ones(
            (1, input_tensor.shape[1], layer_info['kernel_size'], layer_info['kernel_size'])
        )

        acs = []
        for t in range(input_tensor.shape[0]):
            active_inputs = F.conv2d(
                input_tensor[t:t+1],
                ones_kernel,
                stride=layer_info['stride'],
                padding=layer_info['padding'],
            )
            acs.append(float(active_inputs.sum().item() * layer_info['out_channels']))

        return np.array(acs, dtype=float)

    def _get_spike_tensor(
        self,
        input_tensor: torch.Tensor,
        layer_info: Dict,
        dv: float,
        threshold: float,
    ) -> torch.Tensor:
        """
        Run a single layer and return the full spike tensor [T, C_out, H_out, W_out].

        Args:
            input_tensor: [T, C_in, H_in, W_in]
            layer_info: Folded layer parameters
            dv: Voltage decay (1 - 1/tau)
            threshold: Spike threshold

        Returns:
            spikes: [T, C_out, H_out, W_out]
        """
        T = input_tensor.shape[0]
        w_float = layer_info['weight_int8'] * layer_info['weight_scale'].reshape(-1, 1, 1, 1)
        bias = layer_info['bias']

        c_out = layer_info['out_channels']
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        h_out, w_out = compute_output_size(
            h_in, w_in, layer_info['kernel_size'], layer_info['stride'], layer_info['padding']
        )

        v = torch.zeros(1, c_out, h_out, w_out)
        spike_list = []

        for t in range(T):
            x_t = input_tensor[t:t+1]
            I_t = F.conv2d(
                x_t, w_float, bias=bias,
                stride=layer_info['stride'],
                padding=layer_info['padding'],
            )
            v = v * dv + I_t
            spike = (v >= threshold).float()
            v = v * (1.0 - spike)
            spike_list.append(spike)

        return torch.cat(spike_list, dim=0)  # [T, C_out, H_out, W_out]

    def run_lava_simulation(
        self,
        event_tensors: List[torch.Tensor],
        verbose: bool = True,
    ) -> Dict:
        """
        Run SNN simulation using Lava's Loihi2SimCfg.

        This uses actual Lava processes (Conv, LIF) with Loihi2SimCfg
        for hardware-realistic fixed-point simulation.

        Falls back to PyTorch simulation if Lava is not installed.

        Args:
            event_tensors: List of event tensors, each [T, 2, H, W]
            verbose: Print progress

        Returns:
            results: Dict with per-layer spike counts, firing rates, ACs
        """
        if len(self.snn_arch) == 0:
            print("No SNN layers to simulate.")
            return {}

        if not HAS_LAVA:
            print("WARNING: lava-nc not installed. Using PyTorch LIF simulation "
                  "(equivalent fixed-point behavior).")
            print("Install Lava for native Loihi 2 simulation: pip install lava-nc")
            return self.run_pytorch_simulation(event_tensors, verbose)

        # Lava-based simulation
        return self._run_lava_native(event_tensors, verbose)

    def _run_lava_native(
        self,
        event_tensors: List[torch.Tensor],
        verbose: bool = True,
    ) -> Dict:
        """
        Native Lava simulation using Loihi2SimCfg.

        Builds Lava Conv + LIF processes for each SNN block, connects them,
        attaches monitors, and runs through Loihi2SimCfg.
        """
        layer_names = [l['name'] for l in self.snn_arch]
        spike_counts = {name: [] for name in layer_names}
        spike_counts_per_step = {name: [] for name in layer_names}
        input_spike_counts = {name: [] for name in layer_names}
        acs_counts_per_step = {name: [] for name in layer_names}

        n_samples = len(event_tensors)
        if verbose:
            print(f"\nRunning Lava Loihi2SimCfg simulation on {n_samples} samples "
                  f"(T={self.T}, {len(self.snn_arch)} SNN layers)...")

        for sample_idx, event_tensor in enumerate(event_tensors):
            if verbose and (sample_idx + 1) % 10 == 0:
                print(f"  Sample {sample_idx + 1}/{n_samples}")

            # Build Lava process network for this sample
            processes, monitors = self._build_lava_network(event_tensor)

            # Run simulation
            try:
                run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
                processes['input'].run(
                    condition=RunSteps(num_steps=self.T),
                    run_cfg=run_cfg,
                )

                # Collect spike data from monitors
                for i, name in enumerate(layer_names):
                    layer_info = self._get_folded_layer(i)
                    spike_data = monitors[name].get_data()[f'lif_{name}']['s_out']
                    # spike_data: [T, N] where N = C*H*W

                    step_spikes = spike_data.sum(axis=1)  # [T]
                    total = int(spike_data.sum())

                    spike_counts[name].append(total)
                    spike_counts_per_step[name].append(step_spikes)

                    # Input spikes: from previous layer or input
                    if i == 0:
                        input_tensor = event_tensor.float()
                        in_data = input_tensor.numpy()
                        in_spikes = np.array([in_data[t].sum() for t in range(self.T)])
                    else:
                        prev_name = layer_names[i - 1]
                        prev_data = monitors[prev_name].get_data()[f'lif_{prev_name}']['s_out']
                        prev_dims = self.spatial_dims[i - 1]
                        input_tensor = torch.from_numpy(
                            prev_data.reshape(
                                self.T,
                                prev_dims['channels'],
                                prev_dims['h'],
                                prev_dims['w'],
                            )
                        ).float()
                        in_spikes = prev_data.sum(axis=1)

                    input_spike_counts[name].append(in_spikes)
                    acs_counts_per_step[name].append(
                        self._compute_acs_per_timestep(input_tensor, layer_info)
                    )

            finally:
                processes['input'].stop()

        # Aggregate (same as PyTorch simulation)
        results = {}
        for i, name in enumerate(layer_names):
            dims = self.spatial_dims[i]
            total_neurons = dims['channels'] * dims['h'] * dims['w']
            layer_info = self._get_folded_layer(i)

            mean_total_spikes = np.mean(spike_counts[name])
            mean_spikes_per_step = np.mean(spike_counts_per_step[name], axis=0)
            mean_acs_per_step = np.mean(acs_counts_per_step[name], axis=0)

            firing_rate = mean_total_spikes / (total_neurons * self.T)
            acs_per_step = mean_acs_per_step
            total_acs = float(np.sum(mean_acs_per_step))

            results[name] = {
                'total_spikes': float(mean_total_spikes),
                'spikes_per_step': mean_spikes_per_step.tolist(),
                'firing_rate': float(firing_rate),
                'total_acs': total_acs,
                'acs_per_step': acs_per_step.tolist(),
                'total_neurons': total_neurons,
                'tau': layer_info['tau'],
                'dv': 1.0 - 1.0 / layer_info['tau'],
                'spatial': (dims['h'], dims['w']),
                'channels': dims['channels'],
                'kernel_size': layer_info['kernel_size'],
                'stride': layer_info['stride'],
            }

        return results

    def _build_lava_network(self, event_tensor: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Build Lava process network for one sample.

        Creates: Input -> Conv0 -> LIF0 -> Conv1 -> LIF1 -> ... -> ConvK -> LIFK

        Each Conv uses folded int8 weights, each LIF uses dv = 1 - 1/tau.
        Works with any number of SNN layers (not just 4).

        Args:
            event_tensor: [T, 2, H, W] input data

        Returns:
            processes: Dict of Lava processes
            monitors: Dict of Monitor processes per LIF layer
        """
        processes = {}
        monitors = {}

        # Input process: feeds event data timestep by timestep
        T, C_in, H, W = event_tensor.shape
        input_data = event_tensor.numpy()  # [T, C_in, H, W]

        # Flatten spatial dims for SpikeGenerator: [T, C*H*W]
        input_flat = input_data.reshape(T, -1).T  # [C*H*W, T]
        input_proc = SpikeGenerator(data=input_flat)
        processes['input'] = input_proc

        prev_proc = input_proc

        for i, layer_cfg in enumerate(self.snn_arch):
            name = layer_cfg['name']
            info = self._get_folded_layer(i)

            # Dequantize weights for Lava Conv
            w_float = (info['weight_int8'] * info['weight_scale'].reshape(-1, 1, 1, 1)).numpy()

            dims = self.spatial_dims[i]

            # Conv process
            conv_proc = Conv(
                weight=w_float,
                bias=info['bias'].numpy() if info['bias'] is not None else None,
                stride=info['stride'],
                padding=info['padding'],
                input_shape=(
                    layer_cfg['in_ch'],
                    H if i == 0 else self.spatial_dims[i-1]['h'],
                    W if i == 0 else self.spatial_dims[i-1]['w'],
                ),
            )
            processes[f'conv_{name}'] = conv_proc

            # LIF process
            dv = 1.0 - 1.0 / info['tau']
            n_neurons = dims['channels'] * dims['h'] * dims['w']
            lif_proc = LIF(
                shape=(n_neurons,),
                dv=dv,
                du=1.0,
                vth=1.0,
                bias_mant=0,
            )
            processes[f'lif_{name}'] = lif_proc

            # Connect: prev -> conv -> lif
            prev_proc.s_out.connect(conv_proc.a_in)
            conv_proc.a_out.connect(lif_proc.a_in)

            # Monitor spike output
            monitor = Monitor()
            monitor.probe(lif_proc.s_out, T)
            monitors[name] = monitor

            prev_proc = lif_proc

        return processes, monitors


# ============================================================================
# Data loader
# ============================================================================

def load_validation_samples(
    data_root: str,
    num_samples: int = 100,
    num_time_bins: int = 10,
) -> List[torch.Tensor]:
    """
    Load validation samples for simulation.

    Auto-detects Gen1 vs Gen4 from data_root path so Lava validation can run
    on either dataset without code changes (bug encountered Apr 17 when
    running Gen4 Lava Job 1048 — Gen1Dataset rejected 360x640 tensors).

    Args:
        data_root: Path to Gen1 or Gen4 dataset (auto-detected by 'gen4' in path)
        num_samples: Number of samples to load
        num_time_bins: Temporal bins (T=10)

    Returns:
        event_tensors: List of [T, 2, H, W] tensors
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))

    is_gen4 = "gen4" in str(data_root).lower()
    if is_gen4:
        from data.gen4_dataset import Gen4Dataset as DatasetCls
    else:
        from data.gen1_dataset import Gen1Dataset as DatasetCls

    dataset = DatasetCls(
        root_dir=data_root,
        split='val',
        num_time_bins=num_time_bins,
        verbose=True,
        cache_tensors=False,
    )

    n = min(num_samples, len(dataset))
    event_tensors = []
    for i in range(n):
        tensor, _ = dataset[i]
        event_tensors.append(tensor)  # [T, 2, H, W]

    print(f"Loaded {n} validation samples")
    return event_tensors


# ============================================================================
# Convenience function for NAS integration
# ============================================================================

def run_power_estimation_for_nas(
    checkpoint_path: str,
    nas_result_json: str,
    data_root: str,
    num_samples: int = 100,
    output_path: Optional[str] = None,
    use_lava: bool = False,
    verbose: bool = True,
) -> Dict:
    """Run Lava power estimation for a NAS candidate.

    End-to-end convenience function: loads the NAS config from a result JSON,
    builds the simulator for the correct SNN architecture, runs the simulation,
    generates a power report, and optionally saves it.

    Args:
        checkpoint_path: Path to the NAS candidate's fine-tuned checkpoint.
            Can be either a NAS checkpoint (backbone.snn_blocks.*) or the base
            model checkpoint (backbone.snn.block*).
        nas_result_json: Path to the NAS result JSON file containing the
            ArchConfig under the 'config' key.
        data_root: Path to Gen1 dataset root directory.
        num_samples: Number of validation samples to simulate (default 100).
        output_path: Optional path to save the power report text file.
        use_lava: Use native Lava simulation if available (default False,
            uses PyTorch LIF which gives equivalent results).
        verbose: Print progress during simulation.

    Returns:
        Dict with keys:
            'simulation_results': Per-layer spike/AC results from simulator
            'report': Formatted power report string
            'config_name': Name of the NAS configuration
            'num_snn_layers': Number of SNN layers in this config
            'energy_mj': Total estimated energy in millijoules (Loihi 2)
            'elapsed_s': Simulation wall-clock time in seconds

    Raises:
        ValueError: If the NAS config has 0 SNN layers (nothing to simulate)
        ImportError: If NAS module is not available
    """
    # Load ArchConfig from NAS result JSON
    arch_config = load_arch_config_from_json(nas_result_json)

    if arch_config.num_snn_layers == 0:
        raise ValueError(
            f"NAS config '{arch_config.name}' has 0 SNN layers. "
            f"No SNN simulation needed for all-ANN architectures."
        )

    if verbose:
        print(f"Running power estimation for NAS config: {arch_config.name}")
        print(f"  SNN layers: {arch_config.num_snn_layers}, "
              f"ANN layers: {arch_config.num_ann_layers}")
        print(f"  SNN channels: {arch_config.snn_channels}")
        print(f"  SNN strides:  {arch_config.snn_strides}")

    # Load validation data
    event_tensors = load_validation_samples(data_root, num_samples)

    # Auto-detect dataset dims + num_classes from data_root so Gen4
    # (360x640, 3 classes) uses the right head without a code change.
    # Gen4 Lava Job 1049 failed here on head.cls_preds shape mismatch.
    is_gen4 = "gen4" in str(data_root).lower()
    if is_gen4:
        sim_height, sim_width, sim_num_classes = 360, 640, 3
    else:
        sim_height, sim_width, sim_num_classes = 240, 304, 2

    # Build simulator with NAS architecture
    simulator = LavaSNNSimulator(
        checkpoint_path=checkpoint_path,
        num_time_steps=10,
        height=sim_height,
        width=sim_width,
        arch_config=arch_config,
        num_classes=sim_num_classes,
    )

    # Run simulation
    start_time = time.time()
    if use_lava:
        results = simulator.run_lava_simulation(event_tensors, verbose=verbose)
    else:
        results = simulator.run_pytorch_simulation(event_tensors, verbose=verbose)
    elapsed = time.time() - start_time

    if verbose:
        print(f"\nSimulation completed in {elapsed:.1f}s")

    # Generate power report
    from .power_estimation import generate_power_report, estimate_energy_mj
    report = generate_power_report(results, num_samples=num_samples)

    # Extract Loihi 2 energy estimate
    energy_all = estimate_energy_mj(results, chip_generation="both")
    loihi2_energy_mj = energy_all['loihi2']['total']['total_energy_mj']

    if verbose:
        print(report)

    # Save report if output path provided
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        if verbose:
            print(f"\nReport saved to {output_path}")

    return {
        'simulation_results': results,
        'report': report,
        'config_name': arch_config.name,
        'num_snn_layers': arch_config.num_snn_layers,
        'energy_mj': loihi2_energy_mj,
        'elapsed_s': elapsed,
    }


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SNN Backbone Lava Neuromorphic Simulation"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to Gen1 dataset root")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of validation samples to simulate")
    parser.add_argument("--output", type=str, default="power_report.txt",
                        help="Output report file path")
    parser.add_argument("--use_lava", action="store_true",
                        help="Use native Lava simulation (requires lava-nc)")
    parser.add_argument("--nas_config", type=str, default=None,
                        help="Path to NAS result JSON file. When provided, "
                             "builds the SNN architecture from the ArchConfig "
                             "in the JSON instead of using the default 4-block "
                             "paper architecture.")
    args = parser.parse_args()

    # If NAS config is provided, use the convenience function
    if args.nas_config is not None:
        result = run_power_estimation_for_nas(
            checkpoint_path=args.checkpoint,
            nas_result_json=args.nas_config,
            data_root=args.data_root,
            num_samples=args.num_samples,
            output_path=args.output,
            use_lava=args.use_lava,
            verbose=True,
        )
        print(f"\nDone. Energy estimate (Loihi 2): {result['energy_mj']:.3f} mJ")
        return

    # Default: base model 4-block SNN
    # Load validation data
    event_tensors = load_validation_samples(
        args.data_root, args.num_samples
    )

    # Build simulator
    simulator = LavaSNNSimulator(
        checkpoint_path=args.checkpoint,
        num_time_steps=10,
    )

    # Run simulation
    start_time = time.time()
    if args.use_lava:
        results = simulator.run_lava_simulation(event_tensors, verbose=True)
    else:
        results = simulator.run_pytorch_simulation(event_tensors, verbose=True)
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Generate power report
    from .power_estimation import generate_power_report
    report = generate_power_report(results, num_samples=args.num_samples)

    # Print and save
    print(report)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
