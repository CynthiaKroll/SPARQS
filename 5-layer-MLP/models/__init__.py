"""
5-Layer MLP Models for SPARQ Framework

This module contains all MLP model variants:
- MLP: Standard ANN baseline (5 fully-connected layers)
- MLPSNN: Spiking Neural Network version
- QuantizedSNNMLP: Quantized SNN with INT8 weights
- BranchyMLP: Early-exit architecture
- BranchyQuantizedSNNMLP: Full QDSNN with RL-guided exits
"""

from .mlp import MLP
from .mlp_snn import MLPSNN
from .quantized_snn_mlp import QuantizedSNNMLP
from .branchy_mlp import BranchyMLP, EarlyExitBlock
from .branchy_quantized_snn_mlp import BranchyQuantizedSNNMLP, ExitClassifier
from .rl_environment import QLearningAgent

__all__ = [
    'MLP',
    'MLPSNN',
    'QuantizedSNNMLP',
    'BranchyMLP',
    'EarlyExitBlock',
    'BranchyQuantizedSNNMLP',
    'ExitClassifier',
    'QLearningAgent',
]
