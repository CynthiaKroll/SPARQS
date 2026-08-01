"""
LeNet Models for SPARQ Framework

This module contains all LeNet model variants:
- LeNet: Standard ANN baseline
- LeNetSNN: Spiking Neural Network version
- QuantizedSNNLeNet: Quantized SNN with INT8 weights
- BranchyLeNet: Early-exit architecture
- BranchyQuantizedSNN: Full QDSNN with RL-guided exits
"""

from .lenet import LeNet
from .lenet_snn import LeNetSNN
from .quantized_snn_lenet import QuantizedSNNLeNet
from .branchy_models import BranchyLeNet, EarlyExitBlock
from .branchy_quantized_snn import BranchyQuantizedSNN, ExitClassifier
from .rl_environment import QLearningAgent

__all__ = [
    'LeNet',
    'LeNetSNN',
    'QuantizedSNNLeNet',
    'BranchyLeNet',
    'EarlyExitBlock',
    'BranchyQuantizedSNN',
    'ExitClassifier',
    'QLearningAgent',
]
