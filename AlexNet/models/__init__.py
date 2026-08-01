"""
AlexNet Models for SPARQ Framework

This module contains all AlexNet model variants:
- AlexNet: Standard ANN baseline
- AlexNetSNN: Spiking Neural Network version
- AlexNetQCFS: QCFS ANN (for ANN->SNN conversion)
- QuantizedSNNAlexNet: Quantized SNN with INT8 weights (QSNN baseline)
- BranchyAlexNet: Early-exit ANN architecture
- BranchyQuantizedSNN: QDSNN -- the QSNN backbone plus RL-guided early exits
- QLearningAgent: RL agent that learns the early-exit policy

NOTE: this file used to import `BranchyQuantizedSNNAlexNet` and `RLEnvironment`,
neither of which exists, so every `from models.X import Y` raised ImportError and
train.py/evaluate.py could not start at all.
"""

from .alexnet import AlexNet
from .alexnet_snn import AlexNetSNN
from .alexnet_qcfs import AlexNetQCFS
from .quantized_snn_alexnet import QuantizedSNNAlexNet
from .branchy_models import BranchyAlexNet
from .branchy_quantized_snn import BranchyQuantizedSNN, ExitClassifier
from .rl_environment import QLearningAgent

__all__ = [
    'AlexNet',
    'AlexNetSNN',
    'AlexNetQCFS',
    'QuantizedSNNAlexNet',
    'BranchyAlexNet',
    'BranchyQuantizedSNN',
    'ExitClassifier',
    'QLearningAgent',
]
