# SPARQS: Spiking Early-Exit Quantized Neural Networks for Safety-Critical Edge Deployment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org)
[![SNNTorch](https://img.shields.io/badge/SNNTorch-0.6+-green.svg)](https://snntorch.readthedocs.io/)

Official implementation of **SPARQS: Spiking Early-Exit Quantized Neural Networks for Safety-Critical Edge Deployment**.

## Abstract

Spiking neural networks (SNNs) offer inherent energy efficiency due to their event-driven computation model, making them promising for edge AI deployment. However, their practical adoption is limited by the computational overhead of deep architectures and the absence of input-adaptive control.

SPARQS presents a unified framework that integrates:
- **Spiking computation** using Leaky Integrate-and-Fire (LIF) neurons
- **Quantization-aware training (QAT)** for INT8 inference
- **Reinforcement learning-guided early exits** for adaptive inference

The resulting **Quantized Dynamic SNNs (QDSNNs)** consistently outperform conventional SNNs and QSNNs.

## Key Results

| Metric | Improvement |
|--------|-------------|
| Accuracy | Up to **3.3% higher** than baseline SNNs |
| Energy | Up to **4x lower** consumption |
| Operations | Over **90x fewer** operations |
| AlexNet ops reduction | **742x** (200.52M MACs → 0.27M ACs) |

### Performance Comparison (AlexNet on CIFAR-10)

| Model | Accuracy | Operations | Energy |
|-------|----------|------------|--------|
| AlexNet (ANN) | 89.50% | 200.52M MACs | - |
| SNN (T=32) | 77.01% | 26.06M ACs | 888.0 mJ |
| QSNN (T=4) | 74.30% | 0.46M ACs | 2.82 mJ |
| **QDSNN (Ours)** | **78.00%** | **0.27M ACs** | **2.68 mJ** |

## Repository Structure

```
SPARQS/
├── AlexNet/                 # AlexNet implementations (CIFAR-10)
│   ├── models/              # Model architectures
│   │   ├── alexnet.py           # Standard ANN
│   │   ├── alexnet_snn.py       # SNN with LIF neurons
│   │   ├── quantized_snn_alexnet.py  # Quantized SNN (INT8)
│   │   ├── branchy_models.py    # Early-exit architecture
│   │   ├── branchy_quantized_snn.py  # Full QDSNN
│   │   └── rl_environment.py    # Q-learning agent
│   ├── train.py             # Training scripts
│   ├── evaluate.py          # Evaluation scripts
│   └── visualize.py         # Visualization tools
│
├── LeNet-Clean/             # LeNet implementations (MNIST)
│   ├── models/              # LeNet model variants
│   ├── train.py
│   └── evaluate.py
│
├── 5-layer-MLP/             # MLP implementations (MNIST)
│   ├── models/              # MLP model variants
│   ├── train.py
│   └── evaluate.py
│
├── lava_loihi_sim/          # Neuromorphic deployment (Lava / Loihi-2)
│   ├── sparq/               # Weight export + fixed-point network builders
│   ├── src_utils/           # BN folding, Lava SNN helpers, power estimation
│   └── scripts/             # Simulation drivers and smoke tests
│
├── common_utils.py          # Shared utilities
├── requirements.txt         # Python dependencies
└── README.md
```

> Datasets, trained checkpoints, experiment logs and generated figures are not
> tracked in this repository; they are produced by the training and evaluation
> scripts described below.

## Installation

```bash
# Clone the repository
git clone https://github.com/bakasenya/SPARQS.git
cd SPARQS

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.13.0
- SNNTorch >= 0.6.0
- NumPy, Pandas, Matplotlib, Seaborn
- tqdm, scikit-learn

## Quick Start

### 1. Train Standard ANN Baseline

```bash
cd AlexNet/
python train.py --model standard --epochs 50
```

### 2. Train SNN with Surrogate Gradients

```bash
python train.py --model snn --epochs 30 --time-steps 4
```

### 3. Train Quantized SNN (QAT)

```bash
python train_quantized_snn_alexnet.py --epochs 25
```

### 4. Train QDSNN with RL-guided Early Exits

```bash
python train.py --model branchy_qsnn --epochs 30
```

### 5. Evaluate All Models

```bash
python evaluate.py
```

### 6. Neuromorphic Deployment (Lava / Loihi-2)

```bash
# Export a trained model to fixed-point weights
python lava_loihi_sim/sparq/export_lenet_qdsnn.py

# Run it on the Lava simulator and report ACs / power
python lava_loihi_sim/sparq/lava_build_qdsnn.py --npz <exported>.npz --n 50
```

## Methodology

SPARQS implements a four-stage pipeline:

### Stage 1: ANN-to-SNN Conversion
- Start with pre-trained FP32 ANN
- Convert to SNN using LIF neuron model
- Encode inputs as spike trains

### Stage 2: Dynamic Early Exits
- Insert lightweight exit branches at intermediate layers
- Train Q-learning agent for exit policy
- Reward function balances accuracy vs. computational cost

### Stage 3: Quantization-Aware Training
- Convert weights/activations to INT8
- Use FBGEMM backend for x86 inference
- MinMax observers for per-layer scaling

### Stage 4: RL Policy Optimization
- State: exit location + prediction confidence
- Actions: exit now or continue
- Q-learning with ε-greedy exploration

## Model Variants

| Variant | Description |
|---------|-------------|
| `AlexNet` | Standard ANN baseline |
| `AlexNetSNN` | SNN with LIF neurons (surrogate gradients) |
| `QuantizedSNNAlexNet` | INT8 quantized SNN |
| `BranchyAlexNet` | ANN with early exits |
| `BranchyQuantizedSNNAlexNet` | Full QDSNN (quantized + dynamic exits) |

## Energy Modeling

Energy consumption is modeled as:

**For ANNs:**
```
E_ANN = N_MAC × E_MAC + M_mem × E_mem
```

**For SNNs:**
```
E_SNN = N_AC × E_AC + M_SNN × E_mem
```

Where:
- E_MAC = 4.6 × 10⁻¹² J (32-bit MAC)
- E_AC = 0.03 × 10⁻¹² J (8-bit AC)
- E_mem = 80 × 10⁻¹² J/byte

## Visualization

Generate analysis plots:

```bash
python visualize.py
```

Outputs:
- Energy vs. Accuracy scatter plots
- Operation count comparisons
- Exit distribution analysis
- Power consumption breakdown

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sparqs2025,
  title={SPARQS: Spiking Early-Exit Quantized Neural Networks for Safety-Critical Edge Deployment},
  author={Patne, Parth and Taheri, Mahdi and Mahani, Ali and Jenihhin, Maksim and Mahani, Reza and Herglotz, Christian},
  booktitle={Proceedings},
  year={2025}
}
```

## Authors

- **Parth Patne** - Brandenburg Technical University, Cottbus, Germany
- **Mahdi Taheri** - Brandenburg Technical University & Tallinn University of Technology
- **Ali Mahani** - Shahid Bahonar University of Kerman & York University
- **Maksim Jenihhin** - Tallinn University of Technology, Estonia
- **Reza Mahani** - Ferdinand-Braun-Institut, Germany
- **Christian Herglotz** - Brandenburg Technical University, Cottbus, Germany

## Acknowledgements

This work was supported by:
- Estonian Research Council grant PUT PRG1467 "CRASHLESS"
- EU Grant Project 101160182 "TAICHIP"
- Deutsche Forschungsgemeinschaft (DFG) - Project-ID "458578717"

## License

This project is released for academic and research purposes.

## References

- [SNNTorch](https://github.com/jeshraghian/snntorch) - Spiking neural network library
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) - Neural network quantization
- Horowitz, M. (2014). Computing's energy problem. IEEE ISSCC.
