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

The resulting **Quantized Dynamic SNN (QDSNN)** achieves up to **3.5x lower energy** and
**1.8x lower latency** than a conventional QSNN while maintaining competitive accuracy, and
the framework additionally evaluates the reliability and security of each design point.

## Results

Accuracy is measured off-chip in every row. Spiking variants are evaluated with a
bit-accurate fixed-point model of Intel Loihi-2; non-spiking variants use the analytical
MAC/memory model with a roofline latency estimate on an A100-class accelerator. Loihi
figures are simulation-based estimates, not hardware measurements.

| Architecture | Variant | Acc. (%) | Ops (M) | Delay (ms) | Power (W) | Energy (mJ) |
|---|---|---|---|---|---|---|
| **5-Layer MLP** (MNIST) | ANN | 99.03 | 32.22 | 0.083 | 126.2 | 10.474 |
| | Dynamic ANN | 99.18 | 14.94 | 0.039 | 126.2 | 4.858 |
| | SNN (T=32) | 98.65 | 26.76 | 0.440 | 0.535 | 0.235 |
| | QSNN (T=4) | 98.53 | 31.54 | 0.160 | 1.625 | 0.260 |
| | **QDSNN (RL, T=4)** | 97.94 | **9.08** | **0.089** | 0.841 | **0.075** |
| **LeNet-5** (MNIST) | ANN | 99.57 | 0.417 | 0.00018 | 134.9 | 0.0246 |
| | Dynamic ANN | 99.46 | 0.257 | 0.00008 | 139.7 | 0.0108 |
| | SNN (T=32) | 98.36 | 22.44 | 0.410 | 0.457 | 0.188 |
| | QSNN (T=16) | 99.49 | 3.52 | 0.250 | 0.132 | 0.033 |
| | **QDSNN (RL, T=16)** | 97.63 | **2.82** | **0.179** | 0.145 | **0.026** |
| **AlexNet** (CIFAR-10) | ANN | 89.96 | 200.52 | 0.093 | 134.3 | 12.481 |
| | Dynamic ANN | 90.36 | 129.87 | 0.011 | 177.3 | 2.003 |
| | SNN (T=32) | 67.21 | – | – | – | – |
| | QSNN (T=32) | 70.45 | 578 | 0.430 | 11.30 | 4.860 |
| | QDSNN (Cfg 0.5/0.5) | 70.84 | 588 | 0.420 | 11.72 | 4.925 |
| | **QDSNN (Cfg 0.2/0.4)** | 60.16 | **492** | **0.399** | 10.32 | **4.119** |

### QDSNN vs. QSNN

- **MLP** — 71% fewer operations (31.54 → 9.08 M), **3.5x lower energy** (0.260 → 0.075 mJ)
  and **1.8x lower latency** (0.160 → 0.089 ms), at a cost of 0.6 pp accuracy.
- **LeNet-5** — 20% fewer operations, 21% lower energy and 29% lower latency, at a cost of
  1.9 pp accuracy.
- **AlexNet** — depends on the exit configuration. At Cfg 0.5/0.5 accuracy improves by
  0.4 pp at essentially unchanged energy, because ~75% of samples still traverse the full
  network. The aggressive Cfg 0.2/0.4 cuts energy by 15% and latency by 7%, but at a
  substantial accuracy cost.

### Spiking vs. non-spiking

The advantage of the spiking variants is **power**, not latency: 12x lower for AlexNet, 78x
for the MLP and 1022x for LeNet. Energy savings scale with eliminated weight traffic, so
they depend on the model — the MLP QDSNN uses ~140x less energy than its ANN
(0.075 vs. 10.474 mJ) and AlexNet 2.5–3.0x less, whereas for the ~62 k-parameter LeNet the
spiking variants actually consume **more** energy than the ANN (0.026–0.188 vs. 0.0246 mJ).

### Resilience

Per-axis degradation at severity level L3, with the Spiking Resilience Score (SRS, higher is
better) and the figure of merit Φ normalised to each architecture's QSNN. *R* is reliability
degradation under weight-memory bit flips, *S* is security degradation under bounded input
perturbation (both lower is better).

| Architecture | Variant | R ↓ | S ↓ | SRS ↑ | Acc. (%) | Φ ↑ |
|---|---|---|---|---|---|---|
| **LeNet-5** | ANN (FP32) | 37.21 | 5.51 | 77.02 | 99.80 | – |
| | QSNN (INT8) | 1.80 | 11.53 | **93.21** | 99.80 | 1.000 |
| | QDSNN (INT8) | 9.44 | 20.45 | 84.88 | 98.24 | **1.043** |
| **5-Layer MLP** | ANN (FP32) | 46.29 | 1.44 | 72.76 | 99.61 | – |
| | QSNN (INT8) | 4.07 | 2.51 | **96.71** | 99.22 | 1.000 |
| | QDSNN (INT8) | 14.39 | 4.60 | 90.38 | 99.50 | **1.278** |
| **AlexNet** | ANN (FP32) | 45.70 | 22.24 | 64.98 | 92.19 | – |
| | QSNN (INT8) | 1.73 | 8.48 | **94.83** | 73.44 | 1.000 |
| | QDSNN (INT8) | 2.74 | 10.11 | 93.50 | 77.34 | **1.119** |

Quantization is what buys fault tolerance — it removes the floating-point failure modes that
dominate the FP32 ANN's reliability degradation. Dynamic inference then cuts the effective
fault cross-section a further 5–11x versus the FP32 ANN by reducing exposed state, but it
costs security robustness, so the QSNN holds the highest SRS at mid severities. The ranking
flips at the extremes, which is the point: the efficiency-optimal exit policy is **not**
consistently the most resilient operating point.

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
├── resilience/              # Reliability and security evaluation
│   ├── fault_inject.py      # Weight-memory bit-flip injection
│   ├── attack.py            # Shared rho-budget input perturbation
│   ├── campaign.py          # Campaign driver across architectures/variants
│   ├── registry.py          # Loader + metadata for every (arch, variant) pair
│   ├── metrics.py           # The SPARQ resilience metric
│   ├── cost.py              # Threat-aware energy / latency accounting
│   ├── margin_drift.py      # Top-1/Top-2 margin drift under faults
│   ├── sensitivity.py       # Cost-model sensitivity of the joint metric
│   └── summarize.py         # Joint metric + result tables
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
python train.py --model snn --epochs 30 --time-steps 32
```

The timestep budget is architecture-specific: T=4 for the MLP, T=16 for LeNet-5 and T=32
for AlexNet. LeNet additionally needs pipeline-delay-aware training, since its small
synaptic fan-in (25 and 150 inputs per neuron) makes it fail at shorter budgets.

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

### 7. Resilience Evaluation (Faults and Attacks)

```bash
# Fault-injection + attack campaign for one (architecture, variant) pair
python resilience/campaign.py --arch alexnet --variant qdsnn --n 512 --trials 3

# Combine the per-variant CSVs into the joint metric and result tables
python resilience/summarize.py
```

## Methodology

SPARQS implements a five-stage pipeline:

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
- Learned thresholds: θ = (0.7, 0.2) for the MLP, (0.6, 0.6) for LeNet and (0.7, 0.4) for
  AlexNet. The AlexNet policy is deliberately conservative — it still routes 76.6% of
  samples through the full network, because on 10-class CIFAR-10 the accumulated confidence
  values cover only a narrow region of the discretised state space.

### Stage 5: Resilience Evaluation
- Weight-memory bit flips for reliability, bounded input perturbation for security
- Spiking Resilience Score aggregates accuracy, routing, energy and latency degradation
- Exit thresholds are inference-time parameters, so a single trained QDSNN spans a whole
  design space that can be swept without retraining

## Model Variants

| Variant | Description |
|---------|-------------|
| `AlexNet` | Standard ANN baseline |
| `AlexNetSNN` | SNN with LIF neurons (surrogate gradients) |
| `QuantizedSNNAlexNet` | INT8 quantized SNN |
| `BranchyAlexNet` | ANN with early exits |
| `BranchyQuantizedSNNAlexNet` | Full QDSNN (quantized + dynamic exits) |

## Cost Modeling

Two cost models are used, and the results table states which applies to each row.

**Analytical model** (all variants, and the only model for non-spiking networks):

```
E_ANN = N_MAC × E_MAC + M_mem × E_mem
E_SNN = N_AC  × E_AC  + M_SNN  × E_mem
```

- `E_MAC` = 4.6 pJ (32-bit MAC), `E_AC` = 0.03 pJ (8-bit accumulate), `E_mem` = 80 pJ/byte
- `N_AC` counts emitted spike events, so it reflects measured sparsity rather than an
  assumed firing rate. The LIF state update `N_LIF = N_neurons × T × 2` is independent of
  sparsity and is reported separately.
- ANN latency uses a two-sided roofline, `t = max(2·N_MAC/Θ, M_mem/BW)`; every evaluated
  configuration is bandwidth-bound.

**Neuromorphic model** (spiking variants, bit-accurate fixed-point Intel Loihi-2):

```
E_Loihi = N_AC_hw × E_synop + N_neurons × t_run × E_neuron
```

- `E_synop` = 8 pJ, `E_neuron` = 30 pJ — roughly one third of the published Loihi-1 values
  (23.6 pJ / 81 pJ), consistent with the 14 nm → Intel 4 transition. Substituting the
  Loihi-1 constants scales all values by ~3x without changing any ranking.
- There is no separate memory term: weight fetch is inherent to a synaptic operation.
- `N_AC_hw` weights each spike by its layer's mean non-zero fan-out, since one emitted spike
  drives one accumulate per outgoing synapse.
- Dendritic accumulation adds one timestep of delay per layer, so a network runs for
  `t_run = T + L + δ` ticks at `t_tick` = 10 µs.
- Hardware cannot branch at run time, so the QDSNN is costed by path decomposition:
  `E_QDSNN = Σ_i P(exit=i) · E_Loihi(π_i, T_i)`. Reported values are therefore
  exit-weighted averages, not the cost of one end-to-end pass.

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

## Authors

- **Harshvardhan Sharma** - Brandenburg Technical University, Cottbus, Germany
- **Mahdi Taheri** - Brandenburg Technical University & Tallinn University of Technology

## Acknowledgements

This work was supported by:
- Estonian Research Council grant PUT PRG1467 "CRASHLESS"
- EU Grant Project 101160182 "TAICHIP"
- Deutsche Forschungsgemeinschaft (DFG) - Project-ID "458578717"

## License

This project is released for academic and research purposes.

## Citation
If you use SPARQS in your research, experiments, publications, or derivative work, please cite our paper.
SPARQS: SPiking EARly-Exit Quantized Neural Networks for Safety-Critical Edge Deployment
Mahdi Taheri, Harshvardhan Sharma, Christian Herglotz, and Maksim Jenihhin
Journal of Systems Architecture (JSA), Elsevier
Special Issue on the Architecture of Computing Systems (ARCS 2026)

Bibtex:
@article{taheri2026sparqs,
  title   = {SPARQS: SPiking EARly-Exit Quantized Neural Networks for Safety-Critical Edge Deployment},
  author  = {Taheri, Mahdi and Sharma, Harshvardhan and Herglotz, Christian and Jenihhin, Maksim},
  journal = {Journal of Systems Architecture},
  note    = {Special Issue on the Architecture of Computing Systems (ARCS 2026)},
  year    = {2026}
}

- [SNNTorch](https://github.com/jeshraghian/snntorch) - Spiking neural network library
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) - Neural network quantization
- Horowitz, M. (2014). Computing's energy problem. IEEE ISSCC.
