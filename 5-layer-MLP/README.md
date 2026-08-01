# 5-Layer MLP Neural Network Analysis

This project provides a comprehensive analysis of 5-layer Multi-Layer Perceptron (MLP) neural networks on the MNIST dataset, including standard ANNs, Spiking Neural Networks (SNNs), branchy architectures, and quantized models with detailed energy analysis.

## Features

- **Multiple Model Architectures**:
  - Standard 5-layer MLP
  - SNN MLP with temporal dynamics
  - Branchy MLP with early exit capabilities
  - Quantized SNN MLP for efficiency

- **Comprehensive Energy Analysis**:
  - MAC (Multiply-Accumulate) operations counting
  - AC (Accumulate) operations for SNNs
  - Memory access analysis
  - Theoretical energy calculations
  - Hardware-specific energy profiling

- **Dynamic Early Exit**:
  - Confidence-based early termination
  - Reinforcement learning for adaptive thresholds
  - Energy-accuracy trade-off optimization

## Project Structure

```
5-Layer-MLP/
├── models/
│   ├── mlp.py                          # Standard MLP implementation
│   ├── mlp_snn.py                     # SNN MLP implementation
│   ├── branchy_mlp.py                 # Branchy MLP with early exits
│   ├── quantized_snn_mlp.py           # Quantized SNN MLP
│   ├── branchy_quantized_snn_mlp.py   # Branchy quantized SNN MLP
│   └── rl_environment.py              # RL agent for dynamic inference
├── data/                              # MNIST dataset
├── results/
│   ├── logs/                          # Evaluation results (CSV)
│   └── plots/                         # Visualization plots
├── config.py                         # Configuration settings
├── data_loader.py                     # MNIST data loading utilities
├── train.py                          # Training script
├── evaluate.py                       # Evaluation script
├── utils.py                          # Utility functions
├── visualize.py                      # Visualization and plotting
├── quick_eval_all.py                 # Quick evaluation script
├── run_training.sh                   # Training shell script
├── run_evaluation.sh                 # Evaluation shell script
└── requirements.txt                  # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 5-Layer-MLP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download MNIST dataset (automatic on first run):
```bash
python data_loader.py
```

## Usage

### Training Models

Train all models:
```bash
./run_training.sh
```

Train individual models:
```bash
# Standard MLP
python train.py --model mlp --epochs 20 --lr 0.001

# SNN MLP
python train.py --model snn_mlp --epochs 20 --lr 0.001 --time_steps 32

# Branchy MLP
python train.py --model branchy_mlp --epochs 20 --lr 0.001

# Quantized SNN MLP
python train.py --model quantized_snn_mlp --epochs 20 --lr 0.001 --time_steps 32
```

### Evaluation

Evaluate all models and generate plots:
```bash
./run_evaluation.sh
```

Quick evaluation:
```bash
python quick_eval_all.py
```

Individual model evaluation:
```bash
python evaluate.py --model mlp --model_path results/standard_mlp.pth
```

### Energy Analysis

The evaluation automatically performs comprehensive energy analysis including:

- **MAC Operations**: For standard ANN models
- **AC Operations**: For SNN models (memory-based computations)  
- **Memory Access**: Data movement energy costs
- **Theoretical Energy**: Hardware-specific energy calculations
- **Neuromorphic Projections**: Estimated energy on neuromorphic hardware

## Model Architectures

### Standard MLP
- 5 fully connected layers: 784 → 4096 → 2048 → 1024 → 512 → 10
- ReLU activations, Batch Normalization, Dropout
- Traditional backpropagation training

### SNN MLP  
- Same architecture with Leaky Integrate-and-Fire (LIF) neurons
- Temporal spike-based processing
- Surrogate gradient training
- Configurable time windows

### Branchy MLP
- 2 early exit points after layers 1 and 3
- Confidence-based early termination
- Joint loss optimization across all exits
- Reduced inference computational cost

### Quantized SNN MLP
- 8-bit quantization-aware training
- Maintains SNN temporal dynamics
- Optimized for hardware deployment

## Energy Calculations

### Mathematical Formulations

For standard ANNs:
```
E_ANN = (MAC_ops × E_MAC) + (mem_access × E_mem)
```

For SNNs:
```
E_SNN = (AC_ops × E_AC) + (mem_access × E_mem)
```

Where:
- `E_MAC` = 4.6 pJ (32-bit floating-point MAC)
- `E_AC` = 0.03 pJ (8-bit integer accumulate)
- `E_mem` = 4 pJ/Byte (DDR3 memory access)

### Hardware Profiles

- **A100**: 312 TFLOPS, optimized energy constants
- **V100**: 125 TFLOPS, standard GPU profile  
- **RTX3090**: 35.6 TFLOPS, consumer GPU
- **Neuromorphic**: Ultra-low power specialized hardware

## Results

The evaluation generates:

1. **CSV Files** (`results/logs/`):
   - Accuracy metrics
   - Inference time measurements
   - Operation counts (MAC/AC)
   - Memory access statistics
   - Energy consumption estimates

2. **Visualization Plots** (`results/plots/`):
   - Accuracy vs Power consumption
   - Energy consumption comparison
   - Inference time analysis
   - Operation count breakdowns
   - Memory access patterns

## Configuration

Modify `config.py` to adjust:
- Training hyperparameters
- SNN temporal parameters
- Hardware energy constants
- Early exit thresholds
- Quantization settings

## Citation

If you use this work, please cite:
```bibtex
@misc{mlp_energy_analysis,
  title={Comprehensive Energy Analysis of 5-Layer MLP Architectures},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details. 