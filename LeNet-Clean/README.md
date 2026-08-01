# Dynamic LeNet with Early Exits

This repository contains implementations of various LeNet architectures with dynamic inference capabilities for the MNIST dataset.

## Key Features

- Dynamic Inference with early exits
- SNN Implementation for energy efficiency
- Quantization support
- Accurate energy modeling

## Usage

See the documentation for details on training, evaluation, and visualization.

## End-to-End Evaluation and Visualization Pipeline

### 1. Evaluate All Models and Collect Results
Run the following commands to evaluate each model. This will generate CSV logs with all required metrics for plotting:

#### Standard LeNet (ANN):
```bash
python evaluate.py --model standard_lenet --model_path results/standard_lenet.pth
```

#### Dynamic LeNet (Early-Exit):
```bash
python evaluate.py --model dynamic_lenet --model_path results/branchy_lenet.pth
```

#### Quantized Dynamic SNN (if applicable):
```bash
python evaluate.py --model branchy_quantized_snn --model_path results/quant_snn_qat.pth
```

All results will be saved in `results/logs/` as individual CSV files for each model.

### 2. Generate Visualizations
After all evaluations are complete and CSVs are present in `results/logs/`, generate the plots:
```bash
python visualize.py
```

This will create operation count and inference time plots using the collected data.

### 3. Troubleshooting
- If a bar is missing in the plot for a model, check its CSV in `results/logs/` and ensure all relevant fields (e.g., `macs` for ANN, `ac_ops` for SNN) are present and nonzero.
- Re-run the evaluation script for the affected model if any metric is missing.
- For best results, ensure all evaluations are run on the same hardware (ideally GPU).

---

## How to Run: Standard LeNet, Dynamic LeNet, and LeNet SNN

### 1. Standard LeNet (Vanilla)
Train and evaluate the classic LeNet on MNIST:
```bash
# Train standard LeNet
python train.py --model standard_lenet --epochs 30 --batch_size 128 --lr 0.001 --output_model results/standard_lenet.pth

# Evaluate standard LeNet
python evaluate.py --model standard_lenet --model_path results/standard_lenet.pth
```

---

### 2. Dynamic LeNet (Branchy/Early-Exit)
Train and evaluate LeNet with early exits (dynamic inference):
```bash
# Train dynamic/branchy LeNet
python train.py --model branchy_lenet --epochs 30 --batch_size 128 --lr 0.001 --output_model results/branchy_lenet.pth --agent_path results/q_agent.pth

# Evaluate dynamic LeNet (early-exit)
python evaluate.py --model dynamic_lenet --model_path results/branchy_lenet.pth --agent_path results/q_agent.pth
```
- The `--agent_path` parameter is used to save/load the RL agent that learns optimal confidence thresholds for early exits.

---

### 3. Branchy Quantized SNN
Train and evaluate a branchy quantized SNN:
```bash
# Train branchy quantized SNN
python train.py --model branchy_quantized_snn --epochs 30 --batch_size 128 --lr 0.0005 --output_model results/branchy_quantized_snn.pth --time_window 25 --beta 0.9 --qat --agent_path results/q_agent.pth

# Evaluate branchy quantized SNN
python evaluate.py --model branchy_quantized_snn --model_path results/branchy_quantized_snn_int8.pth --agent_path results/q_agent.pth
```
- The `--qat` flag enables quantization-aware training
- The `--time_window` parameter sets the number of time steps for the SNN
- The `--beta` parameter controls the decay factor for LIF neurons

---

### 4. General Notes
- All scripts have `--help` for more options (e.g., `python train.py --help`).
- Output models and logs are saved in the `results/` directory.
- For quantized/dynamic SNNs, see the README's full pipeline or ask for specific instructions.

---

## Project Structure

- `models/`: Contains all model implementations
  - `lenet.py`: Standard LeNet model
  - `lenet_snn.py`: Spiking LeNet model
  - `branchy_models.py`: Early-exit LeNet model
  - `branchy_quantized_snn.py`: Quantized SNN with early exits
  - `quantized_snn_lenet.py`: Quantized SNN LeNet
  - `rl_environment.py`: Q-learning agent for dynamic inference
- `data/`: MNIST dataset storage
- `results/`: Stores model checkpoints, logs, and visualization outputs
- `train.py`: Training script for all models
- `evaluate.py`: Evaluation script for all models
- `visualize.py`: Visualization script for generating plots
- `utils.py`: Utility functions for energy calculation, operation counting, etc.
- `config.py`: Configuration settings
- `data_loader.py`: Data loading utilities for MNIST# Dynamic LeNet with Early Exits

This repository contains implementations of various LeNet architectures with dynamic inference capabilities for the MNIST dataset.

## Key Features

- Dynamic Inference with early exits
- SNN Implementation for energy efficiency
- Quantization support
- Accurate energy modeling

## Usage

See the documentation for details on training, evaluation, and visualization.

## End-to-End Evaluation and Visualization Pipeline

### 1. Evaluate All Models and Collect Results
Run the following commands to evaluate each model. This will generate CSV logs with all required metrics for plotting:

#### Standard LeNet (ANN):
```bash
python evaluate.py --model standard_lenet --model_path results/standard_lenet.pth
```

#### Dynamic LeNet (Early-Exit):
```bash
python evaluate.py --model dynamic_lenet --model_path results/branchy_lenet.pth
```

#### Quantized Dynamic SNN (if applicable):
```bash
python evaluate.py --model branchy_quantized_snn --model_path results/quant_snn_qat.pth
```

All results will be saved in `results/logs/` as individual CSV files for each model.

### 2. Generate Visualizations
After all evaluations are complete and CSVs are present in `results/logs/`, generate the plots:
```bash
python visualize.py
```

This will create operation count and inference time plots using the collected data.

### 3. Troubleshooting
- If a bar is missing in the plot for a model, check its CSV in `results/logs/` and ensure all relevant fields (e.g., `macs` for ANN, `ac_ops` for SNN) are present and nonzero.
- Re-run the evaluation script for the affected model if any metric is missing.
- For best results, ensure all evaluations are run on the same hardware (ideally GPU).

---

## How to Run: Standard LeNet, Dynamic LeNet, and LeNet SNN

### 1. Standard LeNet (Vanilla)
Train and evaluate the classic LeNet on MNIST:
```bash
# Train standard LeNet
python train.py --model standard_lenet --epochs 30 --batch_size 128 --lr 0.001 --output_model results/standard_lenet.pth

# Evaluate standard LeNet
python evaluate.py --model standard_lenet --model_path results/standard_lenet.pth
```

---

### 2. Dynamic LeNet (Branchy/Early-Exit)
Train and evaluate LeNet with early exits (dynamic inference):
```bash
# Train dynamic/branchy LeNet
python train.py --model branchy_lenet --epochs 30 --batch_size 128 --lr 0.001 --output_model results/branchy_lenet.pth

# Evaluate dynamic LeNet (early-exit)
python evaluate.py --model dynamic_lenet --model_path results/branchy_lenet.pth
```
- To use RL agent for early exits, add `--agent_path results/q_agent.pth` if you have a trained agent.

---

### 3. Branchy Quantized SNN
Train and evaluate a branchy quantized SNN:
```bash
# Train branchy quantized SNN
python train.py --model branchy_quantized_snn --epochs 30 --batch_size 128 --lr 0.0005 --output_model results/branchy_quantized_snn.pth --time_window 25 --beta 0.9 --qat

# Evaluate branchy quantized SNN
python evaluate.py --model branchy_quantized_snn --model_path results/branchy_quantized_snn_int8.pth
```

---

### 4. General Notes
- All scripts have `--help` for more options (e.g., `python train.py --help`).
- Output models and logs are saved in the `results/` directory.
- For quantized/dynamic SNNs, see the README's full pipeline or ask for specific instructions.

---

## Project Structure

- `models/`: Contains all model implementations
  - `lenet.py`: Standard LeNet model
  - `lenet_snn.py`: Spiking LeNet model
  - `branchy_models.py`: Early-exit LeNet model
  - `branchy_quantized_snn.py`: Quantized SNN with early exits
  - `quantized_snn_lenet.py`: Quantized SNN LeNet
  - `rl_environment.py`: Q-learning agent for dynamic inference
- `data/`: MNIST dataset storage
- `results/`: Stores model checkpoints, logs, and visualization outputs
- `train.py`: Training script for all models
- `evaluate.py`: Evaluation script for all models
- `visualize.py`: Visualization script for generating plots
- `utils.py`: Utility functions for energy calculation, operation counting, etc.
- `config.py`: Configuration settings
- `data_loader.py`: Data loading utilities for MNIST