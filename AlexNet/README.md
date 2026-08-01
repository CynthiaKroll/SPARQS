# Dynamic AlexNet with Early Exits

This repository contains implementations of various AlexNet architectures with dynamic inference capabilities.

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

#### Standard AlexNet (ANN):
```bash
python evaluate.py --model standard_alexnet --model_path results/standard_alexnet.pth
```

#### Dynamic AlexNet (Early-Exit):
```bash
python evaluate.py --model dynamic_alexnet --model_path results/branchy_alexnet.pth
```

#### Quantized Dynamic AlexNet (if applicable):
```bash
python evaluate.py --model branchy_quantized_snn --model_path results/quant_snn_qat.pth
```

#### SNN AlexNet:
> **Note:** For SNN AlexNet, evaluation is performed using SpikingJelly with the script `snn_cifar10/convert_with_spikingjelly.py`.

```bash
python snn_cifar10/convert_with_spikingjelly.py \
    --ann_checkpoint results/standard_alexnet.pth \
    --time_steps 200 \
    --batch_size 128 \
    --calib_batch_size 128 \
    --mode '99.9%' \
    --momentum 1.0
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

## How to Run: Standard AlexNet, Dynamic AlexNet, and AlexNet SNN

### 1. Standard AlexNet (Vanilla)
Train and evaluate the classic AlexNet on CIFAR-10:
```bash
# Train standard AlexNet
python train.py --model standard_alexnet --epochs 100 --batch_size 128 --lr 0.01 --output_model results/standard_alexnet.pth

# Evaluate standard AlexNet
python evaluate.py --model standard_alexnet --model_path results/standard_alexnet.pth
```

---

### 2. Dynamic AlexNet (Branchy/Early-Exit)
Train and evaluate AlexNet with early exits (dynamic inference):
```bash
# Train dynamic/branchy AlexNet
python train.py --model branchy_alexnet --epochs 100 --batch_size 128 --lr 0.01 --output_model results/branchy_alexnet.pth

# Evaluate dynamic AlexNet (early-exit)
python evaluate.py --model dynamic_alexnet --model_path results/branchy_alexnet.pth
```
- To use RL agent for early exits, add `--agent_path results/q_agent.pth` if you have a trained agent.

---

### 3. AlexNet SNN (Spiking Neural Network)
Convert a trained ANN to SNN and evaluate using SpikingJelly:
```bash
# 1. Train standard AlexNet (or QCFS-ANN)
python train.py --model standard_alexnet --epochs 100 --batch_size 128 --lr 0.01 --output_model results/standard_alexnet.pth

# 2. Convert and evaluate SNN using SpikingJelly
python snn_cifar10/convert_with_spikingjelly.py \
    --ann_checkpoint results/standard_alexnet.pth \
    --time_steps 200 \
    --batch_size 128 \
    --calib_batch_size 128 \
    --mode '99.9%' \
    --momentum 1.0
```
> This script will convert the trained ANN to an SNN using SpikingJelly, calibrate thresholds, and evaluate the SNN. Results (including fair-comparison metrics) are saved to `results/logs/snn_standard_alexnet_eval.csv`.

---

### 4. General Notes
- All scripts have `--help` for more options (e.g., `python train.py --help`).
- Output models and logs are saved in the `results/` directory.
- For quantized/dynamic SNNs, see the README’s full pipeline or ask for specific instructions.

---
Run end-to-end from ANN training through SNN conversion and quantization:
```bash
# 1. Train QCFS-ANN (AlexNet with QCFS activations)
python train.py --model qcfs_ann --output results/qcfs_ann.pth --lr ${config["qat_learning_rate"]} --num_epochs ${config["qat_num_epochs"]}

# 2. Calibrate thresholds for SNN from QCFS lambdas
python calibrate_thresholds.py \
    --qcfs_model results/qcfs_ann.pth \
    --output results/snn_thresholds.pth

# 3. Convert QCFS-ANN to Quantized Dynamic SNN
python convert_snn.py \
    --ann results/qcfs_ann.pth \
    --output results/quant_snn.pth \
    --time_steps ${config["num_steps"]} \
    --beta ${config["beta"]}

# 4. Prepare and fine-tune with QAT
python quantize_qat.py \
    --snn_checkpoint results/quant_snn.pth \
    --output_path results/quant_snn_qat.pth \
    --num_epochs ${config["qat_num_epochs"]} \
    --lr ${config["qat_learning_rate"]}

# 5. Run unit test for forward pass and exit logic
python test_quantized_dynamic_snn.py
