#!/bin/bash

# Train All LeNet Models
# This script trains all available models in sequence

echo "Starting training of all LeNet models..."

# Create results directory if it doesn't exist
mkdir -p results

# Training parameters
# 20 matches config.py; the paper does not specify an epoch count. (Prefer
# run_training.sh, which trains the quantized pair with QAT + RL.)
EPOCHS=20
BATCH_SIZE=128
LR=0.001

# 1. Train Standard LeNet
echo "=========================================="
echo "Training Standard LeNet"
echo "=========================================="
python train.py --model standard_lenet \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_model results/standard_lenet.pth

if [ $? -eq 0 ]; then
    echo "✓ Standard LeNet training completed successfully"
else
    echo "✗ Standard LeNet training failed"
fi

# 2. Train Branchy LeNet
echo "=========================================="
echo "Training Branchy LeNet"
echo "=========================================="
python train.py --model branchy_lenet \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_model results/branchy_lenet.pth \
    --agent_path results/rl_agent_branchy.pkl

if [ $? -eq 0 ]; then
    echo "✓ Branchy LeNet training completed successfully"
else
    echo "✗ Branchy LeNet training failed"
fi

# 3. Train Branchy Quantized SNN (without QAT)
echo "=========================================="
echo "Training Branchy Quantized SNN (FP32)"
echo "=========================================="
python train.py --model branchy_quantized_snn \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_model results/branchy_quantized_snn_fp32.pth \
    --agent_path results/rl_agent_qsnn.pkl \
    --time_window 25 \
    --beta 0.9

if [ $? -eq 0 ]; then
    echo "✓ Branchy Quantized SNN (FP32) training completed successfully"
else
    echo "✗ Branchy Quantized SNN (FP32) training failed"
fi

# 4. Train Branchy Quantized SNN with QAT
echo "=========================================="
echo "Training Branchy Quantized SNN with QAT"
echo "=========================================="
python train.py --model branchy_quantized_snn \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_model results/branchy_quantized_snn_qat.pth \
    --agent_path results/rl_agent_qat.pkl \
    --time_window 25 \
    --beta 0.9 \
    --qat

if [ $? -eq 0 ]; then
    echo "✓ Branchy Quantized SNN with QAT training completed successfully"
else
    echo "✗ Branchy Quantized SNN with QAT training failed"
fi

echo "=========================================="
echo "Training Summary"
echo "=========================================="
echo "Models trained:"
ls -la results/*.pth 2>/dev/null || echo "No model files found"
echo ""
echo "RL agents saved:"
ls -la results/*.pkl 2>/dev/null || echo "No RL agent files found"
echo ""
echo "All training completed!" 