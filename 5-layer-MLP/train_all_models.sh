#!/bin/bash

# Complete training script for all 5-Layer MLP models on MNIST
# This script trains all model variants and evaluates them

echo "=========================================="
echo "Training All 5-Layer MLP Models on MNIST"
echo "=========================================="

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directories
mkdir -p results/logs
mkdir -p results/plots

# Training configurations
EPOCHS=20
LEARNING_RATE=0.001
BATCH_SIZE=128
TIME_STEPS=32

echo "Training configuration:"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Time Steps (SNN): $TIME_STEPS"
echo ""

# 1. Standard MLP
echo "1. Training Standard MLP..."
python train.py --model mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --save_path results/standard_mlp.pth
echo "✓ Standard MLP training completed"
echo ""

# 2. SNN MLP (with fixed thresholds=1.0 and proper spike encoding)
echo "2. Training SNN MLP..."
python train.py --model snn_mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --time_steps $TIME_STEPS --save_path results/snn_mlp.pth
echo "✓ SNN MLP training completed"
echo ""

# 3. Branchy MLP (Dynamic MLP)
echo "3. Training Branchy MLP (Dynamic MLP)..."
python train.py --model branchy_mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --save_path results/branchy_mlp.pth
echo "✓ Branchy MLP training completed"
echo ""

# 4. Quantized SNN MLP (QSNN)
echo "4. Training Quantized SNN MLP (QSNN)..."
python train.py --model quantized_snn_mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --time_steps $TIME_STEPS --qat --save_path results/quantized_snn_mlp.pth
echo "✓ Quantized SNN MLP training completed"
echo ""

# 5. Branchy Quantized SNN MLP (QDSNN) - Default thresholds
echo "5. Training Branchy Quantized SNN MLP (QDSNN)..."
python train.py --model branchy_quantized_snn_mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --time_steps $TIME_STEPS --qat --save_path results/branchy_quantized_snn_mlp.pth
echo "✓ Branchy Quantized SNN MLP training completed"
echo ""

# 6. Branchy Quantized SNN MLP with threshold 0.6, 0.7
echo "6. Training Branchy Quantized SNN MLP (QDSNN) with thresholds 0.6, 0.7..."
python train.py --model branchy_quantized_snn_mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --time_steps $TIME_STEPS --qat --thresholds 0.6 0.7 --save_path results/branchy_quantized_snn_mlp_th0.6_0.7.pth
echo "✓ Branchy Quantized SNN MLP (0.6, 0.7) training completed"
echo ""

# 7. Branchy Quantized SNN MLP with threshold 0.2, 0.4
echo "7. Training Branchy Quantized SNN MLP (QDSNN) with thresholds 0.2, 0.4..."
python train.py --model branchy_quantized_snn_mlp --epochs $EPOCHS --lr $LEARNING_RATE --batch_size $BATCH_SIZE --time_steps $TIME_STEPS --qat --thresholds 0.2 0.4 --save_path results/branchy_quantized_snn_mlp_th0.2_0.4.pth
echo "✓ Branchy Quantized SNN MLP (0.2, 0.4) training completed"
echo ""

echo "=========================================="
echo "All 7 MLP models trained successfully!"
echo ""
echo "Model types trained:"
echo "  1. Standard MLP"
echo "  2. SNN MLP"
echo "  3. Dynamic MLP (Branchy MLP)"
echo "  4. QSNN (Quantized SNN MLP)"
echo "  5. QDSNN (default thresholds)"
echo "  6. QDSNN (thresholds 0.6, 0.7)"
echo "  7. QDSNN (thresholds 0.2, 0.4)"
echo ""
echo "Model files saved in results/ directory:"
ls -la results/*.pth 2>/dev/null || echo "  (Model files will appear after training)"
echo ""

echo "Next steps:"
echo "  1. Run evaluation: python evaluate.py --model all"
echo "  2. Generate visualizations: python visualize.py"
echo "  3. Or use the evaluation script: ./run_evaluation.sh"
echo "==========================================" 