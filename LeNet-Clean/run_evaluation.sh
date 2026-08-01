#!/bin/bash
# Evaluate the SPARQ LeNet models.
#
# The headline comparison (QSNN vs QDSNN on power AND delay) is produced by the
# UNIFIED evaluator, which measures both models on the same code path (same T,
# encoder, spike/AC counter, memory counter, energy constants). The old flow
# evaluated them separately at different time steps, so the gap was a
# measurement artifact.
set -e

mkdir -p results/logs results/plots

echo "=== Unified QSNN vs QDSNN evaluation ==="
python evaluate_quantized_pair.py \
    --qsnn_path results/quantized_snn_lenet_int8.pth \
    --qdsnn_path results/branchy_quantized_snn_int8.pth \
    --agent_path results/rl_agent_qdsnn.pkl \
    --output_csv results/qsnn_vs_qdsnn.csv

# FP32 baselines (optional) via the per-model evaluator.
echo "=== FP32 baselines ==="
python evaluate.py --model lenet --model_path results/standard_lenet.pth || true
python evaluate.py --model dynamic_lenet --model_path results/branchy_lenet.pth || true
python evaluate.py --model snn_lenet --model_path results/lenet_snn.pth || true

echo "Evaluation complete. See results/qsnn_vs_qdsnn.csv and results/logs/."
