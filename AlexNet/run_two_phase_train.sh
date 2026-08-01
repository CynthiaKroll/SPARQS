#!/usr/bin/env bash
# Two-phase FP32 pretrain then QAT finetune for BranchyQuantizedSNN

# Phase 1: Extended FP32 Pre-training with RL Curriculum
python3 train.py \
  --model branchy_quantized_snn \
  --use_rl \
  --agent_output results/rl_agent_fp32_200e.pkl \
  --output_model results/branchy_quantized_snn_fp32_rl_200e.pth \
  --epochs 100 \
  --lr 1e-3 \
  --lr_scheduler cosine \
  --mixup_alpha 0.2

# Phase 2: QAT with Progressive Quantization (Optimized Hyperparameters)
python3 train.py \
  --model branchy_quantized_snn \
  --qat \
  --use_rl \
  --agent_output results/rl_agent_qat_200e.pkl \
  --output_model results/branchy_quantized_snn_int8_rl_200e.pth \
  --epochs 100 \
  --lr 3e-5 \
  --lr_scheduler cosine \
  --quant_steps 5 \
  --teacher_model_path "results/standard_alexnet.pth" \
  --kd_alpha 0.9 \
  --kd_temperature 4.0

