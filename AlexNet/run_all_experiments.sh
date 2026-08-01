#!/bin/bash
# SPARQ / AlexNet (CIFAR-10) experiments.
#
# The headline claim is QDSNN vs QSNN on power AND delay, so those two are the
# models that matter. They now share:
#   - the same backbone (5 conv + 3 fc AlexNet, 35,873,866 backbone params),
#   - the same time window T (config["num_steps"] = 32),
#   - the same Poisson input encoding,
#   - the same evaluation script (evaluate_quantized_pair.py).
# The only difference between them is the two early exits, which is what makes
# the comparison a valid attribution of any savings TO early exiting.
#
# (Previously the QSNN was a 35.9M-param AlexNet trained at T=4 with direct
# encoding and scored by evaluate_quantized_snn.py, while the "QDSNN" was a
# separate 6.4M-param 2-conv net trained at T=32 and scored by evaluate.py.
# Also, step 3 below used to call train_fp32_snn_alexnet.py, which does not
# exist in this repo.)
set -e

# Batch size: 32 fits the 8 GB RTX 3070 (T=32 QAT backward does NOT fit at 128).
# On a big-memory GPU (A100 40/80 GB) run:  BS=128 ./run_all_experiments.sh
BS="${BS:-32}"

mkdir -p results

# ---------------------------------------------------------------------------
# 1. QSNN baseline -- quantized SNN AlexNet, no early exits
# ---------------------------------------------------------------------------
echo "==== Training Quantized SNN AlexNet (QSNN baseline) ===="
# batch 32, NOT 128: at T=32 the QAT backward graph spans all 32 timesteps and
# batch 128 does not fit in 8 GB of GPU memory (illegal-memory-access crash +
# ~600 s/it thrashing). Drop to 16 if 32 still OOMs on your card. Never run
# another training on the same GPU concurrently.
python train_quantized_snn_alexnet.py \
  --epochs 20 \
  --batch_size "$BS" \
  --lr 0.001 \
  --output_model results/quantized_snn_alexnet_qat.pth

# ---------------------------------------------------------------------------
# 2. QDSNN -- the SAME backbone plus 2 RL-gated early exits
# ---------------------------------------------------------------------------
echo "==== Training Quantized Dynamic SNN AlexNet (QDSNN) ===="
python train.py \
  --model branchy_quantized_snn \
  --epochs 20 \
  --batch_size "$BS" \
  --lr 0.001 \
  --qat \
  --use_rl \
  --rl_confidence_bins 10 \
  --output_model results/branchy_quantized_snn_qat.pth \
  --agent_output results/rl_agent_qat.pkl

# ---------------------------------------------------------------------------
# 3. The comparison the paper's claim rests on (one script, one measurement path)
# ---------------------------------------------------------------------------
echo "==== Evaluating QSNN vs QDSNN (identical measurement path) ===="
python evaluate_quantized_pair.py \
  --qsnn_path  results/quantized_snn_alexnet_qat.pth \
  --qdsnn_path results/branchy_quantized_snn_qat.pth \
  --agent_path results/rl_agent_qat.pkl \
  --output_csv results/qsnn_vs_qdsnn.csv

# ---------------------------------------------------------------------------
# 4. Optional FP32 ANN reference points (NOT part of the QDSNN-vs-QSNN claim)
# ---------------------------------------------------------------------------
# echo "==== Standard AlexNet (FP32 ANN reference) ===="
# python train.py --model standard_alexnet --epochs 50 --lr 0.001 \
#   --output_model results/standard_alexnet.pth
# python evaluate.py --model standard_alexnet \
#   --model_path results/standard_alexnet.pth --batch_size 128

echo "==== Done. Results table: results/qsnn_vs_qdsnn.csv ===="
