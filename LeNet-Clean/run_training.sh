#!/bin/bash
# Train the SPARQ LeNet models (MNIST).
#
# Time windows follow the 5-layer-MLP scheme (its SNN_T=32 / QNT_T=4 split):
#   - the FP32 SNN baseline (snn) trains at SNN_T=32, the window a plain rate
#     coder needs for good accuracy;
#   - the QAT pair (QSNN + QDSNN) trains at QNT_T=4, which is what makes their
#     energy competitive (energy ~ linear in T). Both use real INT8 QAT (--qat);
#     the QDSNN additionally learns its early-exit policy with the RL agent
#     (--use_rl). Evaluation must use the SAME per-model T (quick_eval_all.py
#     and evaluate_quantized_pair.py already default to these config values).
#
# Usage:
#   ./run_training.sh              # train everything
#   ./run_training.sh qsnn         # train a single target
#   EPOCHS=1 ./run_training.sh     # quick smoke run
#   SNN_T=16 QNT_T=8 ./run_training.sh   # override the time windows
#
# Default epoch count is config["epochs"] (20). The SPARQ paper does not state an
# epoch count anywhere, so 20 (from config.py) is the documented number; the old
# scripts' "30" was arbitrary. Override with EPOCHS=... if you want more.
set -e

EPOCHS="${EPOCHS:-20}"
BS="${BS:-128}"
SNN_T="${SNN_T:-32}"   # FP32 SNN window (config.STANDARD_SNN_STEPS)
QNT_T="${QNT_T:-4}"    # QAT pair window (config.QUANTIZED_SNN_STEPS)
TARGET="${1:-all}"

mkdir -p results/logs results/plots

run() { echo "=== $1 ==="; shift; python "$@"; }

if [ "$TARGET" = "all" ] || [ "$TARGET" = "lenet" ]; then
  run "Standard LeNet (FP32 ANN)" train.py --model standard_lenet \
      --epochs "$EPOCHS" --batch_size "$BS" --lr 0.001 \
      --output_model results/standard_lenet.pth
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "branchy_lenet" ]; then
  run "Branchy LeNet (FP32 early-exit ANN)" train.py --model branchy_lenet \
      --epochs "$EPOCHS" --batch_size "$BS" --lr 0.001 \
      --output_model results/branchy_lenet.pth
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "snn" ]; then
  run "SNN LeNet (FP32 SNN, T=$SNN_T)" train.py --model lenet_snn \
      --epochs "$EPOCHS" --batch_size "$BS" --lr 0.001 --time_window "$SNN_T" \
      --output_model results/lenet_snn.pth
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "qsnn" ]; then
  run "Quantized SNN LeNet (QSNN, INT8 QAT, T=$QNT_T)" train.py --model quantized_snn_lenet \
      --qat --epochs "$EPOCHS" --batch_size "$BS" --lr 0.001 --time_window "$QNT_T" \
      --output_model results/quantized_snn_lenet_int8.pth
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "qdsnn" ]; then
  run "Branchy Quantized SNN (QDSNN, INT8 QAT + RL early exit, T=$QNT_T)" train.py \
      --model branchy_quantized_snn --qat --use_rl \
      --epochs "$EPOCHS" --batch_size "$BS" --lr 0.001 --time_window "$QNT_T" \
      --output_model results/branchy_quantized_snn_int8.pth \
      --agent_output results/rl_agent_qdsnn.pkl
fi

echo "Training complete. Models in results/."
