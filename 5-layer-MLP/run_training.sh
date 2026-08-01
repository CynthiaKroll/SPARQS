#!/usr/bin/env bash
#
# Train the 5-layer MLP models on MNIST (SPARQ).
#
# Usage:
#   ./run_training.sh              # train everything, one model at a time
#   ./run_training.sh snn_mlp      # train a single model
#   EPOCHS=1 ./run_training.sh     # quick smoke run
#
# Time steps follow the paper: the standard SNN uses T=32, QSNN and QDSNN use T=4.
# --qat is REQUIRED for the quantized models; without it they train in FP32.
#
set -uo pipefail
cd "$(dirname "$0")"

# Prefer the project venv (it has the CUDA build); fall back to python on PATH.
PYTHON="../venv/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON="python"

EPOCHS=${EPOCHS:-20}
LR=${LR:-0.001}
SNN_T=${SNN_T:-32}   # paper: standard SNN at T=32
QNT_T=${QNT_T:-4}    # paper: QSNN and QDSNN at T=4

mkdir -p results/logs

run () {
  name=$1; shift
  log="results/logs/train_${name}.log"
  raw="results/logs/train_${name}.raw"
  echo ""
  echo "=================================================================="
  echo ">>> $name"
  echo ">>> train.py $*"
  echo "=================================================================="

  # Show live progress on the console (tqdm bar) while capturing everything.
  ok=1
  "$PYTHON" train.py "$@" 2>&1 | tee "$raw" || ok=0

  # Write a CLEAN log: split the \r progress redraws into separate lines, then drop
  # every tqdm bar line (they all carry an "it/s" / "s/it" rate, or start with the
  # bar's description). Tracebacks and epoch summaries have neither, so they survive.
  sed 's/\r/\n/g' "$raw" \
    | grep -vE "it/s|s/it|^[[:space:]]*(Training|Validation|Evaluating):|^[[:space:]]*$" \
    > "$log"
  rm -f "$raw"

  if [ "$ok" -eq 1 ]; then
    echo ">>> $name: DONE  (clean log: $log)"
  else
    echo ">>> $name: FAILED (see $log)" >&2
    return 1
  fi
}

# --- models --------------------------------------------------------------
train_mlp ()         { run standard_mlp --model mlp         --epochs "$EPOCHS" --lr "$LR"; }
train_branchy_mlp () { run branchy_mlp  --model branchy_mlp --epochs "$EPOCHS" --lr "$LR"; }
train_snn_mlp ()     { run snn_mlp      --model snn_mlp     --epochs "$EPOCHS" --lr "$LR" --time_steps "$SNN_T"; }

train_qsnn ()        { run quantized_snn_mlp --model quantized_snn_mlp \
                           --epochs "$EPOCHS" --lr "$LR" --time_steps "$QNT_T" --qat; }

# QDSNN with the RL agent learning the exit thresholds (no --thresholds passed).
train_qdsnn_rl ()    { run qdsnn_rl --model branchy_quantized_snn_mlp \
                           --epochs "$EPOCHS" --lr "$LR" --time_steps "$QNT_T" --qat; }

# The paper's two fixed-threshold configurations. Passing --thresholds disables the
# RL agent, which is what the paper's "Cfg:" rows actually are.
train_qdsnn_02_04 () { run qdsnn_cfg0204 --model branchy_quantized_snn_mlp \
                           --epochs "$EPOCHS" --lr "$LR" --time_steps "$QNT_T" --qat \
                           --thresholds 0.2 0.4 \
                           --save_path results/branchy_quantized_snn_mlp_cfg0204.pth; }

train_qdsnn_06_07 () { run qdsnn_cfg0607 --model branchy_quantized_snn_mlp \
                           --epochs "$EPOCHS" --lr "$LR" --time_steps "$QNT_T" --qat \
                           --thresholds 0.6 0.7 \
                           --save_path results/branchy_quantized_snn_mlp_cfg0607.pth; }

# --- dispatch ------------------------------------------------------------
echo "python : $PYTHON"
echo "epochs : $EPOCHS   lr: $LR   SNN T=$SNN_T   QSNN/QDSNN T=$QNT_T"

case "${1:-all}" in
  mlp)         train_mlp ;;
  branchy_mlp) train_branchy_mlp ;;
  snn_mlp)     train_snn_mlp ;;
  qsnn)        train_qsnn ;;
  qdsnn)       train_qdsnn_rl ;;
  qdsnn_cfgs)  train_qdsnn_02_04 && train_qdsnn_06_07 ;;
  all)
    train_mlp
    train_branchy_mlp
    train_snn_mlp
    train_qsnn
    train_qdsnn_rl
    train_qdsnn_02_04
    train_qdsnn_06_07
    ;;
  *)
    echo "unknown target: $1" >&2
    echo "valid: mlp | branchy_mlp | snn_mlp | qsnn | qdsnn | qdsnn_cfgs | all" >&2
    exit 1
    ;;
esac

echo ""
echo "Done. Checkpoints in results/, logs in results/logs/."
