#!/bin/bash
# Stage C: QDSNN (same backbone + 2 RL-gated early exits), then the paired
# evaluation that the paper's power/delay claim rests on.
# Do not submit this directly -- it depends on Stage A and Stage B.
# Use  ./submit_all.sh  which wires up the dependencies.
#SBATCH --job-name=sparq_qdsnn
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.log

set -e

# --- activate environment (must match slurm_qsnn.sh) ---
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate sparq

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=2
BS="${BS:-64}"

echo "=== node=$(hostname) job=$SLURM_JOB_ID batch=$BS ==="
nvidia-smi

mkdir -p results

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

echo "==== Evaluating QSNN vs QDSNN (identical measurement path) ===="
python evaluate_quantized_pair.py \
  --qsnn_path  results/quantized_snn_alexnet_qat.pth \
  --qdsnn_path results/branchy_quantized_snn_qat.pth \
  --agent_path results/rl_agent_qat.pkl \
  --output_csv results/qsnn_vs_qdsnn.csv

echo "==== Done. Results table: results/qsnn_vs_qdsnn.csv ===="
