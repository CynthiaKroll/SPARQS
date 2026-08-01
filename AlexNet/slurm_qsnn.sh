#!/bin/bash
# Stage B: QSNN baseline (quantized SNN AlexNet, no early exits).
# Submitted by ./submit_all.sh, or standalone with:  sbatch slurm_qsnn.sh
#SBATCH --job-name=sparq_qsnn
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.log

set -e

# --- activate environment ---
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate sparq

cd "$SLURM_SUBMIT_DIR"

# num_workers=4 per loader in data_loader.py; keep threads from oversubscribing.
export OMP_NUM_THREADS=2

# Batch 64, not 128: the MIG slice is a 20 GB half-A100, and at T=32 the QAT
# backward graph spans all 32 timesteps. 32 was the 8 GB figure, so 64 is the
# proportionate scale-up with headroom. Drop to 32 if this OOMs.
BS="${BS:-64}"

echo "=== node=$(hostname) job=$SLURM_JOB_ID batch=$BS ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

mkdir -p results

echo "==== Training Quantized SNN AlexNet (QSNN baseline) ===="
python train_quantized_snn_alexnet.py \
  --epochs 20 \
  --batch_size "$BS" \
  --lr 0.001 \
  --output_model results/quantized_snn_alexnet_qat.pth

echo "==== Stage 1 done: results/quantized_snn_alexnet_qat_int8.pth ===="
