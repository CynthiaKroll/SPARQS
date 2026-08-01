#!/bin/bash
# Stage A: the three FP32 ANN models. These have no time dimension (no T=32
# unroll), so they are cheap and fit comfortably at batch 128 on a 20 GB MIG
# slice. standard_alexnet is also the KD teacher that train.py looks for at
# results/standard_alexnet.pth, so it is trained first.
#SBATCH --job-name=sparq_ann
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%x_%j.log

set -e

# --- activate environment ---
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate sparq

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2

# No T unroll on these, so the QAT memory ceiling does not apply.
BS="${ANN_BS:-128}"
EPOCHS="${ANN_EPOCHS:-50}"

echo "=== node=$(hostname) job=$SLURM_JOB_ID batch=$BS epochs=$EPOCHS ==="
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

mkdir -p results

echo "==== 1/3 Standard AlexNet (FP32 ANN baseline / KD teacher) ===="
python train.py \
  --model standard_alexnet \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr 0.001 \
  --output_model results/standard_alexnet.pth

echo "==== 2/3 Branchy AlexNet (early-exit ANN) ===="
python train.py \
  --model branchy_alexnet \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr 0.001 \
  --output_model results/branchy_alexnet.pth

echo "==== 3/3 AlexNet QCFS (ANN->SNN conversion source) ===="
python train.py \
  --model alexnet_qcfs \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr 0.001 \
  --output_model results/alexnet_qcfs.pth

echo "==== Stage A done: standard / branchy / qcfs checkpoints in results/ ===="
