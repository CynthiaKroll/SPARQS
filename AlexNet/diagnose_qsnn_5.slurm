#!/bin/bash
#SBATCH --job-name=qsnn_diag5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qsnn_diag5_%j.out
#SBATCH --error=logs/qsnn_diag5_%j.err

module add anaconda/3-2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparqs

cd "$HOME/Thesis_Prep/SPARQS/AlexNet"

python train_quantized_snn_alexnet.py \
  --epochs 5 \
  --batch_size 16 \
  --time_steps 32 \
  --lr 5e-5 \
  --output_model outputs/alexnet_qsnn_diagnostic_5epochs_qat.pth
