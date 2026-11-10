#!/bin/bash
#SBATCH --job-name=qsnn_int8_eval
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qsnn_int8_eval_%j.out
#SBATCH --error=logs/qsnn_int8_eval_%j.err

module add anaconda/3-2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparqs

cd "$HOME/Thesis_Prep/SPARQS/AlexNet"

python evaluate_quantized_snn.py \
  --model_path outputs/alexnet_qsnn_20epochs_lr5e5_qat_int8.pth \
  --time_steps 32 \
  --batch_size 16 \
  --encoding poisson \
  --output_method mean \
  --beta 0.95
