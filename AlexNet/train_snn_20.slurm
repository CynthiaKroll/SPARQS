#!/bin/bash
#SBATCH --job-name=sparqs_snn20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/snn20_%j.out
#SBATCH --error=logs/snn20_%j.err

module add anaconda/3-2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparqs

cd "$HOME/Thesis_Prep/SPARQS/AlexNet"

python train_quantized_snn_alexnet.py \
  --no_qat \
  --epochs 20 \
  --batch_size 16 \
  --time_steps 32 \
  --output_model outputs/alexnet_snn_20epochs.pth
