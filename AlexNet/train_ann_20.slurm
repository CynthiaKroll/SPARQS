#!/bin/bash
#SBATCH --job-name=sparqs_ann20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ann20_%j.out
#SBATCH --error=logs/ann20_%j.err

module add anaconda/3-2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparqs

cd "$HOME/Thesis_Prep/SPARQS/AlexNet"

python train.py \
  --model standard_alexnet \
  --epochs 20 \
  --batch_size 128 \
  --output_model outputs/alexnet_ann_20epochs.pth
