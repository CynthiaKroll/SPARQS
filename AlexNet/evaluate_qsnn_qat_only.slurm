#!/bin/bash
#SBATCH --job-name=qsnn_qat_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/qsnn_qat_eval_%j.out
#SBATCH --error=logs/qsnn_qat_eval_%j.err

module add anaconda/3-2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparqs

cd "$HOME/Thesis_Prep/SPARQS/AlexNet"

python evaluate_qsnn_qat_only.py
