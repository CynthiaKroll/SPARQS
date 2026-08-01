#!/bin/bash

# Evaluation script for 5-Layer MLP models on MNIST

echo "Evaluating 5-Layer MLP models on MNIST dataset..."

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directories if they don't exist
mkdir -p results/logs
mkdir -p results/plots

# Run evaluation and visualization
echo "Running evaluation and generating plots..."
python evaluate.py

echo "Generating visualization plots..."
python visualize.py

echo "Evaluation complete! Check results/logs for CSV files and results/plots for visualizations." 