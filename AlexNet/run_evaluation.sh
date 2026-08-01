#!/bin/bash
# Script to run evaluation and visualization

# Create directories if they don't exist
mkdir -p results/logs
mkdir -p results/plots

# Evaluate all models
echo "Evaluating all models..."
python evaluate.py --model all

# Generate visualizations
echo "Generating visualizations..."
python simple_visualize.py

echo "Done! Results are in the results/ directory."
