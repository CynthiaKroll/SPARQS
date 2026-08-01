#!/bin/bash
# Run SNN Backbone Lava Neuromorphic Simulation
#
# Usage: bash scripts/run_lava_simulation.sh [checkpoint] [num_samples]
#
# Requires: lava-nc (optional, falls back to PyTorch LIF simulation)
# Install:  .venv/bin/pip install lava-nc lava-dl

set -e

CHECKPOINT="${1:-checkpoints/best.pth}"
NUM_SAMPLES="${2:-100}"
OUTPUT="power_report.txt"

cd "$(dirname "$0")/.."

echo "=== SNN Backbone Neuromorphic Power Estimation ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples:    ${NUM_SAMPLES}"
echo "Output:     ${OUTPUT}"
echo ""

# Check if lava-nc is installed
if .venv/bin/python -c "import lava" 2>/dev/null; then
    echo "Lava detected — using Loihi2SimCfg simulation"
    LAVA_FLAG="--use_lava"
else
    echo "Lava not installed — using PyTorch LIF simulation (equivalent behavior)"
    echo "To install: .venv/bin/pip install lava-nc lava-dl"
    LAVA_FLAG=""
fi

echo ""

.venv/bin/python -m src.utils.lava_snn \
    --checkpoint "${CHECKPOINT}" \
    --data_root data/gen1 \
    --num_samples "${NUM_SAMPLES}" \
    --output "${OUTPUT}" \
    ${LAVA_FLAG}

echo ""
echo "Done. Report saved to ${OUTPUT}"
