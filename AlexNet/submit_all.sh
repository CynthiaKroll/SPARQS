#!/bin/bash
# Queue every AlexNet model on SLURM with the right ordering.
#
#   Stage A (ANNs)  ─┐
#                    ├─> Stage C (QDSNN + paired evaluation)
#   Stage B (QSNN)  ─┘
#
# A and B are independent, so they occupy the two MIG slots in parallel.
# C waits for BOTH: it needs Stage B's INT8 checkpoint for the paired
# comparison, and Stage A's standard_alexnet if you enable KD (--kd_alpha < 1).
#
# Usage:  ./submit_all.sh          (or:  BS=32 ./submit_all.sh)
set -e

cd "$(dirname "$0")"

A=$(sbatch --parsable slurm_ann.sh)
echo "Stage A (ANNs)  -> job $A"

B=$(sbatch --parsable slurm_qsnn.sh)
echo "Stage B (QSNN)  -> job $B"

C=$(sbatch --parsable --dependency=afterok:$A:$B slurm_qdsnn.sh)
echo "Stage C (QDSNN) -> job $C  (waits on $A and $B)"

echo
squeue -u "$USER" -l
