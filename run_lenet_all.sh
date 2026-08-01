#!/bin/bash
# All three LeNet spiking models on the Loihi sim, sequentially.
# ONE Lava job at a time -- concurrent runtimes deadlock on Windows.
set -x
P=./venv-lava/Scripts/python.exe
E=lava_loihi_sim/exports

$P -u lava_loihi_sim/sparq/lava_build_mlp.py   --npz $E/lenet_qsnn.npz     --n 50 > $E/lenet_qsnn_lava.log 2>&1
$P -u lava_loihi_sim/sparq/lava_build_qdsnn.py --npz $E/lenet_qdsnn.npz    --n 50 > $E/lenet_qdsnn_lava.log 2>&1
$P -u lava_loihi_sim/sparq/lava_build_mlp.py   --npz $E/lenet_snn.npz      --n 25 > $E/lenet_snn_lava.log 2>&1
$P -u lava_loihi_sim/sparq/lava_build_mlp.py   --npz $E/lenet_qsnn_T16.npz --n 50 > $E/lenet_qsnn_T16_lava.log 2>&1
echo "ALL LENET RUNS DONE"
