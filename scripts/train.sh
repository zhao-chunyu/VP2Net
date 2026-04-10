#!/bin/bash
cd "$(dirname "$0")/.."

PARTITION="driving event recognition"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

network=$1
dataset=$2
gpu_id=$3

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="$SCRIPT_DIR/train_sh.log"

echo "========== TRAINING START =========="
echo "Network: $network"
echo "Dataset: $dataset"
echo "GPU ID: $gpu_id"
echo "Timestamp: $timestamp"
echo "===================================="


CUDA_VISIBLE_DEVICES=$gpu_id python train.py --config configs/$dataset/$network.yaml


end_time=$(date "+%Y-%m-%d_%H-%M-%S")
echo "[$end_time] $network, $dataset, $gpu_id, Training finished."
echo "[$end_time] $network, $dataset, $gpu_id, Training finished." >> "$log_file"
