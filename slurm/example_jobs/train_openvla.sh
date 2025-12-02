#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

#SBATCH --job-name=crane_x7_openvla
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/openvla_%j.out
#SBATCH --error=logs/openvla_%j.err

# ============================================================================
# OpenVLA Training Job Script
# ============================================================================

set -euo pipefail

echo "=========================================="
echo "OpenVLA Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# 作業ディレクトリに移動
cd "${SLURM_SUBMIT_DIR:-$HOME/crane_x7_vla}"

# ログディレクトリを作成
mkdir -p logs

# 環境変数の設定
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# Singularityコンテナで実行する場合の例
# singularity exec --nv \
#     --bind $PWD:/workspace \
#     --pwd /workspace \
#     containers/openvla.sif \
#     python -m crane_x7_vla.training.cli train \
#         --dataset-path /workspace/data/tfrecord_logs \
#         --output-dir /workspace/outputs/crane_x7_openvla \
#         --batch-size 4 \
#         --epochs 10

# 直接実行する場合
python -m crane_x7_vla.training.cli train \
    --dataset-path ./data/tfrecord_logs \
    --output-dir ./outputs/crane_x7_openvla \
    --batch-size 4 \
    --epochs 10

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
