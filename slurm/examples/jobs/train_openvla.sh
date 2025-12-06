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
#
# 注: OpenVLAはTFRecordを直接読み込むため、LeRobot変換は不要です。
#
# 使用方法:
#   1. このファイルを jobs/ にコピー: cp examples/jobs/train_openvla.sh jobs/
#   2. 環境に合わせて編集
#   3. slurm-submit submit jobs/train_openvla.sh
# ============================================================================

set -euo pipefail

echo "=========================================="
echo "OpenVLA Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Start time: $(date)"
echo "=========================================="

# =============================================================================
# Configuration - 環境に合わせて編集してください
# =============================================================================

# 作業ディレクトリ
WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/crane_x7_vla}"
cd "${WORKDIR}"

# データパス設定
DATA_ROOT="${DATA_ROOT:-./data/tfrecord_logs}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/crane_x7_openvla}"

# トレーニング設定
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

# W&B設定（オプション）
# export WANDB_API_KEY=your-api-key
# export WANDB_PROJECT=crane_x7_openvla
# export WANDB_ENTITY=your-entity

# =============================================================================
# Environment Setup
# =============================================================================
mkdir -p logs

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

echo ""
echo "=== Configuration ==="
echo "WORKDIR: ${WORKDIR}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo ""

# =============================================================================
# OpenVLA Training
# =============================================================================
echo "=========================================="
echo "OpenVLA Training"
echo "=========================================="

# GPUメモリ情報を表示
nvidia-smi || true

echo ""
echo "Starting OpenVLA training..."

# Singularityコンテナで実行する場合
# singularity exec --nv \
#     --bind $PWD:/workspace \
#     --pwd /workspace \
#     containers/openvla.sif \
#     python -m crane_x7_vla.training.cli train openvla ...

# 直接実行
# OpenVLAはTFRecord形式を直接読み込む
python -m crane_x7_vla.training.cli train openvla \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --experiment-name "crane_x7_openvla" \
    --training-batch-size "${BATCH_SIZE}" \
    --training-learning-rate "${LEARNING_RATE}" \
    --training-max-steps "${MAX_STEPS}" \
    --training-save-interval "${SAVE_INTERVAL}" \
    --training-eval-interval "${EVAL_INTERVAL}" \
    --training-gradient-checkpointing \
    --use-lora

TRAIN_EXIT_CODE=$?

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Checkpoints: ${OUTPUT_DIR}/checkpoints"
echo "=========================================="

exit ${TRAIN_EXIT_CODE}
