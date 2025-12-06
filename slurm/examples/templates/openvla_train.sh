#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# OpenVLA Training Template for Slurm
# =============================================================================
#
# このテンプレートはOpenVLAトレーニングを実行します。
#
# 注: OpenVLAはTFRecordを直接読み込むため、LeRobot変換は不要です。
#
# 使用方法:
#   slurm-submit submit jobs/openvla_train.sh
#
# =============================================================================

#SBATCH --job-name={{SLURM_JOB_PREFIX}}_openvla_sweep
#SBATCH --partition={{SLURM_PARTITION}}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{SLURM_CPUS}}
#SBATCH --gpus-per-task={{SLURM_GPUS}}
#SBATCH --time={{SLURM_TIME}}
#SBATCH --output=logs/openvla_sweep_%j.out
#SBATCH --error=logs/openvla_sweep_%j.err

#SBATCH --container={{SLURM_CONTAINER}}

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "OpenVLA Training"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=========================================="

# 作業ディレクトリに移動
cd "${SLURM_SUBMIT_DIR:-{{SLURM_REMOTE_WORKDIR}}}"
echo "Working directory: $(pwd)"

# ログディレクトリを作成
mkdir -p logs

# 環境変数の設定
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# W&B設定（オプション）
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}
export WANDB_MODE=${WANDB_MODE:-online}

# データパス設定
DATA_ROOT=${DATA_ROOT:-{{DATA_ROOT}}}
OUTPUT_DIR=${OUTPUT_DIR:-{{OUTPUT_DIR}}}


# トレーニング設定（Sweepでオーバーライド可能）
BATCH_SIZE=${BATCH_SIZE:-{{batch_size}}}
LEARNING_RATE=${LEARNING_RATE:-{{learning_rate}}}
MAX_STEPS=${MAX_STEPS:-{{MAX_STEPS}}}
SAVE_INTERVAL=${SAVE_INTERVAL:-{{SAVE_INTERVAL}}}
EVAL_INTERVAL=${EVAL_INTERVAL:-{{EVAL_INTERVAL}}}

# デフォルト値（テンプレートプレースホルダが未置換の場合）
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}

echo ""
echo "=== Configuration ==="
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
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Save interval: ${SAVE_INTERVAL}"
echo "  Eval interval: ${EVAL_INTERVAL}"
echo ""

# OpenVLAトレーニングを実行
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
