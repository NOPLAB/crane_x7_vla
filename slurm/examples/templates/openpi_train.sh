#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# OpenPI Training Template for Slurm
# =============================================================================
#
# このテンプレートはTFRecordデータの変換からOpenPIトレーニングまでの
# 完全なパイプラインを実行します。
#
# OpenPIはLeRobot形式のデータを必要とするため、TFRecordからの変換が必須です。
#
# 使用方法:
#   slurm-submit submit jobs/openpi_train.sh
#
# =============================================================================

#SBATCH --job-name={{SLURM_JOB_PREFIX}}_openpi
#SBATCH --partition={{SLURM_PARTITION}}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{SLURM_CPUS}}
#SBATCH --gpus-per-task={{SLURM_GPUS}}
#SBATCH --mem={{SLURM_MEM}}
#SBATCH --time={{SLURM_TIME}}
#SBATCH --output=logs/openpi_%j.out
#SBATCH --error=logs/openpi_%j.err

#SBATCH --container={{SLURM_CONTAINER}}

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
echo "=========================================="
echo "OpenPI Training Pipeline (JAX/Flax)"
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

# JAX設定
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# W&B設定（オプション）
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}
export WANDB_MODE=${WANDB_MODE:-online}

# データパス設定
DATA_ROOT=${DATA_ROOT:-{{DATA_ROOT}}}
OUTPUT_DIR=${OUTPUT_DIR:-{{OUTPUT_DIR}}}
LEROBOT_DATASET_NAME=${LEROBOT_DATASET_NAME:-crane_x7_openpi}

# トレーニング設定（Sweepでオーバーライド可能）
BATCH_SIZE=${BATCH_SIZE:-{{batch_size}}}
LEARNING_RATE=${LEARNING_RATE:-{{learning_rate}}}
MAX_STEPS=${MAX_STEPS:-{{MAX_STEPS}}}
SAVE_INTERVAL=${SAVE_INTERVAL:-{{SAVE_INTERVAL}}}
EVAL_INTERVAL=${EVAL_INTERVAL:-{{EVAL_INTERVAL}}}

# OpenPI固有設定
MODEL_TYPE=${MODEL_TYPE:-pi0_fast}
ACTION_HORIZON=${ACTION_HORIZON:-50}
LORA_RANK=${LORA_RANK:-32}

# デフォルト値（テンプレートプレースホルダが未置換の場合）
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
MAX_STEPS=${MAX_STEPS:-200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}

echo ""
echo "=== Configuration ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "LEROBOT_DATASET_NAME: ${LEROBOT_DATASET_NAME}"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "ACTION_HORIZON: ${ACTION_HORIZON}"
echo "LORA_RANK: ${LORA_RANK}"
echo ""

# =============================================================================
# Step 1: TFRecord to LeRobot Conversion (Required for OpenPI)
# =============================================================================
echo "=========================================="
echo "Step 1: TFRecord to LeRobot Conversion"
echo "=========================================="

# LeRobotデータセットの保存先
export HF_LEROBOT_HOME="${OUTPUT_DIR}/lerobot_datasets"
mkdir -p "${HF_LEROBOT_HOME}"

# 既存のLeRobotデータセットをチェック
LEROBOT_DATASET_PATH="${HF_LEROBOT_HOME}/${LEROBOT_DATASET_NAME}"
if [ -d "${LEROBOT_DATASET_PATH}" ]; then
    echo "LeRobot dataset already exists at ${LEROBOT_DATASET_PATH}"
    echo "Skipping conversion..."
else
    echo "Converting TFRecord data to LeRobot format..."
    echo "Input: ${DATA_ROOT}"
    echo "Output: ${LEROBOT_DATASET_PATH}"

    python -m crane_x7_vla.scripts.convert_to_lerobot \
        --data_dir "${DATA_ROOT}" \
        --output_repo "${LEROBOT_DATASET_NAME}" \
        --fps 10 \
        --overwrite

    echo "Conversion completed!"
fi

# =============================================================================
# Step 2: Compute Normalization Statistics
# =============================================================================
echo ""
echo "=========================================="
echo "Step 2: Compute Normalization Statistics"
echo "=========================================="

NORM_STATS_PATH="${OUTPUT_DIR}/norm_stats"
mkdir -p "${NORM_STATS_PATH}"

if [ -f "${NORM_STATS_PATH}/action_stats.json" ]; then
    echo "Normalization statistics already exist at ${NORM_STATS_PATH}"
    echo "Skipping computation..."
else
    echo "Computing normalization statistics..."
    python -m crane_x7_vla.scripts.compute_crane_x7_norm_stats \
        --data_dir "${DATA_ROOT}" \
        --output_dir "${NORM_STATS_PATH}"
    echo "Statistics computation completed!"
fi

# =============================================================================
# Step 3: OpenPI Training (JAX/Flax)
# =============================================================================
echo ""
echo "=========================================="
echo "Step 3: OpenPI Training (JAX/Flax)"
echo "=========================================="

# GPUメモリ情報を表示
nvidia-smi || true

# JAXがGPUを認識しているか確認
echo ""
echo "JAX device check:"
python -c "import jax; print(f'JAX devices: {jax.devices()}')" || echo "JAX not available"

echo ""
echo "Starting OpenPI training..."
echo "  Model type: ${MODEL_TYPE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Action horizon: ${ACTION_HORIZON}"
echo "  LoRA rank: ${LORA_RANK}"
echo ""

# OpenPIトレーニングを実行
# OpenPIはLeRobot形式を使用
python -m crane_x7_vla.training.cli train openpi \
    --data-root "${LEROBOT_DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --experiment-name "crane_x7_openpi" \
    --training-batch-size "${BATCH_SIZE}" \
    --training-learning-rate "${LEARNING_RATE}" \
    --training-max-steps "${MAX_STEPS}" \
    --training-save-interval "${SAVE_INTERVAL}" \
    --training-eval-interval "${EVAL_INTERVAL}" \
    --backend-config-model-type "${MODEL_TYPE}" \
    --backend-config-action-horizon "${ACTION_HORIZON}" \
    --backend-config-lora-rank "${LORA_RANK}" \
    --use-lora

TRAIN_EXIT_CODE=$?

# =============================================================================
# Cleanup and Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Training Pipeline Completed"
echo "=========================================="
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Checkpoints: ${OUTPUT_DIR}/checkpoints"
echo "  LeRobot Dataset: ${LEROBOT_DATASET_PATH}"
echo "  Normalization Stats: ${NORM_STATS_PATH}"
echo "=========================================="

exit ${TRAIN_EXIT_CODE}
