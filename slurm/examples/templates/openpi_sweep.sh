#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop
#
# =============================================================================
# OpenPI Sweep Job Template for Slurm (JAX/Flax)
# =============================================================================
#
# このテンプレートはW&B Sweep統合用のSlurmジョブスクリプトです。
# crane_x7_vla.training.cli の agent コマンドを使用して
# Sweepからパラメータを取得し、RunをSweepに正しく関連付けます。
#
# 重要: OpenPIはLeRobot形式のデータを必要とするため、TFRecordからの
#       変換が必須です（初回のみ実行）。
#
# 使用方法:
#   slurm-submit sweep start examples/sweeps/sweep_openpi.yaml \
#     --template examples/templates/openpi_sweep.sh --max-runs 10
#
# =============================================================================

#SBATCH --job-name={{SLURM_JOB_PREFIX}}_openpi_sweep
#SBATCH --partition={{SLURM_PARTITION}}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{SLURM_CPUS}}
#SBATCH --gpus-per-task={{SLURM_GPUS}}
#SBATCH --mem={{SLURM_MEM}}
#SBATCH --time={{SLURM_TIME}}
#SBATCH --output=logs/openpi_sweep_%j.out
#SBATCH --error=logs/openpi_sweep_%j.err

#SBATCH --container={{SLURM_CONTAINER}}

set -euo pipefail

# =============================================================================
# Environment Setup
# =============================================================================
echo "=== Environment Information ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
cat /etc/os-release 2>/dev/null || true
echo ""

# 作業ディレクトリに移動
cd "{{SLURM_CONTAINER_WORKDIR}}"
echo "Working directory: $(pwd)"

# ログディレクトリを作成
mkdir -p logs

# W&B Configuration
export WANDB_MODE=online
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}

# Python/CUDA/JAX Configuration
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# データパス設定
DATA_ROOT=${DATA_ROOT:-{{DATA_ROOT}}}
OUTPUT_DIR=${OUTPUT_DIR:-{{OUTPUT_DIR}}}
LEROBOT_DATASET_NAME=${LEROBOT_DATASET_NAME:-crane_x7_openpi}

# トレーニング設定（Sweepでオーバーライドされない固定パラメータ）
MAX_STEPS=${MAX_STEPS:-{{MAX_STEPS}}}
SAVE_INTERVAL=${SAVE_INTERVAL:-{{SAVE_INTERVAL}}}
EVAL_INTERVAL=${EVAL_INTERVAL:-{{EVAL_INTERVAL}}}
OVERFIT_CHECK_INTERVAL=${OVERFIT_CHECK_INTERVAL:-{{OVERFIT_CHECK_INTERVAL}}}

# デフォルト値
MAX_STEPS=${MAX_STEPS:-200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
OVERFIT_CHECK_INTERVAL=${OVERFIT_CHECK_INTERVAL:-500}

echo "=== Sweep Configuration ==="
echo "SWEEP_ID: {{SWEEP_ID}}"
echo "WANDB_ENTITY: {{WANDB_ENTITY}}"
echo "WANDB_PROJECT: {{WANDB_PROJECT}}"
echo ""
echo "=== Data Configuration ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo ""

# =============================================================================
# Step 1: Data Preparation (Required for OpenPI)
# =============================================================================
echo "=== Step 1: Data Preparation (LeRobot Conversion) ==="

# LeRobotデータセットの保存先
export HF_LEROBOT_HOME="${OUTPUT_DIR}/lerobot_datasets"
mkdir -p "${HF_LEROBOT_HOME}"

LEROBOT_DATASET_PATH="${HF_LEROBOT_HOME}/${LEROBOT_DATASET_NAME}"

# LeRobotデータセットに変換（初回のみ、OpenPIには必須）
if [ -d "${LEROBOT_DATASET_PATH}" ]; then
    echo "LeRobot dataset already exists, skipping conversion..."
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

# 正規化統計の計算（初回のみ）
NORM_STATS_PATH="${OUTPUT_DIR}/norm_stats"
mkdir -p "${NORM_STATS_PATH}"

if [ -f "${NORM_STATS_PATH}/action_stats.json" ]; then
    echo "Normalization statistics already exist, skipping..."
else
    echo "Computing normalization statistics..."
    python -m crane_x7_vla.scripts.compute_crane_x7_norm_stats \
        --data_dir "${DATA_ROOT}" \
        --output_dir "${NORM_STATS_PATH}"
    echo "Statistics computation completed!"
fi

# =============================================================================
# Step 2: Sweep Agent Execution
# =============================================================================
echo ""
echo "=== Step 2: Starting W&B Sweep Agent (JAX/Flax) ==="
echo "Sweep ID: {{SWEEP_ID}}"
echo "Entity: {{WANDB_ENTITY}}"
echo "Project: {{WANDB_PROJECT}}"
echo ""

# GPUメモリ情報を表示
nvidia-smi || true

# JAXがGPUを認識しているか確認
echo ""
echo "JAX device check:"
python -c "import jax; print(f'JAX devices: {jax.devices()}')" || echo "JAX not available"
echo ""

# crane_x7_vla の agent コマンドを使用してSweepからパラメータを取得し、トレーニングを実行
# wandb.agent()が内部で呼ばれ、RunがSweepに正しく関連付けられる
# OpenPIはLeRobot形式を使用するため、変換済みデータを指定
python -m crane_x7_vla.training.cli agent openpi \
    --sweep-id "{{SWEEP_ID}}" \
    --entity "{{WANDB_ENTITY}}" \
    --project "{{WANDB_PROJECT}}" \
    --data-root "${LEROBOT_DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}/checkpoints" \
    --experiment-name "crane_x7_sweep" \
    --training-max-steps "${MAX_STEPS}" \
    --training-save-interval "${SAVE_INTERVAL}" \
    --training-eval-interval "${EVAL_INTERVAL}" \
    --overfitting-overfit-check-interval "${OVERFIT_CHECK_INTERVAL}" \
    --use-lora

echo "=== Job Completed ==="
echo "End time: $(date)"
