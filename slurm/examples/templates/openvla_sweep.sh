#!/bin/bash
# =============================================================================
# OpenVLA Sweep Job Template for Slurm
# =============================================================================
#
# このテンプレートはW&B Sweep統合用のSlurmジョブスクリプトです。
# crane_x7_vla.training.cli の agent コマンドを使用して
# Sweepからパラメータを取得し、RunをSweepに正しく関連付けます。
#
# 使用方法:
#   slurm-submit sweep start examples/sweeps/sweep_openvla.yaml \
#     --template examples/templates/openvla_sweep.sh -n 10
#
# =============================================================================

#SBATCH --partition={{SLURM_PARTITION}}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{SLURM_CPUS}}
#SBATCH --gpus-per-task={{SLURM_GPUS}}
#SBATCH --mem={{SLURM_MEM}}
#SBATCH --container={{SLURM_CONTAINER}}

# =============================================================================
# Environment Setup
# =============================================================================
echo "=== Environment Information ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
cat /etc/os-release
echo ""

# W&B Configuration
export WANDB_MODE=online
export WANDB_API_KEY={{WANDB_API_KEY}}
export WANDB_PROJECT={{WANDB_PROJECT}}
export WANDB_ENTITY={{WANDB_ENTITY}}

# Python/CUDA Configuration
export PYTHONUNBUFFERED=1

# =============================================================================
# Sweep Agent Execution
# =============================================================================
echo "=== Starting W&B Sweep Agent ==="
echo "Sweep ID: {{SWEEP_ID}}"
echo "Entity: {{WANDB_ENTITY}}"
echo "Project: {{WANDB_PROJECT}}"
echo ""

# crane_x7_vla の agent コマンドを使用してSweepからパラメータを取得し、トレーニングを実行
# wandb.agent()が内部で呼ばれ、RunがSweepに正しく関連付けられる
python -m crane_x7_vla.training.cli agent openvla \
  --sweep-id {{SWEEP_ID}} \
  --entity {{WANDB_ENTITY}} \
  --project {{WANDB_PROJECT}} \
  --data-root {{DATA_ROOT}} \
  --output-dir {{OUTPUT_DIR}} \
  --experiment-name crane_x7_sweep \
  --training-max-steps {{MAX_STEPS}} \
  --training-save-interval {{SAVE_INTERVAL}} \
  --training-eval-interval {{EVAL_INTERVAL}} \
  --overfitting-overfit-check-interval {{OVERFIT_CHECK_INTERVAL}} \
  --training-gradient-checkpointing \
  --use-lora

echo "=== Job Completed ==="
echo "End time: $(date)"
