#!/bin/bash
# ============================================================================
# Slurm Job Submission Script with W&B Sweep Support
# ============================================================================
# SSHサーバーに接続し、Slurmジョブを投下するスクリプト
# W&B Sweepによるハイパーパラメータ自動探索もサポート
#
# 使い方:
#   ./submit.sh <job_script.sh>           # ジョブスクリプトを投下
#   ./submit.sh <job_script.sh> --dry-run # ドライラン（実際には投下しない）
#   ./submit.sh --status                  # キュー状態を確認
#   ./submit.sh --cancel <job_id>         # ジョブをキャンセル
#
# W&B Sweep:
#   ./submit.sh sweep <config.yaml>       # 新規Sweepを開始
#   ./submit.sh sweep --resume <id>       # 既存Sweepを再開
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
SWEEP_STATE_DIR="${SCRIPT_DIR}/.sweep_state"

# Sweepデフォルト値
SWEEP_BACKEND="openvla"
SWEEP_MAX_RUNS=10
SWEEP_POLL_INTERVAL=300
SWEEP_DRY_RUN=false
SWEEP_RESUME_ID=""

# ----------------------------------------------------------------------------
# 設定読み込み
# ----------------------------------------------------------------------------
load_config() {
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "エラー: .envファイルが見つかりません"
        echo "以下のコマンドで作成してください:"
        echo "  cp ${SCRIPT_DIR}/.env.template ${ENV_FILE}"
        exit 1
    fi

    # .envファイルを読み込み
    set -a
    source "$ENV_FILE"
    set +a

    # 必須項目のチェック
    if [[ -z "${SLURM_SSH_HOST:-}" ]]; then
        echo "エラー: SLURM_SSH_HOST が設定されていません"
        exit 1
    fi
    if [[ -z "${SLURM_SSH_USER:-}" ]]; then
        echo "エラー: SLURM_SSH_USER が設定されていません"
        exit 1
    fi

    # デフォルト値の設定
    SLURM_SSH_PORT="${SLURM_SSH_PORT:-22}"
    SLURM_SSH_KEY="${SLURM_SSH_KEY:-}"
    SLURM_SSH_AUTH="${SLURM_SSH_AUTH:-password}"
    SLURM_REMOTE_WORKDIR="${SLURM_REMOTE_WORKDIR:-~/crane_x7_vla}"
    SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
    SLURM_GPUS="${SLURM_GPUS:-1}"
    SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-}"
    SLURM_TIME="${SLURM_TIME:-24:00:00}"
    SLURM_MEM="${SLURM_MEM:-32G}"
    SLURM_CPUS="${SLURM_CPUS:-8}"
    SLURM_JOB_PREFIX="${SLURM_JOB_PREFIX:-crane_x7}"

    # Sweep用設定
    WANDB_ENTITY="${WANDB_ENTITY:-}"
    WANDB_PROJECT="${WANDB_PROJECT:-crane_x7_sweep}"

    # Sweep状態ディレクトリ作成
    mkdir -p "$SWEEP_STATE_DIR"
}

# ----------------------------------------------------------------------------
# SSH接続
# ----------------------------------------------------------------------------
ssh_cmd() {
    local ssh_opts=(
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o LogLevel=ERROR
        -p "$SLURM_SSH_PORT"
    )

    if [[ "$SLURM_SSH_AUTH" == "key" && -n "$SLURM_SSH_KEY" ]]; then
        local key_path
        key_path=$(eval echo "$SLURM_SSH_KEY")
        ssh_opts+=(-i "$key_path")
    fi

    ssh "${ssh_opts[@]}" "${SLURM_SSH_USER}@${SLURM_SSH_HOST}" "$@"
}

scp_cmd() {
    local scp_opts=(
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o LogLevel=ERROR
        -P "$SLURM_SSH_PORT"
    )

    if [[ "$SLURM_SSH_AUTH" == "key" && -n "$SLURM_SSH_KEY" ]]; then
        local key_path
        key_path=$(eval echo "$SLURM_SSH_KEY")
        scp_opts+=(-i "$key_path")
    fi

    scp "${scp_opts[@]}" "$@"
}

# ----------------------------------------------------------------------------
# ジョブ投下
# ----------------------------------------------------------------------------
submit_job() {
    local job_script="$1"
    local dry_run="${2:-false}"

    if [[ ! -f "$job_script" ]]; then
        echo "エラー: ジョブスクリプトが見つかりません: $job_script"
        exit 1
    fi

    local job_name
    job_name="${SLURM_JOB_PREFIX}_$(basename "$job_script" .sh)"
    local remote_script="${SLURM_REMOTE_WORKDIR}/slurm/$(basename "$job_script")"

    echo "=========================================="
    echo "Slurmジョブ投下"
    echo "=========================================="
    echo "ホスト: ${SLURM_SSH_USER}@${SLURM_SSH_HOST}:${SLURM_SSH_PORT}"
    echo "ジョブスクリプト: $job_script"
    echo "リモートパス: $remote_script"
    echo "ジョブ名: $job_name"
    echo "=========================================="

    if [[ "$dry_run" == "true" ]]; then
        echo "[ドライラン] 実際にはジョブを投下しません"
        echo ""
        echo "投下されるジョブスクリプト:"
        echo "----------------------------------------"
        cat "$job_script"
        echo "----------------------------------------"
        return 0
    fi

    # リモートディレクトリの作成
    echo "リモートディレクトリを作成中..."
    ssh_cmd "mkdir -p ${SLURM_REMOTE_WORKDIR}/slurm"

    # ジョブスクリプトの転送
    echo "ジョブスクリプトを転送中..."
    scp_cmd "$job_script" "${SLURM_SSH_USER}@${SLURM_SSH_HOST}:${remote_script}"

    # ジョブの投下
    echo "ジョブを投下中..."
    local result
    result=$(ssh_cmd "cd ${SLURM_REMOTE_WORKDIR} && sbatch ${remote_script}")

    echo ""
    echo "$result"
    echo ""
    echo "ジョブ状態を確認するには: ./submit.sh --status"
}

# ----------------------------------------------------------------------------
# キュー状態確認
# ----------------------------------------------------------------------------
check_status() {
    echo "=========================================="
    echo "Slurmキュー状態"
    echo "=========================================="
    echo "ホスト: ${SLURM_SSH_USER}@${SLURM_SSH_HOST}:${SLURM_SSH_PORT}"
    echo "=========================================="
    echo ""

    ssh_cmd "squeue -u \$USER -o '%.18i %.12j %.10P %.8T %.10M %.6D %.4C %.10m %R'"
}

# ----------------------------------------------------------------------------
# ジョブキャンセル
# ----------------------------------------------------------------------------
cancel_job() {
    local job_id="$1"

    echo "=========================================="
    echo "ジョブキャンセル"
    echo "=========================================="
    echo "ジョブID: $job_id"
    echo "=========================================="

    ssh_cmd "scancel $job_id"
    echo "ジョブ $job_id をキャンセルしました"
}

# ============================================================================
# W&B Sweep 機能
# ============================================================================

# ----------------------------------------------------------------------------
# Sweep作成
# ----------------------------------------------------------------------------
create_sweep() {
    local config_file="$1"
    local project="${WANDB_PROJECT}"
    local entity="${WANDB_ENTITY}"

    python3 << PYTHON
import wandb
import yaml

with open("${config_file}", "r") as f:
    sweep_config = yaml.safe_load(f)

entity = "${entity}" if "${entity}" else None
project = "${project}"

sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
print(sweep_id)
PYTHON
}

# ----------------------------------------------------------------------------
# Sweepから次のパラメータを取得
# ----------------------------------------------------------------------------
init_sweep_run() {
    local sweep_id="$1"
    local entity="${WANDB_ENTITY}"
    local project="${WANDB_PROJECT}"

    python3 << PYTHON
import wandb
import json
import os
import sys

entity = "${entity}" if "${entity}" else None
project = "${project}"
sweep_id = "${sweep_id}"

# Sweepの状態を確認
api = wandb.Api()
if entity:
    sweep_path = f"{entity}/{project}/{sweep_id}"
else:
    viewer = api.viewer
    entity = viewer.entity
    sweep_path = f"{entity}/{project}/{sweep_id}"

try:
    sweep = api.sweep(sweep_path)
except Exception as e:
    print(f"ERROR:Sweep not found: {e}", file=sys.stderr)
    print("SWEEP_ERROR")
    sys.exit(1)

if sweep.state in ["FINISHED", "CANCELED"]:
    print("SWEEP_DONE")
    sys.exit(0)

# 新しいrunを初期化（sweepに紐づけ）
os.environ["WANDB_SILENT"] = "true"
run = wandb.init(
    project=project,
    entity=entity,
    sweep_id=sweep_id,
    reinit=True,
)

config = dict(run.config)

if not config:
    print("NO_PARAMS", file=sys.stderr)
    run.finish(exit_code=1, quiet=True)
    sys.exit(1)

# 出力
output = {
    "run_id": run.id,
    "run_name": run.name,
    "sweep_id": sweep_id,
    "entity": entity or run.entity,
    "project": project,
    "params": config
}
print(json.dumps(output))

run.finish(quiet=True)
PYTHON
}

# ----------------------------------------------------------------------------
# Sweepジョブスクリプト生成
# ----------------------------------------------------------------------------
generate_sweep_job_script() {
    local run_info="$1"
    local backend="$2"
    local output_file="$3"

    # JSONからパラメータを抽出
    local run_id=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['run_id'])")
    local run_name=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['run_name'])")
    local sweep_id=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sweep_id'])")
    local entity=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('entity', ''))")
    local project=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('project', ''))")
    local params=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['params']))")

    # 共通パラメータを抽出
    local learning_rate=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('learning_rate', '5e-4'))")
    local batch_size=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('batch_size', '4'))")
    local lora_rank=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('lora_rank', '32'))")
    local lora_dropout=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('lora_dropout', '0.05'))")
    local weight_decay=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('weight_decay', '0.01'))")
    local warmup_steps=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('warmup_steps', '1000'))")
    local max_grad_norm=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('max_grad_norm', '1.0'))")

    # OpenPI固有パラメータ
    local model_type=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('model_type', 'pi0_fast'))")
    local action_horizon=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('action_horizon', '50'))")
    local normalization_mode=$(echo "$params" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('normalization_mode', 'quantile'))")

    # ジョブスクリプトヘッダーを生成
    cat > "$output_file" << JOBSCRIPT
#!/bin/bash
#SBATCH --job-name=sweep_${run_name}
#SBATCH --partition=${SLURM_PARTITION:-research}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=${SLURM_CPUS:-16}
#SBATCH --gpus-per-task=${SLURM_GPUS:-1}
#SBATCH --mem=${SLURM_MEM:-128G}
#SBATCH --time=${SLURM_TIME:-24:00:00}
#SBATCH --output=logs/sweep_${run_id}_%j.out
#SBATCH --error=logs/sweep_${run_id}_%j.err
${SLURM_CONTAINER:+#SBATCH --container=${SLURM_CONTAINER}}

mkdir -p logs

export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_MODE=online
export WANDB_RUN_ID="${run_id}"
export WANDB_RESUME="must"
export WANDB_PROJECT="${project}"
${entity:+export WANDB_ENTITY="${entity}"}

cd /workspace/vla

echo "=========================================="
echo "W&B Sweep Run: ${run_name}"
echo "Run ID: ${run_id}"
echo "Sweep ID: ${sweep_id}"
echo "=========================================="
echo "Parameters:"
echo "  learning_rate: ${learning_rate}"
echo "  batch_size: ${batch_size}"
echo "  lora_rank: ${lora_rank}"
echo "  lora_dropout: ${lora_dropout}"
echo "  weight_decay: ${weight_decay}"
echo "  warmup_steps: ${warmup_steps}"
echo "  max_grad_norm: ${max_grad_norm}"
JOBSCRIPT

    # バックエンド固有のトレーニングコマンドを追加
    if [[ "$backend" == "openvla" ]]; then
        cat >> "$output_file" << JOBSCRIPT
echo "=========================================="

python -m crane_x7_vla.training.cli train openvla \\
    --data-root "${DATA_ROOT:-/root/vla/data}" \\
    --output-dir "${OUTPUT_DIR:-/root/vla/output}/${run_id}" \\
    --experiment-name "sweep_${run_name}" \\
    --training-batch-size ${batch_size} \\
    --training-learning-rate ${learning_rate} \\
    --training-weight-decay ${weight_decay} \\
    --training-warmup-steps ${warmup_steps} \\
    --training-max-grad-norm ${max_grad_norm} \\
    --training-num-epochs ${NUM_EPOCHS:-10} \\
    --training-save-interval ${SAVE_INTERVAL:-500} \\
    --training-eval-interval ${EVAL_INTERVAL:-100} \\
    --lora-rank ${lora_rank} \\
    --lora-dropout ${lora_dropout} \\
    --use-lora

echo "Training completed!"
JOBSCRIPT
    elif [[ "$backend" == "openpi" ]]; then
        cat >> "$output_file" << JOBSCRIPT
echo "  model_type: ${model_type}"
echo "  action_horizon: ${action_horizon}"
echo "  normalization_mode: ${normalization_mode}"
echo "=========================================="

python -m crane_x7_vla.training.cli train openpi \\
    --data-root "${DATA_ROOT:-/root/vla/data}" \\
    --output-dir "${OUTPUT_DIR:-/root/vla/output}/${run_id}" \\
    --experiment-name "sweep_${run_name}" \\
    --training-batch-size ${batch_size} \\
    --training-learning-rate ${learning_rate} \\
    --training-weight-decay ${weight_decay} \\
    --training-warmup-steps ${warmup_steps} \\
    --training-max-grad-norm ${max_grad_norm} \\
    --training-num-epochs ${NUM_EPOCHS:-10} \\
    --training-save-interval ${SAVE_INTERVAL:-500} \\
    --training-eval-interval ${EVAL_INTERVAL:-100} \\
    --model-type ${model_type} \\
    --action-horizon ${action_horizon} \\
    --normalization-mode ${normalization_mode} \\
    --lora-rank ${lora_rank} \\
    --lora-dropout ${lora_dropout} \\
    --use-lora

echo "Training completed!"
JOBSCRIPT
    fi
}

# ----------------------------------------------------------------------------
# Sweepジョブ投下と完了待機
# ----------------------------------------------------------------------------
submit_sweep_job_and_wait() {
    local job_script="$1"
    local run_id="$2"

    if [[ "$SWEEP_DRY_RUN" == "true" ]]; then
        echo "[ドライラン] ジョブスクリプト:"
        cat "$job_script"
        echo "---"
        return 0
    fi

    local remote_script="${SLURM_REMOTE_WORKDIR}/slurm/sweep_jobs/$(basename "$job_script")"

    # リモートディレクトリ作成
    ssh_cmd "mkdir -p ${SLURM_REMOTE_WORKDIR}/slurm/sweep_jobs ${SLURM_REMOTE_WORKDIR}/logs"

    # ジョブスクリプト転送
    scp_cmd "$job_script" "${SLURM_SSH_USER}@${SLURM_SSH_HOST}:${remote_script}"

    # ジョブ投下
    local submit_result
    submit_result=$(ssh_cmd "cd ${SLURM_REMOTE_WORKDIR} && sbatch ${remote_script}")
    echo "$submit_result"

    # ジョブIDを抽出
    local job_id
    job_id=$(echo "$submit_result" | grep -oP 'Submitted batch job \K\d+' || echo "")

    if [[ -z "$job_id" ]]; then
        echo "エラー: ジョブIDを取得できませんでした"
        return 1
    fi

    echo "ジョブID: $job_id"
    echo "Run ID: $run_id"

    # ジョブIDを保存
    echo "$job_id" > "${SWEEP_STATE_DIR}/job_${run_id}"

    # ジョブ完了を待機
    wait_for_slurm_job "$job_id"
}

wait_for_slurm_job() {
    local job_id="$1"

    echo "ジョブ $job_id の完了を待機中... (確認間隔: ${SWEEP_POLL_INTERVAL}秒)"

    while true; do
        sleep "$SWEEP_POLL_INTERVAL"

        local status
        status=$(ssh_cmd "squeue -j $job_id -h -o '%T' 2>/dev/null" || echo "COMPLETED")

        case "$status" in
            "PENDING"|"RUNNING"|"CONFIGURING"|"COMPLETING")
                echo "  [$(date '+%H:%M:%S')] ジョブ $job_id: $status"
                ;;
            "COMPLETED"|"")
                echo "ジョブ $job_id が完了しました"
                return 0
                ;;
            "FAILED"|"CANCELLED"|"TIMEOUT"|"NODE_FAIL")
                echo "ジョブ $job_id が失敗しました: $status"
                return 1
                ;;
            *)
                echo "  [$(date '+%H:%M:%S')] ジョブ $job_id: $status"
                ;;
        esac
    done
}

# ----------------------------------------------------------------------------
# Sweepメインループ
# ----------------------------------------------------------------------------
run_sweep() {
    local sweep_config="$1"
    local sweep_id=""

    # W&B APIキーチェック
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "エラー: WANDB_API_KEY が設定されていません"
        exit 1
    fi

    # Sweep作成または再開
    if [[ -n "$SWEEP_RESUME_ID" ]]; then
        sweep_id="$SWEEP_RESUME_ID"
        echo "既存のSweepを再開: $sweep_id"
    else
        echo "新しいSweepを作成中..."
        sweep_id=$(create_sweep "$sweep_config")
        echo "Sweep ID: $sweep_id"
        echo "$sweep_id" > "${SWEEP_STATE_DIR}/current_sweep_id"
    fi

    echo ""
    echo "=========================================="
    echo "W&B Sweep コントローラー"
    echo "=========================================="
    echo "Sweep ID: $sweep_id"
    echo "Backend: $SWEEP_BACKEND"
    echo "最大実行回数: $SWEEP_MAX_RUNS"
    echo "ポーリング間隔: ${SWEEP_POLL_INTERVAL}秒"
    echo "=========================================="
    echo ""

    local completed_runs=0
    local failed_runs=0

    while [[ $completed_runs -lt $SWEEP_MAX_RUNS ]]; do
        echo ""
        echo "--- Run $((completed_runs + 1))/$SWEEP_MAX_RUNS ---"

        # 次のパラメータを取得
        echo "W&B Sweepから次のパラメータを取得中..."
        local run_info
        run_info=$(init_sweep_run "$sweep_id" 2>&1)

        if [[ "$run_info" == "SWEEP_DONE" ]]; then
            echo "Sweepが完了しました"
            break
        elif [[ "$run_info" == "NO_PARAMS" ]] || [[ "$run_info" == "SWEEP_ERROR" ]]; then
            echo "パラメータを取得できませんでした。Sweepが完了した可能性があります。"
            break
        fi

        echo "取得したパラメータ:"
        echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['params'], indent=2))"

        local run_id
        run_id=$(echo "$run_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['run_id'])")

        # ジョブスクリプト生成
        local job_script="${SWEEP_STATE_DIR}/job_${run_id}.sh"
        generate_sweep_job_script "$run_info" "$SWEEP_BACKEND" "$job_script"

        # ジョブ投下と待機
        if submit_sweep_job_and_wait "$job_script" "$run_id"; then
            ((completed_runs++))
            echo "Run $completed_runs/$SWEEP_MAX_RUNS 完了"
        else
            ((failed_runs++))
            echo "Run失敗 (失敗数: $failed_runs)"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Sweep完了"
    echo "=========================================="
    echo "完了: $completed_runs"
    echo "失敗: $failed_runs"
    echo "Sweep ID: $sweep_id"
    echo ""
    echo "結果を確認: https://wandb.ai/${WANDB_ENTITY:-your-entity}/${WANDB_PROJECT}/sweeps/$sweep_id"
}

# ----------------------------------------------------------------------------
# ヘルプ表示
# ----------------------------------------------------------------------------
show_help() {
    cat << EOF
使い方: $(basename "$0") [コマンド] [オプション]

コマンド:
  <job_script.sh>           Slurmジョブスクリプトを投下
  sweep <config.yaml>       W&B Sweepを開始
  --status                  キュー状態を確認
  --cancel <job_id>         ジョブをキャンセル
  --help, -h                このヘルプを表示

ジョブ投下オプション:
  --dry-run                 ドライラン（実際には投下しない）

Sweepオプション:
  --backend <backend>       VLAバックエンド (openvla, openpi) [デフォルト: openvla]
  --max-runs <N>            最大実行回数 [デフォルト: 10]
  --poll-interval <秒>      ジョブ状態確認間隔 [デフォルト: 300]
  --resume <sweep_id>       既存のSweepを再開
  --dry-run                 ドライラン

例:
  # 通常のジョブ投下
  ./submit.sh jobs/train_openvla.sh
  ./submit.sh jobs/train_openvla.sh --dry-run

  # W&B Sweep
  ./submit.sh sweep sweeps/sweep_openvla.yaml --max-runs 20
  ./submit.sh sweep sweeps/sweep_openpi.yaml --backend openpi
  ./submit.sh sweep --resume abc123xyz --max-runs 10

  # キュー管理
  ./submit.sh --status
  ./submit.sh --cancel 12345

設定:
  .env ファイルにSSH接続情報とSlurm/W&B設定を記述してください。
  テンプレート: .env.template
EOF
}

# ----------------------------------------------------------------------------
# メイン処理
# ----------------------------------------------------------------------------
main() {
    load_config

    case "${1:-}" in
        --help|-h)
            show_help
            ;;
        --status)
            check_status
            ;;
        --cancel)
            if [[ -z "${2:-}" ]]; then
                echo "エラー: ジョブIDを指定してください"
                echo "使い方: ./submit.sh --cancel <job_id>"
                exit 1
            fi
            cancel_job "$2"
            ;;
        sweep)
            shift
            local sweep_config=""

            # Sweep引数解析
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --backend)
                        SWEEP_BACKEND="$2"
                        shift 2
                        ;;
                    --max-runs)
                        SWEEP_MAX_RUNS="$2"
                        shift 2
                        ;;
                    --poll-interval)
                        SWEEP_POLL_INTERVAL="$2"
                        shift 2
                        ;;
                    --dry-run)
                        SWEEP_DRY_RUN=true
                        shift
                        ;;
                    --resume)
                        SWEEP_RESUME_ID="$2"
                        shift 2
                        ;;
                    -*)
                        echo "不明なオプション: $1"
                        show_help
                        exit 1
                        ;;
                    *)
                        sweep_config="$1"
                        shift
                        ;;
                esac
            done

            # 入力チェック
            if [[ -z "$sweep_config" && -z "$SWEEP_RESUME_ID" ]]; then
                echo "エラー: sweep設定ファイルまたは--resumeオプションが必要です"
                show_help
                exit 1
            fi

            if [[ -n "$sweep_config" && ! -f "$sweep_config" ]]; then
                echo "エラー: 設定ファイルが見つかりません: $sweep_config"
                exit 1
            fi

            run_sweep "$sweep_config"
            ;;
        "")
            show_help
            ;;
        *)
            local job_script="$1"
            local dry_run="false"
            if [[ "${2:-}" == "--dry-run" ]]; then
                dry_run="true"
            fi
            submit_job "$job_script" "$dry_run"
            ;;
    esac
}

main "$@"
