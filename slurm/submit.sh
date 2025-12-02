#!/bin/bash
# ============================================================================
# Slurm Job Submission Script
# ============================================================================
# SSHサーバーに接続し、Slurmジョブを投下するスクリプト
#
# 使い方:
#   ./submit.sh <job_script.sh>           # ジョブスクリプトを投下
#   ./submit.sh <job_script.sh> --dry-run # ドライラン（実際には投下しない）
#   ./submit.sh --status                  # キュー状態を確認
#   ./submit.sh --cancel <job_id>         # ジョブをキャンセル
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

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

    # .envファイルを読み込み（コメント行と空行を除外）
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
    SLURM_SSH_AUTH="${SLURM_SSH_AUTH:-password}"  # password or key
    SLURM_REMOTE_WORKDIR="${SLURM_REMOTE_WORKDIR:-~/crane_x7_vla}"
    SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
    SLURM_GPUS="${SLURM_GPUS:-1}"
    SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-}"
    SLURM_TIME="${SLURM_TIME:-24:00:00}"
    SLURM_MEM="${SLURM_MEM:-32G}"
    SLURM_CPUS="${SLURM_CPUS:-8}"
    SLURM_JOB_PREFIX="${SLURM_JOB_PREFIX:-crane_x7}"
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

# ----------------------------------------------------------------------------
# ヘルプ表示
# ----------------------------------------------------------------------------
show_help() {
    cat << EOF
使い方: $(basename "$0") [オプション] [ジョブスクリプト]

オプション:
  <job_script.sh>       Slurmジョブスクリプトを投下
  --dry-run             ドライラン（実際には投下しない）
  --status              キュー状態を確認
  --cancel <job_id>     ジョブをキャンセル
  --help, -h            このヘルプを表示

例:
  ./submit.sh jobs/train_openvla.sh           # ジョブを投下
  ./submit.sh jobs/train_openvla.sh --dry-run # ドライラン
  ./submit.sh --status                        # キュー状態を確認
  ./submit.sh --cancel 12345                  # ジョブをキャンセル

設定:
  .env ファイルにSSH接続情報とSlurm設定を記述してください。
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
