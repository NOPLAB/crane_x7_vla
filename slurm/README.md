# Slurm ジョブ投下ツール

SSHサーバー経由でSlurmクラスターにジョブを投下するためのツールです。

## セットアップ

1. `.env`ファイルを作成:

```bash
cd slurm
cp .env.template .env
```

2. `.env`を編集してSSH接続情報を設定:

```bash
# 必須設定
SLURM_SSH_HOST=your-cluster.example.com
SLURM_SSH_USER=your-username

# オプション設定
SLURM_SSH_PORT=22
SLURM_SSH_KEY=~/.ssh/id_rsa
SLURM_REMOTE_WORKDIR=~/crane_x7_vla
SLURM_PARTITION=gpu
```

## 使い方

### ジョブの投下

```bash
# OpenVLAトレーニングジョブを投下
./submit.sh jobs/train_openvla.sh

# OpenPIトレーニングジョブを投下
./submit.sh jobs/train_openpi.sh

# ドライラン（実際には投下しない）
./submit.sh jobs/train_openvla.sh --dry-run
```

### キュー状態の確認

```bash
./submit.sh --status
```

### ジョブのキャンセル

```bash
./submit.sh --cancel <job_id>
```

## ディレクトリ構成

```
slurm/
├── .env.template    # 環境変数テンプレート
├── .env             # 環境変数（gitignore対象）
├── submit.sh        # ジョブ投下スクリプト
├── README.md        # このファイル
└── jobs/            # ジョブスクリプト
    ├── train_openvla.sh
    └── train_openpi.sh
```

## ジョブスクリプトのカスタマイズ

`jobs/`ディレクトリ内のジョブスクリプトを編集して、必要に応じてカスタマイズしてください。

### SBATCH オプション

```bash
#SBATCH --job-name=crane_x7_openvla  # ジョブ名
#SBATCH --partition=gpu              # パーティション
#SBATCH --nodes=1                    # ノード数
#SBATCH --ntasks=1                   # タスク数
#SBATCH --cpus-per-task=8            # CPU数
#SBATCH --mem=64G                    # メモリ
#SBATCH --gres=gpu:1                 # GPU数
#SBATCH --time=48:00:00              # 実行時間
```

### Singularityコンテナの使用

リモートサーバーでSingularityコンテナを使用する場合は、ジョブスクリプト内のコメントアウトされたセクションを参考にしてください。
