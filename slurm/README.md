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
# API Keys
WANDB_API_KEY=your-wandb-api-key  # W&B実験トラッキング用

# SSH接続設定（必須）
SLURM_SSH_HOST=your-cluster.example.com
SLURM_SSH_USER=your-username

# SSH認証方式（password または key）
SLURM_SSH_AUTH=password  # パスワード認証の場合
# SLURM_SSH_AUTH=key     # 公開鍵認証の場合
# SLURM_SSH_KEY=~/.ssh/id_rsa

# Slurm設定（オプション）
SLURM_SSH_PORT=22
SLURM_REMOTE_WORKDIR=~/crane_x7_vla
SLURM_PARTITION=gpu
SLURM_GPUS=1
SLURM_TIME=24:00:00
SLURM_MEM=32G
SLURM_CPUS=8
```

3. `example_jobs/`からジョブスクリプトをコピーして環境に合わせてカスタマイズ:

```bash
mkdir -p jobs
cp example_jobs/train_openvla.sh jobs/
# jobs/train_openvla.sh を環境に合わせて編集
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
├── .env.template     # 環境変数テンプレート
├── .env              # 環境変数（gitignore対象）
├── .gitignore        # Git除外設定
├── submit.sh         # ジョブ投下スクリプト
├── README.md         # このファイル
├── example_jobs/     # サンプルジョブスクリプト（テンプレート）
│   ├── train_openvla.sh
│   └── train_openpi.sh
└── jobs/             # 実際に使用するジョブスクリプト（gitignore対象）
    ├── train_openvla.sh
    └── train_openpi.sh
```

**注意**: `jobs/`ディレクトリは`.gitignore`に含まれています。環境固有の設定（パス、GPU数など）を含むため、`example_jobs/`からコピーして各自でカスタマイズしてください。

## ジョブスクリプトのカスタマイズ

`example_jobs/`ディレクトリ内のサンプルを参考に、`jobs/`ディレクトリにカスタマイズしたスクリプトを作成してください。

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

# コンテナを使用する場合（Pyxis/Enroot）
#SBATCH --container=noppdev/vla      # Dockerイメージ
```

### Singularityコンテナの使用

リモートサーバーでSingularityコンテナを使用する場合は、`example_jobs/`内のコメントアウトされたセクションを参考にしてください。

### Pyxis/Enrootコンテナの使用

Pyxis/Enrootプラグインを使用する環境では、`#SBATCH --container=`オプションでDockerイメージを直接指定できます。`jobs/train_openvla.sh`を参照してください。
