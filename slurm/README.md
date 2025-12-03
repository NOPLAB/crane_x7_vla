# slurm-submit

SSH経由でSlurmクラスターにジョブを投下するPythonツールです。

## 特徴

- SSH経由でリモートSlurmクラスターにジョブを投下
- パスワード認証・公開鍵認証の両方をサポート
- W&B Sweepによるハイパーパラメータ自動探索
- pydanticによる設定バリデーション
- richによる見やすいCLI出力

## インストール

```bash
cd slurm
pip install -e .
```

## セットアップ

1. `.env`ファイルを作成:

```bash
cp .env.template .env
```

2. `.env`を編集してSSH接続情報を設定:

```bash
# API Keys
WANDB_API_KEY=your-wandb-api-key  # W&B実験トラッキング用（オプション）

# SSH接続設定（必須）
SLURM_SSH_HOST=your-cluster.example.com
SLURM_SSH_USER=your-username

# SSH認証方式（password または key）
SLURM_SSH_AUTH=password  # パスワード認証の場合
# SLURM_SSH_AUTH=key     # 公開鍵認証の場合
# SLURM_SSH_KEY=~/.ssh/id_rsa

# Slurm設定（オプション）
SLURM_SSH_PORT=22
SLURM_REMOTE_WORKDIR=~/workdir
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
# ジョブスクリプトを投下
slurm-submit submit jobs/train.sh

# ドライラン（実際には投下しない）
slurm-submit submit jobs/train.sh --dry-run

# 別の.envファイルを使用
slurm-submit submit jobs/train.sh --env /path/to/.env
```

### キュー状態の確認

```bash
# 自分のジョブを確認
slurm-submit status

# 特定のジョブを確認
slurm-submit status 12345

# 全ユーザーのジョブを確認
slurm-submit status --all
```

### ジョブのキャンセル

```bash
slurm-submit cancel 12345
```

### ジョブ完了待機

```bash
# ジョブ完了まで待機
slurm-submit wait 12345

# ポーリング間隔を指定（秒）
slurm-submit wait 12345 --interval 120

# タイムアウトを指定（秒）
slurm-submit wait 12345 --timeout 3600
```

## W&B Sweep（ハイパーパラメータ自動探索）

Weights & Biases Sweepsを使用して、ハイパーパラメータの自動探索を行うことができます。

### 動作原理

1. ローカルマシンでW&B Sweepを作成・制御
2. W&B APIから次のハイパーパラメータを取得
3. パラメータを埋め込んだSlurmジョブスクリプトを生成
4. SSHでリモートクラスターにジョブを投下
5. ジョブ完了を定期的にポーリングして待機
6. 指定回数分繰り返し

### セットアップ

`.env`ファイルにSweep用の設定を追加:

```bash
# W&B Sweep設定
WANDB_ENTITY=your-team-or-username  # W&Bエンティティ
WANDB_PROJECT=my_sweep              # プロジェクト名

# トレーニング設定
DATA_ROOT=/path/to/data
OUTPUT_DIR=/path/to/output
NUM_EPOCHS=10
SAVE_INTERVAL=500
EVAL_INTERVAL=100
```

### Sweepの実行

```bash
# Sweepを開始（10回実行）
slurm-submit sweep start sweeps/config.yaml --max-runs 10

# ポーリング間隔を変更（デフォルト: 300秒）
slurm-submit sweep start sweeps/config.yaml --poll-interval 600

# カスタムジョブテンプレートを使用
slurm-submit sweep start sweeps/config.yaml --template jobs/sweep_template.sh

# ドライラン
slurm-submit sweep start sweeps/config.yaml --dry-run
```

### 既存Sweepの再開

```bash
# Sweep IDを指定して再開
slurm-submit sweep resume abc123xyz --max-runs 10
```

### Sweepの状態確認

```bash
slurm-submit sweep status abc123xyz
```

### Sweep設定ファイルの構造

`sweeps/config.yaml`の例:

```yaml
# 探索方法: bayes（ベイズ最適化）, grid（グリッド探索）, random（ランダム探索）
method: bayes

# 最適化するメトリック
metric:
  name: eval/loss
  goal: minimize

# 探索するハイパーパラメータ
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3

  batch_size:
    values: [1, 2, 4, 8, 16]

# 早期終了（性能の悪いrunを早期に終了）
early_terminate:
  type: hyperband
  min_iter: 3
```

### カスタムジョブテンプレート

`--template`オプションでカスタムジョブテンプレートを指定できます。テンプレート内では以下のプレースホルダが使用可能:

- `{{RUN_ID}}`: W&B Run ID
- `{{PARAMS_JSON}}`: パラメータのJSON文字列
- `{{learning_rate}}`, `{{batch_size}}`, など: 個別パラメータ

テンプレート例（`jobs/sweep_template.sh`）:

```bash
#!/bin/bash
#SBATCH --job-name=sweep_{{RUN_ID}}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

export WANDB_RUN_ID={{RUN_ID}}

python train.py \
    --learning-rate {{learning_rate}} \
    --batch-size {{batch_size}}
```

## ディレクトリ構成

```
slurm/
├── src/
│   └── slurm_submit/     # Pythonパッケージ
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py        # CLIエントリーポイント
│       ├── config.py     # 設定管理
│       ├── ssh_client.py # SSH/SCP操作
│       ├── slurm_client.py # Slurmコマンド
│       ├── job_script.py # ジョブスクリプト生成
│       └── sweep/        # W&B Sweep統合
├── pyproject.toml
├── .env.template         # 環境変数テンプレート
├── .env                  # 環境変数（gitignore対象）
├── .gitignore
├── README.md
├── example_jobs/         # サンプルジョブスクリプト
│   ├── train_openvla.sh
│   └── train_openpi.sh
├── jobs/                 # 実際に使用するジョブスクリプト（gitignore対象）
└── sweeps/               # Sweep設定ファイル
    ├── sweep_openvla.yaml
    └── sweep_openpi.yaml
```

**注意**: `jobs/`ディレクトリは`.gitignore`に含まれています。環境固有の設定を含むため、`example_jobs/`からコピーして各自でカスタマイズしてください。

## ジョブスクリプトのカスタマイズ

### SBATCH オプション

```bash
#SBATCH --job-name=my_job       # ジョブ名
#SBATCH --partition=gpu         # パーティション
#SBATCH --nodes=1               # ノード数
#SBATCH --cpus-per-task=8       # CPU数
#SBATCH --mem=64G               # メモリ
#SBATCH --gres=gpu:1            # GPU数
#SBATCH --time=48:00:00         # 実行時間

# コンテナを使用する場合（Pyxis/Enroot）
#SBATCH --container-image=myimage:latest
```

## プログラムからの使用

```python
from slurm_submit import Settings, SSHClient, SlurmClient
from slurm_submit.config import load_settings

# 設定を読み込み
settings = load_settings(".env")

# SSHクライアントを作成して接続
with SSHClient(settings.ssh) as ssh:
    ssh.connect()

    # Slurmクライアントを作成
    slurm = SlurmClient(ssh, settings.slurm)

    # ジョブを投下
    job_id = slurm.submit(Path("jobs/train.sh"))
    print(f"Job ID: {job_id}")

    # 状態を確認
    jobs = slurm.status()
    for job in jobs:
        print(f"{job.job_id}: {job.state}")
```

## トラブルシューティング

### SSH接続エラー

```
SSH接続に失敗しました: Authentication failed
```

→ `.env`ファイルの`SLURM_SSH_USER`、`SLURM_SSH_AUTH`、`SLURM_SSH_KEY`を確認してください。

### W&B APIキーエラー

```
WANDB_API_KEY が設定されていません
```

→ `.env`ファイルに`WANDB_API_KEY`を設定してください。キーは https://wandb.ai/settings で取得できます。

### Sweepが見つからない

```
Sweep状態の取得に失敗
```

→ `WANDB_ENTITY`と`WANDB_PROJECT`が正しく設定されているか確認してください。

## ライセンス

MIT License
