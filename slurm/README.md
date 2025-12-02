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

---

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

1. `.env`ファイルにSweep用の設定を追加:

```bash
# W&B Sweep設定
WANDB_ENTITY=your-team-or-username  # W&Bエンティティ
WANDB_PROJECT=crane_x7_sweep        # プロジェクト名

# トレーニング設定
DATA_ROOT=/root/vla/data            # データディレクトリ（リモート）
OUTPUT_DIR=/root/vla/output         # 出力ディレクトリ（リモート）
NUM_EPOCHS=10                       # エポック数
SAVE_INTERVAL=500                   # チェックポイント保存間隔
EVAL_INTERVAL=100                   # 評価間隔
```

2. 必要に応じてSweep設定ファイルをカスタマイズ:

```bash
# sweeps/sweep_openvla.yaml  - OpenVLA用
# sweeps/sweep_openpi.yaml   - OpenPI用
```

### Sweepの実行

```bash
# OpenVLAでSweepを開始（20回実行）
./submit.sh sweep sweeps/sweep_openvla.yaml --max-runs 20

# OpenPIでSweepを開始
./submit.sh sweep sweeps/sweep_openpi.yaml --backend openpi --max-runs 15

# ポーリング間隔を変更（デフォルト: 300秒）
./submit.sh sweep sweeps/sweep_openvla.yaml --poll-interval 600

# ドライラン（テスト用、実際にはジョブを投下しない）
./submit.sh sweep sweeps/sweep_openvla.yaml --dry-run
```

### 既存Sweepの再開

```bash
# Sweep IDを指定して再開
./submit.sh sweep --resume abc123xyz --max-runs 10

# バックエンドを指定して再開
./submit.sh sweep --resume abc123xyz --backend openpi --max-runs 5
```

### Sweep設定ファイルの構造

`sweeps/sweep_openvla.yaml`の例:

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

  lora_rank:
    values: [8, 16, 32, 64]

# 早期終了（性能の悪いrunを早期に終了）
early_terminate:
  type: hyperband
  min_iter: 3
```

### 探索可能なハイパーパラメータ

#### OpenVLA

| パラメータ | 説明 | 推奨範囲 |
|-----------|------|---------|
| `learning_rate` | 学習率 | 1e-6 ~ 1e-3 |
| `batch_size` | バッチサイズ | 1, 2, 4, 8, 16 |
| `lora_rank` | LoRAランク | 8, 16, 32, 64 |
| `lora_dropout` | LoRAドロップアウト | 0.0 ~ 0.2 |
| `weight_decay` | 重み減衰 | 1e-5 ~ 1e-1 |
| `warmup_steps` | ウォームアップステップ | 100 ~ 2000 |
| `max_grad_norm` | 勾配クリッピング | 0.5 ~ 2.0 |

#### OpenPI

上記に加えて:

| パラメータ | 説明 | 推奨範囲 |
|-----------|------|---------|
| `model_type` | モデルタイプ | pi0, pi0_fast |
| `action_horizon` | アクションホライズン | 10, 25, 50, 100 |
| `normalization_mode` | 正規化モード | zscore, quantile |

### 結果の確認

Sweepの結果はW&Bダッシュボードで確認できます:

```
https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>
```

### ディレクトリ構成（Sweep関連）

```
slurm/
├── submit.sh             # ジョブ投下 & Sweepコントローラー
├── sweeps/               # Sweep設定ファイル
│   ├── sweep_openvla.yaml
│   └── sweep_openpi.yaml
└── .sweep_state/         # Sweep状態ファイル（自動生成、gitignore対象）
    ├── current_sweep_id
    └── job_*.sh
```

### トラブルシューティング

#### W&B APIキーエラー

```
エラー: WANDB_API_KEY が設定されていません
```

→ `.env`ファイルに`WANDB_API_KEY`を設定してください。キーは https://wandb.ai/settings で取得できます。

#### Sweepが見つからない

```
ERROR:Sweep not found
```

→ `WANDB_ENTITY`と`WANDB_PROJECT`が正しく設定されているか確認してください。

#### ジョブが失敗する

ジョブの出力ログを確認:

```bash
# リモートサーバーでログを確認
ssh user@cluster "cat ~/vla/logs/sweep_*_<job_id>.out"
```
