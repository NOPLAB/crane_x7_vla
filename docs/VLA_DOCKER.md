# VLAトレーニング用Docker環境

このディレクトリには、CRANE-X7データセットでOpenVLAモデルをトレーニングするためのDocker設定が含まれています。

## 前提条件

- Docker (>= 20.10)
- Docker Compose (>= 2.0)
- CUDA対応のNVIDIA GPU
- NVIDIA Container Toolkit ([インストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### GPUアクセスの確認

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## クイックスタート

### 1. 初期セットアップ

環境変数テンプレートをコピーし、APIキーを設定:

```bash
cd vla
cp .env.template .env
# お好みのエディタで .env を編集
nano .env  # または vim .env
```

必須の設定:

- `WANDB_API_KEY`: Weights & Biases APIキー(オプション、実験トラッキング用)
- `HF_TOKEN`: HuggingFaceトークン(必須、事前学習済みモデルのダウンロードに必要)
- `DATA_DIR`: TFRecordデータセットへのパス(デフォルト: `../data/tfrecord_logs`)

### 2. Dockerイメージのビルド

トレーニング用イメージのビルド:

```bash
./scripts/build.sh
```

または開発用イメージのビルド(Jupyter、デバッグツールを含む):

```bash
./scripts/build.sh dev
```

### 3. トレーニングの実行

#### オプションA: インタラクティブなトレーニングセッション

インタラクティブコンテナの起動:

```bash
./scripts/run.sh
```

コンテナ内でトレーニングを実行:

```bash
# ラッパースクリプトを使用
./scripts/train.sh

# またはカスタムパラメータでPythonを直接実行
python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --data_dir /workspace/data \
    --output_dir /workspace/checkpoints \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 2e-5
```

#### オプションB: 直接トレーニング(1コマンド)

```bash
docker compose run --rm vla-train python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --data_dir /workspace/data \
    --output_dir /workspace/checkpoints
```

### 4. 開発モード

JupyterとTensorBoardを使用したインタラクティブ開発:

```bash
# 開発用コンテナを起動
./scripts/run.sh dev

# コンテナにアクセス
docker exec -it crane_x7_vla_dev bash

# Jupyter Labを起動(コンテナ内)
jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# またはTensorBoardを起動(コンテナ内)
tensorboard --logdir=/workspace/logs --host=0.0.0.0
```

アクセスポイント:

- Jupyter Lab: <http://localhost:8888>
- TensorBoard: <http://localhost:6006>

## ディレクトリ構造

```text
vla/
├── Dockerfile              # マルチステージビルド(base + dev)
├── docker-compose.yml      # コンテナオーケストレーション
├── .env.template           # 環境変数テンプレート
├── .env                    # ローカル設定(git-ignored)
├── scripts/
│   ├── build.sh           # Dockerイメージのビルド
│   ├── run.sh             # コンテナの実行
│   └── train.sh           # トレーニングラッパースクリプト
├── data/                  # マウントされたデータセットディレクトリ
├── checkpoints/           # 保存されたモデルチェックポイント
└── logs/                  # トレーニングログ
```

## 環境変数

すべての環境変数は`.env`ファイルで設定できます:

| 変数名 | 説明 | デフォルト値 |
|----------|-------------|---------|
| `WANDB_API_KEY` | Weights & Biases APIキー | (空) |
| `HF_TOKEN` | HuggingFace APIトークン | (空) |
| `DATA_DIR` | データセットディレクトリパス | `../data/tfrecord_logs` |
| `CHECKPOINT_DIR` | モデルチェックポイントディレクトリ | `./checkpoints` |
| `LOG_DIR` | トレーニングログディレクトリ | `./logs` |
| `HF_CACHE` | HuggingFaceキャッシュディレクトリ | `~/.cache/huggingface` |
| `JUPYTER_PORT` | Jupyterポート(開発モード) | `8888` |
| `TENSORBOARD_PORT` | TensorBoardポート(開発モード) | `6006` |

## トレーニングパラメータ

コマンドライン引数または環境変数でトレーニングを設定:

```bash
# 環境変数経由
export MODEL_NAME="openvla/openvla-7b"
export DATASET_NAME="crane_x7"
export BATCH_SIZE=8
export NUM_EPOCHS=10
export LEARNING_RATE=2e-5
./scripts/train.sh

# コマンドライン引数経由
python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --use_lora \
    --gradient_checkpointing
```

## 高度な使用方法

### マルチGPUトレーニング

```bash
# 利用可能な全GPUを自動使用
docker compose run --rm vla-train python3 finetune.py \
    --model_name openvla/openvla-7b \
    --multi_gpu
```

### カスタムデータセットパス

```bash
# .envのDATA_DIRを変更するか、カスタムマウントを使用
docker compose run --rm \
    -v /path/to/my/data:/workspace/data \
    vla-train python3 finetune.py --data_dir /workspace/data
```

### チェックポイントから再開

```bash
python3 finetune.py \
    --resume_from_checkpoint /workspace/checkpoints/checkpoint-1000
```

## トラブルシューティング

### GPUが検出されない

```bash
# NVIDIAランタイムを確認
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Dockerデーモン設定を確認(/etc/docker/daemon.json)
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

### メモリ不足

バッチサイズを減らすか、勾配チェックポインティングを有効化:

```bash
python3 finetune.py --batch_size 4 --gradient_checkpointing
```

### HuggingFace認証失敗

`.env`に`HF_TOKEN`が設定されており、モデルへのアクセス権があることを確認:

```bash
# コンテナ内で手動ログイン
huggingface-cli login
```

### パーミッションの問題

マウントされたボリュームでパーミッション問題が発生した場合:

```bash
# 現在のユーザーで実行(docker-compose.ymlに追加)
user: "${UID}:${GID}"
```

## トレーニングの監視

### Weights & Biases

`WANDB_API_KEY`が設定されている場合、トレーニングは自動的にW&Bにログを記録:

```bash
# https://wandb.ai/<your-username>/<project-name> で実行を確認
```

### TensorBoard

```bash
# コンテナ内でTensorBoardを起動
tensorboard --logdir=/workspace/logs --host=0.0.0.0

# <http://localhost:6006> でアクセス
```

### GPU使用率

```bash
# コンテナ内
watch -n 1 nvidia-smi

# またはホストから
docker exec crane_x7_vla_train watch -n 1 nvidia-smi
```

## クリーンアップ

```bash
# すべてのコンテナを停止
docker compose down

# ボリュームを削除
docker compose down -v

# イメージを削除
docker rmi crane_x7_vla:latest crane_x7_vla:dev
```

## 参考資料

- [OpenVLAドキュメント](https://github.com/openvla/openvla)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPU サポート](https://docs.docker.com/compose/gpu-support/)
