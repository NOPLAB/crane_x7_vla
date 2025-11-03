# VLAトレーニング用Docker環境

このガイドでは、CRANE-X7データセットでOpenVLAモデルをトレーニングするためのDocker環境のセットアップと使用方法を説明します。

## 概要

このプロジェクトでは、VLAトレーニング用に2つのDockerセットアップ方法を提供しています：

- **方法A（推奨）**: プロジェクトルートの統合docker-composeを使用
  - ROS2とVLAを同じ環境で管理
  - データ収集からトレーニングまでのワークフローが統一
  - `ros2/docker-compose.yml`の`vla`プロファイルを使用

- **方法B**: vla/ディレクトリの独立したセットアップを使用
  - VLAトレーニングのみに特化
  - 独立した環境設定が可能
  - `vla/docker-compose.yml`を使用

## 前提条件

### ハードウェア要件

**最小要件：**
- GPU: 16GB VRAMのNVIDIA GPU（例: V100、RTX 4090）
- RAM: 32GBのシステムRAM
- ストレージ: 50GBの空き容量

**推奨要件：**
- GPU: NVIDIA A100 40GB/80GB
- RAM: 64GB以上のシステムRAM
- ストレージ: 100GB以上の空き容量（データセットとチェックポイント用）

**マルチGPUスケーリング：**
- 1x A100 (40GB): バッチサイズ ~8-12（LoRA使用時）
- 2x A100 (40GB): バッチサイズ ~16-24（LoRA使用時）
- 4x A100 (40GB): バッチサイズ ~32-48（LoRA使用時）

### ソフトウェア要件

- Docker (>= 20.10)
- Docker Compose (>= 2.0)
- CUDA対応のNVIDIA GPU
- NVIDIA Container Toolkit ([インストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### GPUアクセスの確認

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 方法A: プロジェクトルートのdocker-composeを使用（推奨）

この方法は、ROS2とVLAを統合環境で管理する場合に推奨されます。

### 1. 初期セットアップ

環境変数の設定は不要です。docker-compose.ymlがデフォルト値を使用します。

### 2. Dockerイメージのビルド

```bash
# プロジェクトルートから実行
docker compose -f ros2/docker-compose.yml build vla_finetune
```

これにより、以下を含むDockerイメージが作成されます：
- CUDA 12.4
- CUDA対応のPyTorch 2.2.0
- OpenVLAの依存関係
- Flash Attention 2（ビルドが成功した場合）

### 3. トレーニングの実行

#### インタラクティブセッション

```bash
# インタラクティブコンテナを起動
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune

# コンテナ内でデータセット読み込みをテスト
python3 vla/crane_x7_dataset.py data/tfrecord_logs

# ファインチューニングを実行
cd vla
python3 finetune.py
```

#### ヘルパースクリプトの使用

```bash
# データセット読み込みのテスト
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh test-dataset

# シングルGPUトレーニング
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train

# マルチGPUトレーニング（2 GPU）
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train-multi-gpu 2
```

#### カスタムトレーニングパラメータ

```bash
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune bash -c "
  cd vla && python3 finetune.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --lora_rank 64 \
    --use_wandb \
    --wandb_project my-crane-x7
"
```

### 4. コンテナ内のディレクトリ構造

```
/workspace/
├── vla/                    # ファインチューニングスクリプト（マウント）
│   ├── finetune.py
│   ├── crane_x7_dataset.py
│   ├── finetune_config.py
│   └── README.md
├── data/                   # トレーニングデータ（マウント）
│   └── tfrecord_logs/
│       ├── episode_0000_*/
│       ├── episode_0001_*/
│       └── ...
└── outputs/                # チェックポイントとログ（マウント）
    └── crane_x7_finetune/
```

## 方法B: vla/ディレクトリの独立したセットアップを使用

VLAトレーニングのみを独立して実行する場合に使用します。

### 1. 初期セットアップ

環境変数テンプレートをコピーし、APIキーを設定：

```bash
cd vla
cp .env.template .env
# お好みのエディタで .env を編集
nano .env  # または vim .env
```

必須の設定：
- `WANDB_API_KEY`: Weights & Biases APIキー（オプション、実験トラッキング用）
- `HF_TOKEN`: HuggingFaceトークン（必須、事前学習済みモデルのダウンロードに必要）
- `DATA_DIR`: TFRecordデータセットへのパス（デフォルト: `../data/tfrecord_logs`）

### 2. Dockerイメージのビルド

```bash
cd vla
./scripts/build.sh          # トレーニング用イメージ
# または
./scripts/build.sh dev      # 開発用イメージ（Jupyter、デバッグツール含む）
```

### 3. トレーニングの実行

#### インタラクティブセッション

```bash
# インタラクティブコンテナの起動
./scripts/run.sh

# コンテナ内でトレーニングを実行
./scripts/train.sh

# またはカスタムパラメータで直接実行
python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --data_dir /workspace/data \
    --output_dir /workspace/checkpoints \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 2e-5
```

#### ワンコマンドでトレーニング

```bash
docker compose run --rm vla-train python3 finetune.py \
    --model_name openvla/openvla-7b \
    --dataset_name crane_x7 \
    --data_dir /workspace/data \
    --output_dir /workspace/checkpoints
```

### 4. 開発モード

JupyterとTensorBoardを使用したインタラクティブ開発：

```bash
# 開発用コンテナを起動
./scripts/run.sh dev

# コンテナにアクセス
docker exec -it crane_x7_vla_dev bash

# Jupyter Labを起動（コンテナ内）
jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# またはTensorBoardを起動（コンテナ内）
tensorboard --logdir=/workspace/logs --host=0.0.0.0
```

アクセスポイント：
- Jupyter Lab: http://localhost:8888
- TensorBoard: http://localhost:6006

## GPU設定

### GPU数の指定

docker-compose.ymlは利用可能な全GPUを使用するように設定されています：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all          # 全GPUを使用
          capabilities: [gpu]
```

特定のGPUを使用するには、`count`を変更：
- `count: 1` - 1つのGPUを使用
- `count: 2` - 2つのGPUを使用
- `count: all` - 利用可能な全GPUを使用

または`CUDA_VISIBLE_DEVICES`環境変数を使用：

```bash
docker compose --profile vla run --rm \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  vla_finetune
```

### マルチGPUトレーニング

```bash
# 方法A（プロジェクトルート）
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune bash -c "
  cd vla && torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py \
    --batch_size 8 \
    --learning_rate 5e-4
"

# 方法B（vla/ディレクトリ）
docker compose run --rm vla-train python3 finetune.py \
    --model_name openvla/openvla-7b \
    --multi_gpu
```

## トレーニングの監視

### Weights & Biasesの使用

```bash
# W&Bにログイン（初回セットアップのみ）
docker compose --profile vla run --rm vla_finetune bash -c "
  pip3 install wandb && wandb login
"

# W&Bロギングを有効にしてトレーニングを実行
docker compose --profile vla run --rm vla_finetune bash -c "
  cd vla && python3 finetune.py \
    --use_wandb \
    --wandb_project crane-x7-openvla \
    --wandb_entity YOUR_USERNAME
"
```

### TensorBoard

```bash
# コンテナ内でTensorBoardを起動
tensorboard --logdir=/workspace/logs --host=0.0.0.0

# http://localhost:6006 でアクセス
```

### GPU使用率の確認

```bash
# コンテナ内
watch -n 1 nvidia-smi

# またはホストから
docker exec crane_x7_vla_train watch -n 1 nvidia-smi
```

### ログの表示

```bash
# ログをファイルに保存
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train 2>&1 | tee training.log
```

## トラブルシューティング

### GPUが検出されない

```bash
# NVIDIAランタイムを確認
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Dockerデーモン設定を確認（/etc/docker/daemon.json）
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

### メモリ不足（OOM）

**解決策1: バッチサイズを減らす**

```bash
# finetune_config.pyまたはCLIで設定
python3 finetune.py --batch_size 4
```

**解決策2: 勾配チェックポインティングを有効化**

```python
# finetune_config.py
gradient_checkpointing: bool = True
```

**解決策3: LoRAランクを減らす**

```bash
python3 finetune.py --lora_rank 16
```

**解決策4: 共有メモリを増やす**

docker-compose.ymlで共有メモリを調整：

```yaml
shm_size: '32gb'  # デフォルトは16gb
```

### Flash Attentionのビルド失敗

Flash Attention 2はオプションです。ビルドが失敗してもコンテナは継続して動作します。トレーニングは少し遅くなりますが、機能は問題ありません。

コンテナ内で手動インストールする場合：

```bash
docker compose --profile vla run --rm vla_finetune bash
# コンテナ内で：
pip3 install packaging ninja
pip3 install flash-attn==2.5.5 --no-build-isolation
```

### CUDAバージョンの不一致

DockerfileはCUDA 12.4を使用しています。ホストに異なるCUDAバージョンがある場合は、Dockerfileのベースイメージを変更してください：

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS vla  # CUDAバージョンを変更
```

その後、再ビルド：

```bash
docker compose build vla_finetune --no-cache
```

### HuggingFace認証失敗

`.env`に`HF_TOKEN`が設定されており、モデルへのアクセス権があることを確認：

```bash
# コンテナ内で手動ログイン
huggingface-cli login
```

### パーミッションの問題

マウントされたボリュームでパーミッションエラーが発生した場合：

```bash
# 現在のユーザーとしてコンテナを実行
docker compose --profile vla run --rm --user $(id -u):$(id -g) vla_finetune

# またはdocker-compose.ymlに追加
user: "${UID}:${GID}"
```

## 開発ワークフロー

推奨される開発フローは以下の通りです：

### 1. データ収集

ROS2コンテナを使用してデモンストレーションデータを収集：

```bash
docker compose -f ros2/docker-compose.yml --profile real up  # または --profile sim
```

### 2. ファインチューニング

VLAコンテナに切り替えてトレーニング：

```bash
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune
```

### 3. 推論とデプロイ

ファインチューニングされたモデルをロボット制御に統合：

```bash
# （実装はデプロイ設定に依存）
```

## チェックポイントの管理

### チェックポイントの保存場所

トレーニング完了後、チェックポイントは以下に保存されます：

```
outputs/crane_x7_finetune/
├── checkpoint-1000/
├── checkpoint-2000/
└── checkpoint-final/
```

### チェックポイントからの再開

```bash
python3 finetune.py \
    --resume_from_checkpoint /workspace/outputs/crane_x7_finetune/checkpoint-1000
```

## クリーンアップ

### コンテナとボリュームの削除

```bash
# すべてのコンテナを停止
docker compose down

# ボリュームを削除
docker compose down -v

# イメージを削除（方法A）
docker rmi $(docker images -q '*vla_finetune*')

# イメージを削除（方法B）
docker rmi crane_x7_vla:latest crane_x7_vla:dev
```

## 環境変数リファレンス（方法B）

方法Bを使用する場合、`.env`ファイルで以下の変数を設定できます：

| 変数名 | 説明 | デフォルト値 |
|--------|------|--------------|
| `WANDB_API_KEY` | Weights & Biases APIキー | （空） |
| `HF_TOKEN` | HuggingFace APIトークン | （空） |
| `DATA_DIR` | データセットディレクトリパス | `../data/tfrecord_logs` |
| `CHECKPOINT_DIR` | モデルチェックポイントディレクトリ | `./checkpoints` |
| `LOG_DIR` | トレーニングログディレクトリ | `./logs` |
| `HF_CACHE` | HuggingFaceキャッシュディレクトリ | `~/.cache/huggingface` |
| `JUPYTER_PORT` | Jupyterポート（開発モード） | `8888` |
| `TENSORBOARD_PORT` | TensorBoardポート（開発モード） | `6006` |

## トレーニングパラメータリファレンス

コマンドライン引数または環境変数でトレーニングを設定：

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

## 参考資料

- [OpenVLAドキュメント](https://github.com/openvla/openvla)
- [OpenVLA公式サイト](https://openvla.github.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPUサポート](https://docs.docker.com/compose/gpu-support/)
- [PyTorch分散トレーニング](https://pytorch.org/tutorials/beginner/dist_overview.html)
