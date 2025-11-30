# VLAトレーニング用Docker環境

このガイドでは、CRANE-X7データセットでVLAモデル（OpenVLAおよびOpenPI）をトレーニングするためのDocker環境のセットアップと使用方法を説明します。

## 概要

このプロジェクトでは、2つのVLAバックエンドをサポートしています：

| バックエンド | Dockerfile | Python | PyTorch | Transformers | 主な特徴 |
|------------|-----------|--------|---------|--------------|---------|
| **OpenVLA** | `Dockerfile.openvla` | 3.10 | 2.5.1 | 4.57.3 | Prismatic VLM、単一ステップアクション、LoRA対応 |
| **OpenPI** | `Dockerfile.openpi` | 3.11 | 2.7.1 | 4.53.2 | JAX/Flax、アクションチャンク対応 |

**重要**: OpenVLAとOpenPIは互いに競合する依存関係を持つため、**別々のDockerイメージ**を使用します。

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

**マルチGPUスケーリング（OpenVLA + LoRA）：**
- 1x A100 (40GB): バッチサイズ ~8-12
- 2x A100 (40GB): バッチサイズ ~16-24
- 4x A100 (40GB): バッチサイズ ~32-48

### ソフトウェア要件

- Docker (>= 20.10)
- Docker Compose (>= 2.0)
- CUDA対応のNVIDIA GPU
- NVIDIA Container Toolkit ([インストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### GPUアクセスの確認

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

## クイックスタート

### 1. 環境設定

```bash
cd vla
cp .env.template .env
# .envを編集して設定を変更
```

必須の設定：
- `HF_TOKEN`: HuggingFaceトークン（必須、事前学習済みモデルのダウンロードに必要）
- `WANDB_API_KEY`: Weights & Biases APIキー（オプション、実験トラッキング用）
- `CUDA_VISIBLE_DEVICES`: 使用するGPU ID（デフォルト: `0`）

### 2. Dockerイメージのビルド

```bash
cd vla

# OpenVLA用
docker compose --profile openvla build

# OpenPI用
docker compose --profile openpi build
```

### 3. トレーニングの実行

#### OpenVLA

```bash
cd vla
docker compose --profile openvla run --rm vla-finetune-openvla \
  python -m crane_x7_vla.training.cli train \
    --backend openvla \
    --data-root /workspace/data/tfrecord_logs \
    --experiment-name crane_x7_openvla \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --num-epochs 100
```

#### OpenPI

```bash
cd vla
docker compose --profile openpi run --rm vla-finetune-openpi \
  python -m crane_x7_vla.training.cli train \
    --backend openpi \
    --data-root /workspace/data/tfrecord_logs \
    --experiment-name crane_x7_openpi \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --num-epochs 100
```

## Docker環境の詳細

### Dockerイメージ構成

両方のDockerfileは以下の構成を採用しています：

- **ベースイメージ**: `nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`
- **マルチステージビルド**: builder → base → dev
- **非rootユーザー**: ホストUID/GIDと一致

### コンテナ内のディレクトリ構造

```
/workspace/
├── vla/
│   ├── src/                    # ソースコード
│   │   ├── crane_x7_vla/      # トレーニングCLI
│   │   ├── openvla/           # OpenVLAサブモジュール
│   │   └── openpi/            # OpenPIサブモジュール
│   └── configs/               # 設定ファイル
├── data/                      # トレーニングデータ（マウント）
│   └── tfrecord_logs/
└── outputs/                   # チェックポイントとログ（マウント）
```

### ボリュームマウント

Docker Composeは以下のボリュームをマウントします：

| ホスト | コンテナ | 説明 |
|-------|---------|------|
| `../data` | `/workspace/data` | トレーニングデータ |
| `../outputs` | `/workspace/outputs` | チェックポイント出力 |
| `./src` | `/workspace/vla/src` | ソースコード（開発用） |
| `~/.cache/huggingface` | `/home/vla/.cache/huggingface` | モデルキャッシュ |

## インタラクティブセッション

### コンテナへの接続

```bash
# OpenVLAコンテナをインタラクティブに起動
docker compose --profile openvla run --rm vla-finetune-openvla bash

# コンテナ内でデータセット読み込みをテスト
python3 /workspace/vla/test_crane_x7_loader.py

# ファインチューニングを実行
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --data-root /workspace/data/tfrecord_logs
```

### マルチGPUトレーニング

```bash
docker compose --profile openvla run --rm vla-finetune-openvla \
  torchrun --nproc_per_node=2 -m crane_x7_vla.training.cli train \
    --backend openvla \
    --data-root /workspace/data/tfrecord_logs \
    --batch-size 8 \
    --learning-rate 5e-4
```

## GPU設定

### GPU数の指定

docker-compose.ymlは利用可能な全GPUを使用するように設定されています。

特定のGPUを使用するには、`.env`ファイルを編集：

```bash
# 特定のGPUを指定
CUDA_VISIBLE_DEVICES=0,1

# GPU数を指定
GPU_COUNT=2
```

または環境変数で直接指定：

```bash
docker compose --profile openvla run --rm \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  vla-finetune-openvla
```

## トレーニングの監視

### Weights & Biasesの使用

```bash
# .envファイルでAPIキーを設定
WANDB_API_KEY=your_api_key
WANDB_MODE=online

# W&Bロギングを有効にしてトレーニングを実行
docker compose --profile openvla run --rm vla-finetune-openvla \
  python -m crane_x7_vla.training.cli train \
    --backend openvla \
    --data-root /workspace/data/tfrecord_logs \
    --use-wandb \
    --wandb-project crane-x7-openvla \
    --wandb-entity YOUR_USERNAME
```

### TensorBoard

```bash
# devステージのイメージをビルド
docker compose --profile openvla build --target dev

# コンテナ内でTensorBoardを起動
tensorboard --logdir=/workspace/outputs --host=0.0.0.0

# http://localhost:6006 でアクセス
```

### GPU使用率の確認

```bash
# コンテナ内
watch -n 1 nvidia-smi

# またはホストから
docker exec crane_x7_vla_finetune_openvla nvidia-smi
```

## トラブルシューティング

### GPUが検出されない

```bash
# NVIDIAランタイムを確認
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi

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
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --batch-size 4
```

**解決策2: 勾配蓄積を使用**

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --batch-size 4 \
  --grad-accumulation-steps 4
```

**解決策3: 勾配チェックポインティングを有効化**

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --gradient-checkpointing
```

**解決策4: LoRAランクを減らす（OpenVLA）**

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --lora-rank 16
```

**解決策5: 共有メモリを増やす**

`.env`ファイルで共有メモリを調整：

```bash
SHM_SIZE=32gb
```

### 依存関係の競合

OpenVLAとOpenPIは異なるバージョンの依存関係を使用します：

| パッケージ | OpenVLA | OpenPI |
|-----------|---------|--------|
| transformers | 4.57.3 | 4.53.2 |
| torch | 2.5.1 | 2.7.1 |
| JAX | なし | 0.5.3 |

**解決策**: 適切なDockerイメージを使用し、両方を同じ環境にインストールしないでください。

### HuggingFace認証失敗

`.env`に`HF_TOKEN`が設定されており、モデルへのアクセス権があることを確認：

```bash
# コンテナ内で手動ログイン
huggingface-cli login
```

### パーミッションの問題

マウントされたボリュームでパーミッションエラーが発生した場合、`.env`ファイルでUID/GIDを設定：

```bash
USER_ID=1000
GROUP_ID=1000
```

## CLI引数リファレンス

### 必須引数

| 引数 | 説明 |
|------|------|
| `--backend {openvla,openpi}` | 使用するVLAバックエンド |

### データ設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--data-root PATH` | TFRecordデータディレクトリ | `/workspace/data/tfrecord_logs` |
| `--instruction TEXT` | タスク指示 | `"Pick and place the object"` |
| `--image-size WxH` | 画像サイズ | `224x224` |

### トレーニング設定

| 引数 | 説明 | OpenVLAデフォルト | OpenPIデフォルト |
|------|------|------------------|-----------------|
| `--batch-size INT` | バッチサイズ | 16 | 32 |
| `--num-epochs INT` | エポック数 | 100 | 100 |
| `--learning-rate FLOAT` | 学習率 | 5e-4 | 3e-4 |
| `--grad-accumulation-steps INT` | 勾配蓄積 | 1 | 1 |
| `--warmup-ratio FLOAT` | ウォームアップ比率 | 0.1 | 0.1 |
| `--gradient-checkpointing` | メモリ効率化 | - | - |

### LoRA設定（OpenVLAのみ）

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--use-lora` | LoRAを有効化 | True |
| `--lora-rank INT` | LoRAランク | 32 |
| `--lora-alpha INT` | LoRAアルファ | 64 |
| `--lora-dropout FLOAT` | LoRAドロップアウト | 0.1 |

### 出力設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--experiment-name NAME` | 実験名 | `crane_x7_{backend}` |
| `--output-dir PATH` | 出力ディレクトリ | `/workspace/outputs/{experiment_name}` |
| `--save-steps INT` | チェックポイント間隔 | 500 |

### ロギング設定

| 引数 | 説明 |
|------|------|
| `--use-wandb` | W&Bロギングを有効化 |
| `--wandb-project NAME` | W&Bプロジェクト名 |
| `--wandb-entity NAME` | W&Bエンティティ名 |

## 環境変数リファレンス

`.env`ファイルで以下の変数を設定できます：

| 変数名 | 説明 | デフォルト値 |
|--------|------|--------------|
| `HF_TOKEN` | HuggingFace APIトークン | （空） |
| `WANDB_API_KEY` | Weights & Biases APIキー | （空） |
| `WANDB_MODE` | W&Bモード | `disabled` |
| `CUDA_VERSION` | CUDAバージョン | `12.9.1` |
| `PYTHON_VERSION` | Pythonバージョン | `3.10` |
| `CUDA_VISIBLE_DEVICES` | 使用するGPU | `0` |
| `GPU_COUNT` | GPU数 | `all` |
| `SHM_SIZE` | 共有メモリサイズ | `16gb` |
| `USER_ID` | ユーザーID | `1000` |
| `GROUP_ID` | グループID | `1000` |

## チェックポイントの管理

### チェックポイントの保存場所

トレーニング完了後、チェックポイントは以下に保存されます：

```
outputs/{experiment_name}/
├── checkpoint-500/
├── checkpoint-1000/
├── ...
└── checkpoint-final/
```

### チェックポイントからの再開

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --resume-from-checkpoint /workspace/outputs/crane_x7_openvla/checkpoint-1000
```

## 開発ワークフロー

推奨される開発フローは以下の通りです：

### 1. データ収集（ROS 2コンテナ）

```bash
cd ros2
docker compose --profile teleop-leader-logger up
```

### 2. ファインチューニング（VLAコンテナ）

```bash
cd vla
docker compose --profile openvla run --rm vla-finetune-openvla
```

### 3. 推論とデプロイ（ROS 2コンテナ）

```bash
cd ros2
ros2 launch crane_x7_vla vla_inference.launch.py \
  model_path:=/workspace/outputs/crane_x7_openvla/checkpoint-final
```

## クリーンアップ

### コンテナとボリュームの削除

```bash
cd vla

# すべてのコンテナを停止
docker compose down

# ボリュームを削除
docker compose down -v

# イメージを削除
docker rmi crane_x7_vla:openvla crane_x7_vla:openpi
```

## 参考資料

- [OpenVLAドキュメント](https://github.com/openvla/openvla)
- [OpenVLA公式サイト](https://openvla.github.io/)
- [OpenPI GitHub](https://github.com/rail-berkeley/openpi)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPUサポート](https://docs.docker.com/compose/gpu-support/)
- [PyTorch分散トレーニング](https://pytorch.org/tutorials/beginner/dist_overview.html)
