# CRANE-X7 Vision-Language-Action (VLA) ファインチューニング

このディレクトリには、CRANE-X7ロボットのデモンストレーションデータを使用してVLAモデル（OpenVLAおよびOpenPI）をファインチューニングするためのスクリプトと統合が含まれています。

## 概要

このプロジェクトは、ビジョンベースのロボットマニピュレーションタスクのために、2つの最先端VLAバックエンドをサポートしています:

### サポートされているバックエンド

| バックエンド | Dockerfile | Requirements | PyTorch | Transformers | 主な特徴 |
|------------|-----------|--------------|---------|--------------|---------|
| **OpenVLA** | `Dockerfile.openvla` | `requirements-openvla.txt` | 2.2.0 | 4.40.1 | Prismatic VLM、単一ステップアクション |
| **OpenPI** | `Dockerfile.openpi` | `requirements-openpi.txt` | 2.7.1 | 4.53.2 | JAX/Flax、アクションチャンク、Python 3.11+ |

**重要**: OpenVLAとOpenPIは互いに競合する依存関係を持つため、**別々のDockerイメージとrequirementsファイル**を使用します。同じ環境で両方をインストールすることはできません。

### ディレクトリ構造

```
vla/
├── Dockerfile.openvla          # OpenVLA専用Dockerイメージ
├── Dockerfile.openpi           # OpenPI専用Dockerイメージ
├── requirements-openvla.txt    # OpenVLA依存関係
├── requirements-openpi.txt     # OpenPI依存関係
├── docker-compose.yml          # Docker Compose設定（非推奨、ros2/docker-compose.ymlを使用）
├── src/
│   ├── crane_x7_vla/          # 統一されたトレーニングCLI
│   │   ├── training/
│   │   │   ├── cli.py         # メインCLIエントリポイント
│   │   │   ├── config.py      # 設定データクラス
│   │   │   └── trainer.py     # トレーニングロジック
│   │   ├── backends/          # バックエンド固有の実装
│   │   │   ├── openvla_wrapper.py
│   │   │   └── openpi_wrapper.py
│   │   └── ...
│   └── openvla/               # OpenVLAサブモジュール
├── test_crane_x7_loader.py    # データセット検証スクリプト
└── README.md                  # このファイル
```

## 必要要件

### オプション1: Dockerを使用（推奨）

すべての依存関係がインストール済みの事前設定されたDocker環境を使用します。

#### 環境変数設定

まず、`ros2/.env.template`から`.env`ファイルを作成し、必要な環境変数を設定します:

```bash
cd ros2
cp .env.template .env
```

`.env`ファイルを編集して以下を設定:

```bash
# Hugging Face token (モデルダウンロード用、必須)
HF_TOKEN=your_huggingface_token

# Weights & Biases API key (ロギング用、オプション)
WANDB_API_KEY=your_wandb_api_key

# GPU設定
CUDA_VISIBLE_DEVICES=0  # 使用するGPU ID（カンマ区切りで複数指定可能）
```

#### OpenVLA環境構築

```bash
# OpenVLA用Dockerイメージをビルド
docker compose -f ros2/docker-compose.yml build vla_openvla

# インタラクティブコンテナを起動
docker compose -f ros2/docker-compose.yml run --rm vla_openvla
```

#### OpenPI環境構築

```bash
# OpenPI用Dockerイメージをビルド
docker compose -f ros2/docker-compose.yml build vla_openpi

# インタラクティブコンテナを起動
docker compose -f ros2/docker-compose.yml run --rm vla_openpi
```

### オプション2: ローカルインストール（非推奨）

**警告**: OpenVLAとOpenPIは依存関係が競合するため、別々の仮想環境を使用する必要があります。

#### OpenVLA環境

```bash
python -m venv venv_openvla
source venv_openvla/bin/activate
pip install -r requirements-openvla.txt

# オプション: より高速な学習のためのFlash Attention 2
pip install flash-attn==2.5.5 --no-build-isolation
```

#### OpenPI環境

```bash
python -m venv venv_openpi
source venv_openpi/bin/activate
pip install -r requirements-openpi.txt
```

## データ収集ワークフロー

VLAモデルをファインチューニングする前に、デモンストレーションデータを収集する必要があります。

### 1. テレオペレーションでデータ収集

```bash
# テレオペレーション（手動教示）でデータ収集
# リーダーロボットを手で動かしながらデータを記録
docker compose -f ros2/docker-compose.yml --profile teleop-leader-logger up

# データは自動的に data/tfrecord_logs に保存されます
```

### 2. 言語インストラクションの設定

データ収集中に、別のターミナルから言語インストラクションをパブリッシュできます:

```bash
# コンテナ内で実行
ros2 topic pub /task/language_instruction std_msgs/String "data: 'Pick up the red block and place it in the blue bin'"
```

### 3. データフォーマット

収集されたデータは以下の構造で保存されます:

```
data/tfrecord_logs/
├── episode_0000_YYYYMMDD_HHMMSS/
│   └── episode_data.tfrecord
├── episode_0001_YYYYMMDD_HHMMSS/
│   └── episode_data.tfrecord
└── dataset_statistics.json
```

各TFRecordファイルには以下が含まれます:

- `observation/state`: 関節位置（8自由度: 7アーム + 1グリッパー）
- `observation/image`: RGB画像（JPEGエンコード、オプション）
- `observation/depth`: デプス画像（オプション）
- `observation/timestamp`: UNIXタイムスタンプ
- `action`: 次の状態 / 目標関節位置（`action[t] = state[t+1]`形式）

## クイックスタート

### 1. データセット検証

データセットが正しく読み込めることを確認:

```bash
# OpenVLAコンテナ内で
docker compose -f ros2/docker-compose.yml run --rm vla_openvla \
  python3 /workspace/vla/test_crane_x7_loader.py

# OpenPIコンテナ内で
docker compose -f ros2/docker-compose.yml run --rm vla_openpi \
  python3 /workspace/vla/test_crane_x7_loader.py
```

### 2. ファインチューニングの実行

#### OpenVLAファインチューニング

**シングルGPU**:

```bash
docker compose -f ros2/docker-compose.yml run --rm vla_openvla \
  python -m crane_x7_vla.training.cli train \
    --backend openvla \
    --data-root /workspace/data/tfrecord_logs \
    --experiment-name crane_x7_openvla \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --num-epochs 100
```

**マルチGPU（例: 2台）**:

```bash
docker compose -f ros2/docker-compose.yml run --rm vla_openvla \
  torchrun --nproc_per_node=2 -m crane_x7_vla.training.cli train \
    --backend openvla \
    --data-root /workspace/data/tfrecord_logs \
    --experiment-name crane_x7_openvla \
    --batch-size 8 \
    --learning-rate 5e-4 \
    --num-epochs 100
```

チェックポイントは`/workspace/outputs/crane_x7_openvla/`に保存されます。

#### OpenPIファインチューニング

**シングルGPU**:

```bash
docker compose -f ros2/docker-compose.yml run --rm vla_openpi \
  python -m crane_x7_vla.training.cli train \
    --backend openpi \
    --data-root /workspace/data/tfrecord_logs \
    --experiment-name crane_x7_openpi \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --num-epochs 100
```

チェックポイントは`/workspace/outputs/crane_x7_openpi/`に保存されます。

### 3. Weights & Biasロギング

トレーニング進行状況を追跡するため、W&Bを有効化:

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla \
  --use-wandb \
  --wandb-project crane-x7-vla \
  --wandb-entity your-username
```

## CLI引数

`crane_x7_vla.training.cli`は以下の引数をサポートしています:

### 必須引数

- `--backend {openvla,openpi}`: 使用するVLAバックエンド

### データ設定

- `--data-root PATH`: TFRecordデータディレクトリ（デフォルト: `/workspace/data/tfrecord_logs`）
- `--instruction TEXT`: タスク指示（デフォルト: `"Pick and place the object"`）
- `--image-size WIDTHxHEIGHT`: 画像サイズ（デフォルト: `224x224`）

### トレーニング設定

- `--batch-size INT`: GPU毎のバッチサイズ（デフォルト: OpenVLA=16, OpenPI=32）
- `--num-epochs INT`: 学習エポック数（デフォルト: 100）
- `--learning-rate FLOAT`: 学習率（デフォルト: OpenVLA=5e-4, OpenPI=3e-4）
- `--grad-accumulation-steps INT`: 勾配蓄積ステップ数（デフォルト: 1）
- `--warmup-ratio FLOAT`: ウォームアップ比率（デフォルト: 0.1）

### LoRA設定（OpenVLAのみ）

- `--use-lora`: LoRAファインチューニングを有効化（デフォルト: True）
- `--lora-rank INT`: LoRAランク（デフォルト: 32）
- `--lora-alpha INT`: LoRAアルファ（デフォルト: 64）
- `--lora-dropout FLOAT`: LoRAドロップアウト（デフォルト: 0.1）

### 出力設定

- `--experiment-name NAME`: 実験名（デフォルト: `crane_x7_{backend}`）
- `--output-dir PATH`: 出力ディレクトリ（デフォルト: `/workspace/outputs/{experiment_name}`）
- `--save-steps INT`: チェックポイント保存間隔（デフォルト: 500）

### ロギング設定

- `--use-wandb`: Weights & Biasロギングを有効化
- `--wandb-project NAME`: W&Bプロジェクト名
- `--wandb-entity NAME`: W&Bエンティティ名

### その他

- `--gradient-checkpointing`: メモリ効率化のため勾配チェックポイントを有効化
- `--num-workers INT`: データローダーワーカー数（デフォルト: 4）
- `--seed INT`: 乱数シード（デフォルト: 42）

## メモリ要件

### OpenVLA（LoRAファインチューニング）

LoRA（rank=32）を使用する場合のメモリ要件:

- **シングルGPU（A100 40GB）**: バッチサイズ 8 - 12
- **シングルGPU（A100 80GB）**: バッチサイズ 16 - 24
- **マルチGPU（2x A100 40GB）**: バッチサイズ 8 x 2 = 16（有効）

### OpenPI

OpenPIはJAX/Flaxベースで、異なるメモリ特性を持ちます:

- **シングルGPU（A100 40GB）**: バッチサイズ 16 - 32
- **シングルGPU（A100 80GB）**: バッチサイズ 32 - 64
- **マルチGPU**: JAXの自動シャーディングを使用

## 出力

トレーニングスクリプトは以下を保存します:

1. **チェックポイント**: `{output_dir}/checkpoint-{step}/`
   - LoRAアダプター重み（OpenVLA + LoRA使用時）
   - 完全なモデル重み（フルファインチューニング時）
   - プロセッサ設定
   - トレーニング状態（オプティマイザ、スケジューラ）

2. **ログ**:
   - コンソール出力（損失、学習率、進行状況）
   - Weights & Biases（有効時）
   - TensorBoard（`{output_dir}/tensorboard/`）

3. **設定ファイル**:
   - `{output_dir}/training_config.json`: トレーニング設定
   - `{output_dir}/model_config.json`: モデル設定

## ファインチューニング済みモデルの使用

### OpenVLAモデルの読み込み

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch

# ベースモデルを読み込み
base_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRAアダプターを読み込み
model = PeftModel.from_pretrained(
    base_model,
    "/workspace/outputs/crane_x7_openvla/checkpoint-5000"
)

# プロセッサを読み込み
processor = AutoProcessor.from_pretrained(
    "/workspace/outputs/crane_x7_openvla/checkpoint-5000",
    trust_remote_code=True
)

# 推論モード
model.eval()
model.to("cuda")

# 推論実行
inputs = processor(
    images=image,
    text="Pick up the red block",
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)
    action = processor.decode(outputs[0], skip_special_tokens=True)
```

### OpenPIモデルの読み込み

```python
# OpenPI固有のロード方法については、
# src/crane_x7_vla/backends/openpi_wrapper.py を参照してください
```

## トラブルシューティング

### メモリ不足（CUDA Out of Memory）

1. `--batch-size`を減らす（例: 16 → 8）
2. `--grad-accumulation-steps`を増やす（例: 1 → 2）
3. `--gradient-checkpointing`を有効化
4. `--lora-rank`を減らす（例: 32 → 16）
5. マルチGPUトレーニングを使用

### tokenizers/transformersバージョンエラー

```
ERROR: tokenizers>=0.21,<0.22 is required but...
```

**原因**: OpenVLAとOpenPIで必要なバージョンが異なります。

**解決策**: 適切なDockerイメージを使用してください:
- OpenVLA: `docker compose -f ros2/docker-compose.yml build vla_openvla`
- OpenPI: `docker compose -f ros2/docker-compose.yml build vla_openpi`

**重要**: 両方を同じ環境にインストールしないでください。

### データセット読み込みエラー

```bash
# データセット構造を確認
ls -lh /workspace/data/tfrecord_logs/

# データセット統計を確認
cat /workspace/data/tfrecord_logs/dataset_statistics.json

# TFRecordファイルを検証
python3 /workspace/vla/test_crane_x7_loader.py
```

### トレーニングが遅い

1. `--num-workers`を増やす（例: 4 → 8）
2. Flash Attention 2を有効化（OpenVLA）
3. マルチGPUトレーニングを使用（`torchrun`）
4. より小さい画像サイズを使用（例: `--image-size 224x224`）

### JAX/Flaxエラー（OpenPI使用時）

```bash
# CUDA設定を確認
nvidia-smi

# 環境変数を設定
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
```

### Hugging Faceトークンエラー

```bash
# .envファイルでHF_TOKENが設定されているか確認
cat ros2/.env | grep HF_TOKEN

# または環境変数で設定
export HF_TOKEN=your_huggingface_token
```

## 高度な使い方

### カスタムデータセット指示

複数の指示でトレーニング:

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --instruction "Pick up the {color} block and place it in the {location}"
```

### 画像なしのトレーニング（状態のみ）

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --no-use-image
```

### チェックポイントから再開

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --resume-from-checkpoint /workspace/outputs/crane_x7_openvla/checkpoint-5000
```

### カスタムモデルパス

```bash
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --vla-path openvla/openvla-7b-rlhf
```

## デプロイメント

ファインチューニング済みモデルをROS 2と統合してリアルタイム推論を実行:

```bash
# VLA推論ノードを起動
ros2 launch crane_x7_vla vla_inference.launch.py \
  model_path:=/workspace/outputs/crane_x7_openvla/checkpoint-5000
```

詳細については、[crane_x7_vla ROS 2パッケージ](../ros2/src/crane_x7_vla/)を参照してください。

## 参考資料

### OpenVLA

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA論文](https://arxiv.org/abs/2406.09246)
- [OpenVLAモデル（Hugging Face）](https://huggingface.co/openvla)

### OpenPI

- [OpenPI GitHub](https://github.com/rail-berkeley/openpi)
- [OpenPI論文](https://arxiv.org/abs/2410.14369)
- [OpenPIモデル（Hugging Face）](https://huggingface.co/rail-berkeley/openpi)

### CRANE-X7ロボット

- [CRANE-X7 ROS 2](https://github.com/rt-net/crane_x7_ros)
- [CRANE-X7 Description](https://github.com/rt-net/crane_x7_description)
- [RT Corporation](https://rt-net.jp/)

### Open X-Embodiment

- [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/)
- [OXE論文](https://arxiv.org/abs/2310.08864)

## ライセンス

- このパッケージ（crane_x7_vla）: MITライセンス（Copyright 2025 nop）
- OpenVLA: MITライセンス
- OpenPI: MITライセンス
- 事前訓練済みモデル: 各モデルのライセンスに従う（例: Llama-2ライセンス）

詳細については、各プロジェクトのLICENSEファイルを参照してください。
