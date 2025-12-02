# CRANE-X7 Vision-Language-Action (VLA) ファインチューニング

このディレクトリには、CRANE-X7ロボットのデモンストレーションデータを使用してVLAモデル（OpenVLAおよびOpenPI）をファインチューニングするためのスクリプトと統合が含まれています。

## 概要

このプロジェクトは、ビジョンベースのロボットマニピュレーションタスクのために、2つの最先端VLAバックエンドをサポートしています:

### サポートされているバックエンド

| バックエンド | Dockerfile | Requirements | Python | PyTorch | Transformers | 主な特徴 |
|------------|-----------|--------------|--------|---------|--------------|---------|
| **OpenVLA** | `Dockerfile.openvla` | `requirements-openvla.txt` | 3.10 | 2.5.1 | 4.57.3 | Prismatic VLM、単一ステップアクション |
| **OpenPI** | `Dockerfile.openpi` | `requirements-openpi.txt` | 3.11 | 2.7.1 | 4.53.2 | JAX/Flax、アクションチャンク |

**重要**: OpenVLAとOpenPIは互いに競合する依存関係を持つため、**別々のDockerイメージとrequirementsファイル**を使用します。同じ環境で両方をインストールすることはできません。

### ディレクトリ構造

```
vla/
├── Dockerfile.openvla          # OpenVLA専用Dockerイメージ (CUDA 12.9, Python 3.10)
├── Dockerfile.openpi           # OpenPI専用Dockerイメージ (CUDA 12.9, Python 3.11)
├── requirements-openvla.txt    # OpenVLA依存関係
├── requirements-openpi.txt     # OpenPI依存関係
├── docker-compose.yml          # Docker Compose設定
├── .env.template               # 環境変数テンプレート
├── configs/                    # 設定ファイル
│   ├── openvla_default.yaml   # OpenVLAデフォルト設定
│   └── openpi_default.yaml    # OpenPIデフォルト設定
├── src/
│   ├── crane_x7_vla/          # 統一されたトレーニングCLI
│   │   ├── training/
│   │   │   ├── cli.py         # メインCLIエントリポイント
│   │   │   └── trainer.py     # トレーニングロジック
│   │   ├── backends/          # バックエンド固有の実装
│   │   │   ├── openvla.py     # OpenVLAバックエンド
│   │   │   ├── openpi.py      # OpenPIバックエンド
│   │   │   └── base.py        # 基底クラス
│   │   ├── config/            # 設定モジュール
│   │   │   ├── base.py        # 基本設定（UnifiedVLAConfig等）
│   │   │   ├── openvla_config.py  # OpenVLA固有設定
│   │   │   └── openpi_config.py   # OpenPI固有設定
│   │   ├── data/              # データローダー
│   │   ├── transforms/        # データ変換
│   │   └── scripts/           # ユーティリティスクリプト
│   │       └── merge_lora.py  # LoRAアダプターマージスクリプト
│   ├── openvla/               # OpenVLAサブモジュール
│   └── openpi/                # OpenPIサブモジュール
└── README.md                  # このファイル
```

## 必要要件

### オプション1: Dockerを使用（推奨）

すべての依存関係がインストール済みの事前設定されたDocker環境を使用します。

#### 環境変数設定

まず、`.env.template`から`.env`ファイルを作成し、必要な環境変数を設定します:

```bash
cd vla
cp .env.template .env
```

`.env`ファイルを編集して以下を設定:

```bash
# Hugging Face token (モデルダウンロード用、必須)
HF_TOKEN=your_huggingface_token

# Weights & Biases API key (ロギング用、オプション)
WANDB_API_KEY=your_wandb_api_key
WANDB_MODE=disabled  # disabled, online, offline

# GPU設定
CUDA_VISIBLE_DEVICES=0  # 使用するGPU ID（カンマ区切りで複数指定可能）
GPU_COUNT=all           # 使用するGPU数

# ユーザー設定（パーミッション用）
USER_ID=1000   # id -u で取得
GROUP_ID=1000  # id -g で取得
USERNAME=vla
```

#### OpenVLA環境構築

```bash
cd vla

# OpenVLA用Dockerイメージをビルド
docker compose --profile openvla build

# インタラクティブコンテナを起動
docker compose --profile openvla run --rm vla-finetune-openvla bash
```

#### OpenPI環境構築

```bash
cd vla

# OpenPI用Dockerイメージをビルド
docker compose --profile openpi build

# インタラクティブコンテナを起動
docker compose --profile openpi run --rm vla-finetune-openpi bash
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

### 1. ファインチューニングの実行

#### OpenVLAファインチューニング

**シングルGPU**:

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

**設定ファイルを使用**:

```bash
cd vla
docker compose --profile openvla run --rm vla-finetune-openvla \
  python -m crane_x7_vla.training.cli train \
    --config /workspace/vla/configs/openvla_default.yaml
```

**マルチGPU（例: 2台）**:

```bash
cd vla
docker compose --profile openvla run --rm vla-finetune-openvla \
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
cd vla
docker compose --profile openpi run --rm vla-finetune-openpi \
  python -m crane_x7_vla.training.cli train \
    --backend openpi \
    --data-root /workspace/data/tfrecord_logs \
    --experiment-name crane_x7_openpi \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --num-epochs 100
```

チェックポイントは`/workspace/outputs/crane_x7_openpi/`に保存されます。

### 2. デフォルト設定ファイルの生成

カスタム設定ファイルを生成してから編集:

```bash
python -m crane_x7_vla.training.cli config \
  --backend openvla \
  --output my_config.yaml \
  --data-root ./data \
  --experiment-name my_experiment
```

### 3. 評価

```bash
python -m crane_x7_vla.training.cli evaluate \
  --config my_config.yaml \
  --checkpoint /workspace/outputs/crane_x7_openvla/checkpoint-5000
```

## CLI引数

`crane_x7_vla.training.cli`は3つのサブコマンドをサポートしています: `train`, `evaluate`, `config`

### trainコマンド

```bash
python -m crane_x7_vla.training.cli train [オプション]
```

#### 基本設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--backend {openvla,openpi}` | 使用するVLAバックエンド | `openvla` |
| `--config PATH` | YAML設定ファイルのパス | なし |
| `--data-root PATH` | TFRecordデータディレクトリ | 必須（configなしの場合） |
| `--output-dir PATH` | 出力ディレクトリ | `./outputs` |
| `--experiment-name NAME` | 実験名 | `crane_x7_vla` |

#### トレーニング設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--batch-size INT` | GPU毎のバッチサイズ | 16 |
| `--num-epochs INT` | 学習エポック数 | 100 |
| `--learning-rate FLOAT` | 学習率 | 5e-4 |
| `--weight-decay FLOAT` | 重み減衰 | 0.01 |
| `--gradient-checkpointing` | 勾配チェックポイントを有効化 | false |

#### 検証設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--val-interval INT` | 検証実行間隔（ステップ） | 500 |
| `--val-steps INT` | 検証時のバッチ数 | 50 |

#### チェックポイント設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--save-interval INT` | チェックポイント保存間隔 | 1000 |

#### OpenVLA固有設定

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--lora-rank INT` | LoRAランク | 32 |
| `--use-quantization` | 量子化を有効化（4-bit/8-bit） | false |

### evaluateコマンド

```bash
python -m crane_x7_vla.training.cli evaluate [オプション]
```

| 引数 | 説明 |
|------|------|
| `--config PATH` | YAML設定ファイルのパス（必須） |
| `--checkpoint PATH` | チェックポイントのパス（必須） |
| `--test-data PATH` | テストデータのパス（オプション） |

### configコマンド

```bash
python -m crane_x7_vla.training.cli config [オプション]
```

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--backend {openvla,openpi}` | VLAバックエンド（必須） | - |
| `--output PATH` | 出力設定ファイルパス | `config.yaml` |
| `--data-root PATH` | データディレクトリ | `./data` |
| `--output-dir PATH` | 出力ディレクトリ | `./outputs` |
| `--experiment-name NAME` | 実験名 | `crane_x7_vla` |

## 設定ファイル（YAML）

YAML設定ファイルを使用すると、より詳細なトレーニング設定が可能です。

### 基本構造

```yaml
# バックエンドと基本設定
backend: openvla  # または openpi
output_dir: ./outputs/openvla
experiment_name: crane_x7_openvla
seed: 42
resume_from_checkpoint: null

# データ設定
data:
  data_root: ./data/tfrecord_logs
  format: tfrecord
  train_split: 0.9
  val_split: 0.1
  shuffle: true
  num_workers: 4
  prefetch_factor: 2
  cameras:
    - name: primary
      topic: /camera/color/image_raw
      enabled: true
      width: 640
      height: 480

# トレーニング設定
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0005
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  mixed_precision: bf16
  gradient_checkpointing: false
  save_interval: 1000
  eval_interval: 500
  log_interval: 10

# 検証設定
validation:
  val_split_ratio: 0.1
  val_interval: 500
  val_steps: 50

# バックエンド固有設定（backend_config）
backend_config:
  model_id: openvla/openvla-7b
  use_lora: true
  lora_rank: 32
  lora_alpha: 16
  lora_dropout: 0.05
  # ... 詳細は configs/ 内のサンプルを参照
```

### サンプル設定ファイル

- `configs/openvla_default.yaml`: OpenVLA用のデフォルト設定
- `configs/openpi_default.yaml`: OpenPI用のデフォルト設定

## LoRAアダプターのマージ

トレーニング後、LoRAアダプターをベースモデルにマージして推論用のスタンドアロンモデルを作成できます。

### 使い方

```bash
python -m crane_x7_vla.scripts.merge_lora \
  --adapter_path /workspace/outputs/crane_x7_openvla/lora_adapters \
  --output_path /workspace/outputs/crane_x7_openvla_merged \
  --base_model openvla/openvla-7b
```

### 引数

| 引数 | 説明 | デフォルト |
|------|------|----------|
| `--adapter_path` | LoRAアダプターのディレクトリ（必須） | - |
| `--output_path` | マージ済みモデルの保存先（必須） | - |
| `--base_model` | ベースモデルのHF ID | `openvla/openvla-7b` |
| `--no_copy_processor` | プロセッサファイルをコピーしない | false |
| `--no_copy_statistics` | データセット統計をコピーしない | false |

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

2. **設定ファイル**:
   - `{output_dir}/config.yaml`: トレーニング設定

3. **ログ**:
   - コンソール出力（損失、学習率、進行状況）
   - Weights & Biases（WANDB_MODE=onlineの場合）

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
    "/workspace/outputs/crane_x7_openvla/lora_adapters"
)

# プロセッサを読み込み
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
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

### マージ済みモデルの読み込み

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

# マージ済みモデルを直接読み込み
model = AutoModelForVision2Seq.from_pretrained(
    "/workspace/outputs/crane_x7_openvla_merged",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "/workspace/outputs/crane_x7_openvla_merged",
    trust_remote_code=True
)
```

## トラブルシューティング

### メモリ不足（CUDA Out of Memory）

1. `--batch-size`を減らす（例: 16 → 8）
2. `--gradient-checkpointing`を有効化
3. `--lora-rank`を減らす（例: 32 → 16）
4. `--use-quantization`を有効化（OpenVLA）
5. マルチGPUトレーニングを使用

### tokenizers/transformersバージョンエラー

```
ERROR: tokenizers>=0.21,<0.22 is required but...
```

**原因**: OpenVLAとOpenPIで必要なバージョンが異なります。

**解決策**: 適切なDockerイメージを使用してください:
- OpenVLA: `docker compose --profile openvla build`
- OpenPI: `docker compose --profile openpi build`

**重要**: 両方を同じ環境にインストールしないでください。

### データセット読み込みエラー

```bash
# データセット構造を確認
ls -lh /workspace/data/tfrecord_logs/

# データセット統計を確認
cat /workspace/data/tfrecord_logs/dataset_statistics.json
```

### トレーニングが遅い

1. データローダーワーカー数を増やす（設定ファイルの`num_workers`）
2. Flash Attention 2を有効化（OpenVLA、設定ファイルの`use_flash_attention`）
3. マルチGPUトレーニングを使用（`torchrun`）
4. より小さい画像サイズを使用（設定ファイルの`image_size`）

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
cat vla/.env | grep HF_TOKEN

# または環境変数で設定
export HF_TOKEN=your_huggingface_token
```

## デプロイメント

ファインチューニング済みモデルをROS 2と統合してリアルタイム推論を実行:

```bash
# VLA推論ノードを起動
ros2 launch crane_x7_vla vla_inference.launch.py \
  model_path:=/workspace/outputs/crane_x7_openvla_merged
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
