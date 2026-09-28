# CRANE-X7 VLA ファインチューニング

CRANE-X7ロボットアーム用のVision-Language-Action（VLA）モデルをファインチューニングするためのフレームワーク。

## 概要

このディレクトリでは、以下のVLAバックエンドをサポートしています：

| バックエンド | 説明 | パラメータ | 推論速度 | 状態 |
|-------------|------|-----------|---------|------|
| **OpenVLA** | Prismatic VLMベースの7Bパラメータモデル | ~7B | ~5Hz | 実装済み |
| **OpenVLA-OFT** | L1 Regression + Action Chunking + FiLM | ~7B | ~8Hz | 実装済み |
| **MiniVLA** | Qwen 2.5 0.5B + VQ Action Chunking | ~1B | ~12.5Hz | 実装済み |
| **Pi0** | PaliGemma + Expert Gemma + Flow Matching | ~2.3B | ~3Hz | 実装済み |
| **Pi0.5** | Pi0 + adaRMSNorm + Discrete State | ~2.3B | ~3Hz | 実装済み |

すべてのバックエンドは統一Dockerfile（`vla/Dockerfile`）に含まれています。

### OpenVLA-OFT（Optimized Fine-Tuning）の特徴

[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)は、標準OpenVLAの改良版です：

- **L1 Regression Action Head**: トークン離散化の代わりに連続アクション予測
- **Action Chunking**: 複数の将来アクション（デフォルト8ステップ）を一度に予測
- **FiLM（Feature-wise Linear Modulation）**: 言語-ビジョン統合の改善
- **Proprioceptive Input**: ロボット状態を入力として使用
- **Multi-image Support**: 複数カメラ入力に対応

### MiniVLAの特徴

MiniVLAは軽量で高速な推論を実現するVLAモデルです：

- **軽量**: OpenVLAの約1/7のパラメータ（~1B vs ~7B）
- **高速推論**: ~12.5Hz（OpenVLAの~2.5倍）
- **VQ Action Chunking**: 複数の将来アクションを効率的に予測
- **Multi-image Support**: 画像履歴 + 手首カメラ入力に対応

### Pi0/Pi0.5の特徴

OpenPIのPyTorch実装に基づくPi0/Pi0.5モデル：

- **PaliGemma + Expert Gemma**: VLMとアクション専門家の組み合わせアーキテクチャ
- **Flow Matching**: 拡散モデルベースのアクション生成
- **50-step Action Chunking**: 長期アクション予測
- **マルチカメラサポート**: 最大3カメラ入力に対応

**Pi0 vs Pi0.5の違い:**

| 特徴 | Pi0 | Pi0.5 |
|------|-----|-------|
| State入力 | 連続（MLPで処理） | 離散（言語トークンに含む） |
| Timestep注入 | MLP | adaRMSNorm |
| max_token_len | 48 | 200 |
| メモリ使用量 | 低 | 高 |

## クイックスタート

### 1. Dockerイメージのビルド

```bash
cd /path/to/crane_x7_vla/vla

# 統一イメージをビルド（全バックエンド含む）
docker build -t crane_x7_vla .
```

### 2. トレーニングの実行

```bash
# データディレクトリをマウントしてコンテナを起動
docker run --gpus all -it --rm \
  --env-file .env \
  --net host \
  -v $(pwd)/..:/workspace \
  -v ~/.cache:/home/vla/.cache \
  crane_x7_vla

# コンテナ内でトレーニング実行（OpenVLA）
python -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla

# OpenVLA-OFT（Action Chunking + L1 Regression）
python -m crane_x7_vla.training.cli train openvla-oft \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla_oft

# MiniVLA（軽量・高速）
python -m crane_x7_vla.training.cli train minivla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_minivla

# Pi0（PaliGemma + Expert Gemma）
python -m crane_x7_vla.training.cli train pi0 \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_pi0

# Pi0.5（adaRMSNorm + Discrete State）
python -m crane_x7_vla.training.cli train pi0.5 \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_pi05
```

## 環境構築

### 前提条件

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU（VRAM 24GB以上推奨）
- CUDA 12.x対応ドライバ

### Dockerイメージ

統一Dockerfileにすべてのバックエンドの依存関係が含まれています：

```bash
# 統一イメージのビルド（推奨）
docker build -t crane_x7_vla .

# 異なるCUDAバージョンでビルド
docker build --build-arg CUDA_VERSION=12.8.0 --build-arg CUDA_SHORT=cu128 -t crane_x7_vla .
```

**環境仕様**:
- CUDA 12.8.0
- Python 3.11
- PyTorch 2.9.1
- Flash Attention 2.8.3

### HuggingFaceモデルのダウンロード

事前学習済みモデルは自動的にダウンロードされますが、手動でキャッシュすることも可能です：

```bash
# HuggingFaceにログイン
huggingface-cli login

# OpenVLAモデルのダウンロード（約14GB）
huggingface-cli download openvla/openvla-7b

# Pi0モデルのダウンロード
huggingface-cli download lerobot/pi0_base
```

## データの準備

### TFRecord形式（ROS 2 crane_x7_logから出力）

データは`crane_x7_log`パッケージで収集し、以下の形式でTFRecordとして保存されます：

```
data/tfrecord_logs/
├── episode_0/
│   ├── episode_0.tfrecord
│   └── episode_metadata.json
├── episode_1/
│   └── ...
└── ...
```

各TFRecordには以下のデータが含まれます：

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `observation/state` | float32[8] | 関節位置（7軸 + グリッパー） |
| `observation/image` | bytes | JPEG圧縮RGB画像 |
| `action` | float32[8] | 次ステップの関節位置 |
| `language_instruction` | string | タスク指示（例: "pick up the red block"） |

### データ収集の方法

```bash
# ROS 2環境でデータ収集
docker compose --profile log up
```

## トレーニング

### 基本的な使い方

```bash
# OpenVLAでトレーニング（デフォルト設定）
python -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name my_experiment

# 設定ファイルを使用
python -m crane_x7_vla.training.cli train openvla \
  --config /workspace/vla/configs/openvla_default.yaml

# 設定ファイル + CLI引数でオーバーライド
python -m crane_x7_vla.training.cli train openvla \
  --config /workspace/vla/configs/openvla_default.yaml \
  --batch-size 32 \
  --learning-rate 1e-4
```

### 利用可能なバックエンド

| バックエンド名 | 説明 |
|--------------|------|
| `openvla` | 標準OpenVLA（トークン化アクション） |
| `openvla-oft` | OpenVLA-OFT（L1 Regression + Action Chunking） |
| `minivla` | MiniVLA（軽量 + VQ Action Chunking） |
| `pi0` | Pi0（PaliGemma + Expert Gemma + Flow Matching） |
| `pi0.5` | Pi0.5（adaRMSNorm + Discrete State入力） |

### CLI引数一覧

#### 共通引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--config` | - | YAML設定ファイルのパス |
| `--data-root` | - | トレーニングデータのディレクトリ |
| `--output-dir` | `./outputs` | 出力ディレクトリ |
| `--experiment-name` | `crane_x7_vla` | 実験名 |

#### トレーニング設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--batch-size` | 16 | バッチサイズ |
| `--learning-rate` | 5e-4 | 学習率 |
| `--num-epochs` | 100 | エポック数 |
| `--max-steps` | - | 最大ステップ数 |
| `--grad-accumulation-steps` | 1 | 勾配累積ステップ |

### 設定ファイルの生成

```bash
# OpenVLAデフォルト設定ファイルを生成
python -m crane_x7_vla.training.cli config \
  --backend openvla \
  --output openvla_config.yaml \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name my_experiment

# OpenVLA-OFT設定
python -m crane_x7_vla.training.cli config \
  --backend openvla-oft \
  --output openvla_oft_config.yaml

# MiniVLA設定
python -m crane_x7_vla.training.cli config \
  --backend minivla \
  --output minivla_config.yaml

# Pi0設定
python -m crane_x7_vla.training.cli config \
  --backend pi0 \
  --output pi0_config.yaml

# Pi0.5設定
python -m crane_x7_vla.training.cli config \
  --backend pi0.5 \
  --output pi05_config.yaml
```

### マルチGPUトレーニング

```bash
# 2GPU並列
torchrun --nproc_per_node=2 -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name multi_gpu_experiment

# 4GPU並列
torchrun --nproc_per_node=4 -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --batch-size 8  # GPUあたりのバッチサイズ
```

## バックエンド固有設定

### OpenVLA固有設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `model_id` | `openvla/openvla-7b` | HuggingFaceモデルID |
| `use_lora` | True | LoRAを使用 |
| `lora_rank` | 32 | LoRAランク |
| `lora_alpha` | 16 | LoRAアルファ |
| `lora_dropout` | 0.05 | LoRAドロップアウト |
| `action_tokenization_bins` | 256 | アクション離散化ビン数 |
| `image_aug` | True | 画像拡張を使用 |
| `skip_merge_on_save` | True | 保存時にLoRAマージをスキップ |

### OpenVLA-OFT固有設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `action_horizon` | 8 | アクションチャンク長 |
| `film.enabled` | True | FiLMを有効化 |
| `proprio.enabled` | True | プロプリオセプティブ入力を有効化 |
| `multi_image.enabled` | True | マルチ画像入力を有効化 |
| `multi_image.num_images` | 2 | 画像数（primary + wrist） |
| `action_head.hidden_dim` | 4096 | Action Head隠れ層次元 |
| `action_head.num_blocks` | 2 | MLPResNetブロック数 |

### MiniVLA固有設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `llm_model_id` | `Qwen/Qwen2.5-0.5B` | LLMモデルID |
| `vision_backbone` | `dinosiglip-vit-so-224px` | ビジョンバックボーン |
| `use_lora` | True | LoRAを使用 |
| `lora_rank` | 16 | LoRAランク |
| `use_flash_attention` | True | Flash Attentionを使用 |

#### VQ Action Chunking設定（MiniVLA）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `vq.enabled` | True | VQ Action Chunkingを有効化 |
| `vq.action_horizon` | 8 | アクションチャンク長 |
| `vq.n_embed` | 256 | コードブックサイズ |
| `vq.n_latent` | 512 | 潜在次元 |
| `vq.n_groups` | 7 | Residual VQグループ数 |

#### Multi-image設定（MiniVLA）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `multi_image.enabled` | True | マルチ画像入力を有効化 |
| `multi_image.image_history` | 2 | 履歴フレーム数 |
| `multi_image.use_wrist_camera` | True | 手首カメラを使用 |

### Pi0/Pi0.5固有設定

Pi0とPi0.5は同じバックエンドクラスで`model_type`設定により切り替えます。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `model_type` | `pi0` | モデルタイプ（`pi0`または`pi0.5`） |
| `paligemma_variant` | `gemma_2b` | VLMバックボーン |
| `action_expert_variant` | `gemma_300m` | アクション専門家モデル |
| `pretrained_checkpoint` | `null` | 事前学習済みチェックポイント |
| `action_dim` | 32 | アクション次元（パディング含む） |
| `state_dim` | 32 | 状態次元 |
| `action_horizon` | 50 | アクションチャンク長 |
| `max_token_len` | 48/200 | 最大トークン長（Pi0: 48, Pi0.5: 200） |
| `discrete_state_input` | false/true | 離散状態入力（Pi0.5で自動true） |
| `num_denoise_steps` | 10 | Flow Matchingデノイズステップ |
| `normalize_actions` | true | アクション正規化 |
| `normalization_mode` | `quantile` | 正規化モード（`quantile`または`zscore`） |
| `quantile_low` | 0.01 | Quantile正規化下限（normalization_mode=quantile時） |
| `quantile_high` | 0.99 | Quantile正規化上限（normalization_mode=quantile時） |
| `use_delta_actions` | false | デルタアクション（action = next_state - current_state）を使用 |
| `freeze_vlm` | true | VLM（PaliGemma）を凍結 |
| `freeze_action_expert` | false | アクション専門家を凍結 |
| `precision` | `bfloat16` | 計算精度 |
| `openpi_checkpoint` | null | OpenPIチェックポイント名（下記参照） |

#### OpenPIチェックポイント（Pi0/Pi0.5）

事前学習済みPi0/Pi0.5モデルを使用できます。`openpi_checkpoint`に以下のいずれかを指定：

| チェックポイント名 | 説明 |
|------------------|------|
| `pi0_base` | 基本Pi0モデル（ファインチューニング用） |
| `pi05_base` | 基本Pi0.5モデル（ファインチューニング用） |
| `pi0_droid` | DROIDデータセットで学習済みPi0 |
| `pi05_droid` | DROIDデータセットで学習済みPi0.5 |

**使用例:**

```bash
python -m crane_x7_vla.training.cli train pi0 \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_pi0 \
  --openpi-checkpoint pi0_base
```

#### カメラ設定（Pi0/Pi0.5）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `image_size` | [224, 224] | 入力画像サイズ |
| `num_cameras` | 1 | カメラ数（最大3） |
| `camera_names` | [base_0_rgb] | カメラ名リスト |

利用可能なカメラ名: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`

#### LoRA設定（Pi0/Pi0.5）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `use_lora` | false | LoRAを使用 |
| `lora_rank` | 32 | LoRAランク |
| `lora_alpha` | 16 | LoRAアルファ |
| `lora_dropout` | 0.1 | LoRAドロップアウト |

## LoRAアダプターのマージ

トレーニング中は効率のためLoRAアダプターのみが保存されます。**推論にはマージ済みモデルが必須**です。

> **重要**: LoRAアダプターのパス（`checkpoint-XXXX/lora_adapters`）を直接`VLA_MODEL_PATH`に指定するとエラーになります。必ず以下の手順でマージを実行してください。

### マージの実行

```bash
# GPU環境でマージを実行
docker run --gpus all --rm \
  -v /path/to/vla/outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  crane_x7_vla python -m crane_x7_vla.scripts.merge_lora \
  --adapter_path /workspace/outputs/my_experiment/checkpoint-7000/lora_adapters \
  --output_path /workspace/outputs/my_experiment_merged \
  --base_model openvla/openvla-7b
```

### 出力ディレクトリ構成

マージ後のモデルは以下の場所に保存されます：

```
outputs/my_experiment_merged/
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── tokenizer.json
├── preprocessor_config.json
└── dataset_statistics.json
```

### チェックポイントの選択

トレーニング中に複数のチェックポイントが保存されます：

```
outputs/my_experiment/
├── checkpoint-1000/
│   └── lora_adapters/
├── checkpoint-2000/
│   └── lora_adapters/
└── checkpoint-7000/    ← 最終チェックポイント
    └── lora_adapters/
```

評価ロスを確認してベストなチェックポイントを選択してください（W&Bのeval/lossを参照）。

## ハイパーパラメータチューニング（W&B Sweeps）

### Sweepの作成

```bash
# W&Bでsweepを作成
wandb sweep sweep_config.yaml
# 出力例: Created sweep with ID: abc123xyz
```

`sweep_config.yaml`の例：

```yaml
program: python
method: bayes
metric:
  name: eval/loss
  goal: minimize
parameters:
  learning_rate:
    distribution: log_uniform
    min: -10  # 1e-5
    max: -6   # 1e-3
  batch_size:
    values: [8, 16, 32]
  lora_rank:
    values: [16, 32, 64]
  lora_dropout:
    distribution: uniform
    min: 0.0
    max: 0.2
```

### Sweepエージェントの実行

VLAトレーニングCLIにはW&B Sweepエージェント機能が組み込まれています：

```bash
# Pi0.5でSweepエージェントを実行
python -m crane_x7_vla.training.cli agent pi0.5 \
  --sweep-id <SWEEP_ID> \
  --entity <WANDB_ENTITY> \
  --project <WANDB_PROJECT> \
  --data-root /workspace/data/tfrecord_logs \
  --output-dir /workspace/outputs/checkpoints \
  --training-max-steps 10000

# OpenVLAでSweepエージェントを実行
python -m crane_x7_vla.training.cli agent openvla \
  --sweep-id abc123xyz \
  --entity myteam \
  --project crane-x7-sweep \
  --data-root /workspace/data/tfrecord_logs
```

## 推論（ROS 2統合）

学習済みモデルをROS 2で使用する場合：

```bash
# 実機で推論
docker compose --profile vla up

# シミュレーションで推論
docker compose --profile vla-sim up
```

### モデルパスの設定

`.env`ファイルで**マージ済みモデル**のパスを設定：

```env
# 正しい例（マージ済みモデル）
VLA_MODEL_PATH=/workspace/vla/outputs/my_experiment_merged

# 間違った例（LoRAアダプター） - エラーになる
# VLA_MODEL_PATH=/workspace/vla/outputs/my_experiment/checkpoint-7000/lora_adapters
```

> **注意**: `Failed to load VLA model`エラーが発生する場合は、パスがマージ済みモデルを指しているか確認してください。LoRAアダプターのパスは直接ロードできません。

## ディレクトリ構成

```
vla/
├── Dockerfile                     # 統一Dockerfile（全バックエンド含む）
├── requirements-base.txt          # 共通依存関係
├── requirements-openvla.txt       # OpenVLA依存関係
├── requirements-minivla.txt       # MiniVLA依存関係
├── configs/
│   ├── openvla_default.yaml       # OpenVLAデフォルト設定
│   ├── minivla_default.yaml       # MiniVLAデフォルト設定
│   ├── pi0_default.yaml           # Pi0デフォルト設定
│   └── pi05_default.yaml          # Pi0.5デフォルト設定
├── outputs/                       # 学習出力（チェックポイント）
├── src/
│   ├── crane_x7_vla/              # 統一トレーニングフレームワーク
│   │   ├── training/
│   │   │   ├── cli.py             # コマンドラインインターフェース
│   │   │   └── trainer.py         # 統一トレーナー
│   │   ├── backends/              # バックエンド実装（サブパッケージ）
│   │   │   ├── __init__.py        # バックエンド登録
│   │   │   ├── openvla/           # OpenVLAバックエンド
│   │   │   │   ├── backend.py
│   │   │   │   ├── config.py
│   │   │   │   └── dataset.py
│   │   │   ├── openvla_oft/       # OpenVLA-OFTバックエンド
│   │   │   │   ├── backend.py
│   │   │   │   ├── config.py
│   │   │   │   ├── components.py  # FiLM, ActionHead等
│   │   │   │   └── dataset.py
│   │   │   ├── minivla/           # MiniVLAバックエンド
│   │   │   │   ├── backend.py
│   │   │   │   ├── config.py
│   │   │   │   ├── dataset.py
│   │   │   │   └── action_tokenizer/
│   │   │   │       ├── vq.py              # Residual VQ実装
│   │   │   │       └── vq_tokenizer.py    # VQアクショントークナイザー
│   │   │   └── pi0/               # Pi0/Pi0.5バックエンド
│   │   │       ├── backend.py     # Pi0Backend実装
│   │   │       ├── config.py      # Pi0Config
│   │   │       ├── model.py       # PaliGemmaWithExpertModel
│   │   │       └── dataset.py     # CraneX7Pi0Dataset
│   │   ├── core/                  # 共有コンポーネント
│   │   │   ├── base.py            # VLABackend基底クラス
│   │   │   ├── config/            # 設定データクラス
│   │   │   ├── data/              # データ変換・検証
│   │   │   ├── transforms/        # 画像・アクション変換
│   │   │   └── utils/             # ユーティリティ
│   │   ├── scripts/               # ユーティリティスクリプト
│   │   │   ├── merge_lora.py      # LoRAマージスクリプト
│   │   │   └── compute_crane_x7_norm_stats.py
│   │   └── policies/              # 推論用ポリシー
│   ├── openvla/                   # OpenVLAサブモジュール
│   └── openpi/                    # OpenPIサブモジュール
└── README.md
```

## トラブルシューティング

### Failed to load VLA model エラー

LoRAアダプターのパスを直接指定した場合に発生します：

```
# エラー例
VLA_MODEL_PATH=/workspace/vla/outputs/.../checkpoint-7000/lora_adapters
```

**解決方法**: LoRAマージを実行してマージ済みモデルを作成し、そのパスを指定してください。詳細は[LoRAアダプターのマージ](#loraアダプターのマージ)を参照。

### OOM（Out of Memory）エラー

```bash
# バッチサイズを小さくする
python -m crane_x7_vla.training.cli train openvla \
  --batch-size 8

# 勾配チェックポインティングを有効化（設定ファイルで）
# training.gradient_checkpointing: true
```

### NCCL タイムアウト（マルチGPU）

`skip_merge_on_save`はデフォルトで有効です。これにより、チェックポイント保存時のLoRAマージをスキップし、NCCLタイムアウトを回避します。

### TensorFlowの警告

TensorFlowの警告は環境変数で抑制できます：

```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

## 参考リンク

- [OpenVLA](https://github.com/openvla/openvla) - Prismatic VLMベースのVLAモデル
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) - Optimized Fine-Tuning論文
- [MiniVLA Blog](https://ai.stanford.edu/blog/minivla/) - Stanford SAILによるMiniVLA紹介
- [OpenPI](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence社のπ₀モデル
- [HuggingFace OpenVLA](https://huggingface.co/openvla/openvla-7b) - 事前学習済みモデル
- [HuggingFace Pi0](https://huggingface.co/lerobot/pi0_base) - Pi0事前学習済みモデル
- [HuggingFace Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B) - MiniVLAベースモデル
- [VQ-BeT](https://arxiv.org/abs/2403.03181) - VQ Action Chunkingの参考論文

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）
- **OpenVLA**: MIT License（コード部分）
  - 事前学習済みモデルには別途制限あり（Llama-2ライセンス等）
- **OpenPI**: Apache License 2.0
