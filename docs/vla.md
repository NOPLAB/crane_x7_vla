# CRANE-X7 VLA ファインチューニング

CRANE-X7ロボットアーム用のVision-Language-Action（VLA）モデルをファインチューニングするためのフレームワーク。

## 概要

このディレクトリでは、以下のVLAバックエンドをサポートしています：

| バックエンド | 説明 | Dockerfile | パラメータ | 推論速度 | 状態 |
|-------------|------|------------|-----------|---------|------|
| **OpenVLA** | Prismatic VLMベースの7Bパラメータモデル | `Dockerfile.openvla` | ~7B | ~5Hz | 実装済み |
| **MiniVLA** | Qwen 2.5 0.5B + VQ Action Chunking | `Dockerfile.minivla` | ~1B | ~12.5Hz | 実装済み |
| **OpenPI** | Physical Intelligence社のπ₀モデル（JAX版） | `Dockerfile.openpi` | - | - | 未実装 |
| **OpenPI PyTorch** | π₀モデルのHuggingFace/PyTorch実装 | `Dockerfile.openpi-pytorch` | - | - | 未実装 |

### MiniVLAの特徴

MiniVLAは軽量で高速な推論を実現するVLAモデルです：

- **軽量**: OpenVLAの約1/7のパラメータ（~1B vs ~7B）
- **高速推論**: ~12.5Hz（OpenVLAの~2.5倍）
- **VQ Action Chunking**: 複数の将来アクションを効率的に予測
- **Multi-image Support**: 画像履歴 + 手首カメラ入力に対応

## クイックスタート

### 1. Dockerイメージのビルド

```bash
cd /path/to/crane_x7_vla/vla

# OpenVLA用
docker build -f Dockerfile.openvla -t crane_x7_vla_openvla .

# MiniVLA用（軽量版）
docker build -f Dockerfile.minivla -t crane_x7_vla_minivla .
```

### 2. トレーニングの実行

```bash
# データディレクトリをマウントしてコンテナを起動
docker run --gpus all -it --rm \
  --env-file .env \
  --net host \
  -v $(pwd)/..:/workspace \
  -v ~/.cache:/home/vla/.cache \
  crane_x7_vla_openvla

# コンテナ内でトレーニング実行（OpenVLA）
python -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla

# MiniVLAの場合
docker run --gpus all -it --rm \
  --env-file .env \
  --net host \
  -v $(pwd)/..:/workspace \
  -v ~/.cache:/home/vla/.cache \
  crane_x7_vla_minivla

# コンテナ内でトレーニング実行（MiniVLA）
python -m crane_x7_vla.training.cli train minivla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_minivla \
  --vq-enabled \
  --multi-image-enabled
```

## 環境構築

### 前提条件

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU（VRAM 24GB以上推奨）
- CUDA 12.x対応ドライバ

### Dockerイメージ

各VLAバックエンドは依存関係が異なるため、別々のDockerイメージを使用します：

```bash
# OpenVLA（Python 3.10, PyTorch 2.5.1, ~7Bパラメータ）
docker build -f Dockerfile.openvla -t crane_x7_vla_openvla .

# MiniVLA（Python 3.10, PyTorch 2.5.1, ~1Bパラメータ、軽量・高速）
docker build -f Dockerfile.minivla -t crane_x7_vla_minivla .

# OpenPI JAX版（Python 3.11, JAX）
docker build -f Dockerfile.openpi -t crane_x7_vla_openpi .

# OpenPI PyTorch版（Python 3.11, PyTorch 2.7.1）
docker build -f Dockerfile.openpi-pytorch -t crane_x7_vla_openpi_pytorch .
```

### HuggingFaceモデルのダウンロード

事前学習済みモデルは自動的にダウンロードされますが、手動でキャッシュすることも可能です：

```bash
# HuggingFaceにログイン
huggingface-cli login

# OpenVLAモデルのダウンロード（約14GB）
huggingface-cli download openvla/openvla-7b
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
cd ros2/docker
docker compose --profile teleop up
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
  --training-batch-size 32 \
  --training-learning-rate 1e-4
```

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
| `--training-batch-size` | 16 | バッチサイズ |
| `--training-num-epochs` | 100 | エポック数 |
| `--training-learning-rate` | 5e-4 | 学習率 |
| `--training-weight-decay` | 0.01 | Weight decay |
| `--training-warmup-steps` | 1000 | Warmupステップ数 |
| `--training-max-grad-norm` | 1.0 | 勾配クリッピング |
| `--training-mixed-precision` | bf16 | 混合精度（no/fp16/bf16） |
| `--training-save-interval` | 1000 | チェックポイント保存間隔 |

#### OpenVLA固有設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--model-id` | `openvla/openvla-7b` | HuggingFaceモデルID |
| `--use-lora` | True | LoRAを使用 |
| `--lora-rank` | 32 | LoRAランク |
| `--lora-alpha` | 16 | LoRAアルファ |
| `--lora-dropout` | 0.05 | LoRAドロップアウト |
| `--image-aug` | True | 画像拡張を使用 |
| `--skip-merge-on-save` | True | 保存時にLoRAマージをスキップ |

#### MiniVLA固有設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--llm-model-id` | `Qwen/Qwen2.5-0.5B` | LLMモデルID |
| `--use-lora` | True | LoRAを使用 |
| `--lora-rank` | 16 | LoRAランク |
| `--lora-alpha` | 8 | LoRAアルファ |
| `--use-flash-attention` | True | Flash Attentionを使用 |
| `--image-aug` | True | 画像拡張を使用 |

#### VQ Action Chunking設定（MiniVLA）

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--vq-enabled` | True | VQ Action Chunkingを有効化 |
| `--vq-action-horizon` | 8 | アクションチャンク長 |
| `--vq-n-embed` | 256 | コードブックサイズ |
| `--vq-n-latent` | 512 | 潜在次元 |
| `--vq-n-groups` | 7 | Residual VQグループ数 |

#### Multi-image設定（MiniVLA）

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--multi-image-enabled` | True | マルチ画像入力を有効化 |
| `--multi-image-image-history` | 2 | 履歴フレーム数 |
| `--multi-image-use-wrist-camera` | True | 手首カメラを使用 |

### 設定ファイルの生成

```bash
# OpenVLAデフォルト設定ファイルを生成
python -m crane_x7_vla.training.cli config \
  --backend openvla \
  --output openvla_config.yaml \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name my_experiment

# MiniVLAデフォルト設定ファイルを生成
python -m crane_x7_vla.training.cli config \
  --backend minivla \
  --output minivla_config.yaml \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name my_experiment
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
  --training-batch-size 8  # GPUあたりのバッチサイズ
```

## LoRAアダプターのマージ

トレーニング中は効率のためLoRAアダプターのみが保存されます。**推論にはマージ済みモデルが必須**です。

> **重要**: LoRAアダプターのパス（`checkpoint-XXXX/lora_adapters`）を直接`VLA_MODEL_PATH`に指定するとエラーになります。必ず以下の手順でマージを実行してください。

### マージの実行

```bash
# GPU環境でマージを実行
docker run --gpus all --rm \
  -v /path/to/vla/outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  noppdev/vla python -m crane_x7_vla.scripts.merge_lora \
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

```bash
# 単一エージェント
python -m crane_x7_vla.training.cli agent openvla \
  --sweep-id abc123xyz \
  --data-root /workspace/data/tfrecord_logs \
  --project crane_x7

# 複数回実行
python -m crane_x7_vla.training.cli agent openvla \
  --sweep-id abc123xyz \
  --data-root /workspace/data/tfrecord_logs \
  --count 10
```

### Slurmクラスターでの実行

```bash
cd ../slurm
slurm-submit sweep start examples/sweeps/sweep_openvla.yaml --max-runs 10
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
├── Dockerfile.openvla          # OpenVLA用Dockerfile
├── Dockerfile.minivla          # MiniVLA用Dockerfile
├── Dockerfile.openpi           # OpenPI JAX版用Dockerfile
├── Dockerfile.openpi-pytorch   # OpenPI PyTorch版用Dockerfile
├── requirements-openvla.txt    # OpenVLA依存関係
├── requirements-minivla.txt    # MiniVLA依存関係
├── requirements-openpi.txt     # OpenPI依存関係
├── configs/
│   ├── openvla_default.yaml    # OpenVLAデフォルト設定
│   ├── minivla_default.yaml    # MiniVLAデフォルト設定
│   └── openpi_default.yaml     # OpenPIデフォルト設定
├── outputs/                    # 学習出力（チェックポイント）
├── src/
│   ├── crane_x7_vla/           # 統一トレーニングフレームワーク
│   │   ├── training/
│   │   │   ├── cli.py          # コマンドラインインターフェース
│   │   │   └── trainer.py      # 統一トレーナー
│   │   ├── backends/           # バックエンド実装
│   │   │   ├── openvla.py      # OpenVLAバックエンド
│   │   │   ├── minivla.py      # MiniVLAバックエンド
│   │   │   └── openpi.py       # OpenPIバックエンド
│   │   ├── action_tokenizer/   # アクショントークナイザー
│   │   │   ├── vq.py           # Residual VQ実装
│   │   │   └── vq_tokenizer.py # VQアクショントークナイザー
│   │   ├── config/             # 設定データクラス
│   │   ├── data/               # データローダー
│   │   │   └── minivla_dataset.py  # MiniVLAマルチ画像データセット
│   │   ├── scripts/            # ユーティリティスクリプト
│   │   │   └── merge_lora.py   # LoRAマージスクリプト
│   │   └── transforms/         # データ変換
│   ├── openvla/                # OpenVLAサブモジュール
│   └── openpi/                 # OpenPIサブモジュール
└── README.md                   # このファイル
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
  --training-batch-size 8

# 勾配チェックポインティングを有効化
python -m crane_x7_vla.training.cli train openvla \
  --training-gradient-checkpointing
```

### NCCL タイムアウト（マルチGPU）

`--skip-merge-on-save`はデフォルトで有効です。これにより、チェックポイント保存時のLoRAマージをスキップし、NCCLタイムアウトを回避します。

### TensorFlowの警告

TensorFlowの警告は環境変数で抑制できます：

```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

## 参考リンク

- [OpenVLA](https://github.com/openvla/openvla) - Prismatic VLMベースのVLAモデル
- [MiniVLA Blog](https://ai.stanford.edu/blog/minivla/) - Stanford SAILによるMiniVLA紹介
- [OpenPI](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence社のπ₀モデル
- [HuggingFace OpenVLA](https://huggingface.co/openvla/openvla-7b) - 事前学習済みモデル
- [HuggingFace Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B) - MiniVLAベースモデル
- [VQ-BeT](https://arxiv.org/abs/2403.03181) - VQ Action Chunkingの参考論文

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）
- **OpenVLA**: MIT License
- **OpenPI**: Apache License 2.0
