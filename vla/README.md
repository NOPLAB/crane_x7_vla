# CRANE-X7 OpenVLA ファインチューニング

このディレクトリには、CRANE-X7ロボットのデモンストレーションデータを使用してOpenVLAをファインチューニングするためのスクリプトが含まれています。

## 📢 重要なお知らせ: RLDS形式への移行

**crane_x7_logパッケージはRLDS（Robot Learning Dataset Standard）形式で直接データを出力するようになりました！**

これにより：
- ✅ OpenVLAとの完全な互換性
- ✅ データ変換スクリプト不要
- ✅ 言語インストラクション対応
- ✅ データセット統計の自動計算

詳細は [CRANE_X7_INTEGRATION.md](CRANE_X7_INTEGRATION.md) を参照してください。

## 概要

### 新しいRLDS統合（推奨）

1. **test_crane_x7_loader.py**: RLDS形式データの検証スクリプト
2. **CRANE_X7_INTEGRATION.md**: OpenVLAとの統合完全ガイド
3. **openvla/prismatic/vla/datasets/rlds/oxe/configs.py**: crane_x7データセット設定
4. **openvla/prismatic/vla/datasets/rlds/oxe/transforms.py**: crane_x7データ変換関数

### レガシーファインチューニングパイプライン

1. **crane_x7_dataset.py**: 旧形式TFRecordデータ用のPyTorch Datasetローダー（非推奨）
2. **finetune_config.py**: ファインチューニングパラメータの設定データクラス
3. **finetune.py**: LoRAサポート付きメイン学習スクリプト

## 必要要件

### オプション1: Dockerを使用（推奨）

すべての依存関係がインストール済みの事前設定されたDocker環境を使用：

```bash
# VLAファインチューニングイメージをビルド
docker compose build vla_finetune

# インタラクティブコンテナを実行
docker compose --profile vla run --rm vla_finetune

# またはヘルパースクリプトを使用
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train
```

詳細なDocker手順については [docker_usage.md](docker_usage.md) を参照してください。

### オプション2: ローカルインストール

必要な依存関係をインストール：

```bash
# 全ての要件をインストール
pip install -r requirements.txt

# オプション: より高速な学習のためのFlash Attention 2
pip install flash-attn==2.5.5 --no-build-isolation
```

## データフォーマット

ファインチューニングスクリプトは以下の構造のデータを想定しています：

```
data/tfrecord_logs/
├── episode_0000_TIMESTAMP/
│   └── episode_data.tfrecord
├── episode_0001_TIMESTAMP/
│   └── episode_data.tfrecord
└── ...
```

各TFRecordファイルには以下が含まれます：

- `observation/state`: 関節位置（7自由度float配列）
- `observation/image`: RGB画像（JPEGエンコードされたbytes）
- `observation/timestamp`: タイムスタンプ（float）
- `action`: 次の状態 / 目標関節位置（7自由度float配列）

## クイックスタート

### 1. データセット読み込みテスト

データセットが正しく読み込めることを確認：

```bash
cd vla
python crane_x7_dataset.py ../data/tfrecord_logs
```

### 2. ファインチューニングの実行

#### シングルGPU

```bash
cd vla
python finetune.py
```

#### マルチGPU（PyTorch DDP）

```bash
cd vla
torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py
```

#### カスタムパラメータ指定

```bash
python finetune.py \
  --data_root ../data/tfrecord_logs \
  --output_dir ../outputs/my_finetune \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_epochs 10 \
  --lora_rank 32
```

#### Weights & Biases ロギング付き

```bash
python finetune.py \
  --use_wandb \
  --wandb_project crane-x7-openvla \
  --wandb_entity your-username
```

## 設定

主要な設定パラメータ（詳細は `finetune_config.py` を参照）：

### モデル設定

- `vla_path`: HuggingFaceモデルパス（デフォルト: `"openvla/openvla-7b"`）
- `use_flash_attention`: より高速な学習のためにFlash Attention 2を使用（デフォルト: `True`）

### データ設定

- `data_root`: TFRecordデータへのパス（デフォルト: `"data/tfrecord_logs"`）
- `instruction`: 条件付けのためのタスク指示（デフォルト: `"Pick and place the object"`）
- `use_image`: カメラ画像を使用するか（デフォルト: `True`）
- `image_size`: ターゲット画像サイズ（デフォルト: `(224, 224)`）

### 学習設定

- `batch_size`: GPU毎のバッチサイズ（デフォルト: `8`）
- `num_epochs`: 学習エポック数（デフォルト: `10`）
- `learning_rate`: 学習率（デフォルト: `5e-4`）
- `grad_accumulation_steps`: 勾配蓄積ステップ数（デフォルト: `1`）

### LoRA設定

- `use_lora`: パラメータ効率的ファインチューニングのためにLoRAを使用（デフォルト: `True`）
- `lora_rank`: LoRAランク（デフォルト: `32`）
- `lora_alpha`: LoRAアルファスケーリング係数（デフォルト: `64`）
- `lora_dropout`: LoRAドロップアウト（デフォルト: `0.1`）

### チェックポイント設定

- `output_dir`: 出力ディレクトリ（デフォルト: `"outputs/crane_x7_finetune"`）
- `save_steps`: チェックポイント保存間隔（デフォルト: `500`）
- `save_total_limit`: 保持する最大チェックポイント数（デフォルト: `3`）

## メモリ要件

### LoRAファインチューニング（推奨）

LoRA（rank=32）を使用する場合のメモリ要件：

- **シングルGPU（A100 40GB）**: バッチサイズ 8 - 12
- **シングルGPU（A100 80GB）**: バッチサイズ 16 - 24
- **マルチGPU**: バッチサイズを適宜スケール

### フルファインチューニング

フルファインチューニングにはかなり多くのメモリが必要です。7Bパラメータモデルの場合：

- **シングルGPU（A100 80GB）**: バッチサイズ 2 - 4（勾配チェックポイント使用時）
- **マルチGPU（8x A100）**: 安定した学習のために推奨

## 出力

学習スクリプトは以下を保存します：

1. **チェックポイント**: `output_dir/checkpoint-{step}/` に保存
   - LoRAアダプター重み（LoRA使用時）
   - 完全なモデル重み（LoRA未使用時）
   - プロセッサ設定

2. **ログ**: 学習メトリクスの記録先：
   - コンソール出力
   - Weights & Biases（有効時）

## ファインチューニング済みモデルの読み込み

ファインチューニング後、推論用にモデルを読み込む：

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
    "outputs/crane_x7_finetune/checkpoint-5000"
)

# プロセッサを読み込み
processor = AutoProcessor.from_pretrained(
    "outputs/crane_x7_finetune/checkpoint-5000",
    trust_remote_code=True
)

# 推論モード
model.eval()
model.to("cuda")

# ... モデルを推論に使用 ...
```

## 高度な使い方

### カスタムデータセット指示

`finetune_config.py` でタスク指示を変更：

```python
instruction: str = "Grasp the red block and place it in the bin"
```

### 画像なしの学習

データセットに画像がない場合（状態のみ）：

```python
use_image: bool = False
```

### 勾配チェックポイント

学習速度を犠牲にしてメモリを節約：

```python
gradient_checkpointing: bool = True
```

### チェックポイントから再開

```python
resume_from_checkpoint: Optional[str] = "outputs/crane_x7_finetune/checkpoint-5000"
```

## トラブルシューティング

### メモリ不足

1. `batch_size` を減らす
2. 有効なバッチサイズを維持するため `grad_accumulation_steps` を増やす
3. `gradient_checkpointing` を有効化
4. `lora_rank` を減らす

### 学習が遅い

1. `use_flash_attention` を有効化（flash-attnのインストールが必要）
2. データ読み込みのため `num_workers` を増やす
3. PyTorch DDPでマルチGPUを使用

### データセット読み込みの問題

1. TFRecordファイルが有効か確認: `python crane_x7_dataset.py <data_root>`
2. TensorFlowがインストールされているか確認: `pip install tensorflow`
3. データ構造が期待される形式と一致しているか確認

## 参考資料

OpenVLAの詳細情報：

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA論文](https://arxiv.org/abs/2406.09246)
- [OpenVLAモデル](https://huggingface.co/openvla)

CRANE-X7ロボットについて：

- [CRANE-X7 ROS 2](https://github.com/rt-net/crane_x7_ros)
- [CRANE-X7 Description](https://github.com/rt-net/crane_x7_description)
