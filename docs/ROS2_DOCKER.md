# DockerによるVLAファインチューニング

このガイドでは、CRANE-X7データでOpenVLAをファインチューニングするためのDockerの使用方法を説明します。

## 前提条件

- GPU対応Docker(nvidia-docker2)
- CUDA 12.4以降に対応したNVIDIA GPU
- 最低16GBのGPUメモリ(推奨:快適なトレーニングのために40GB以上)

## クイックスタート

### 1. VLA Dockerイメージのビルド

```bash
# VLAファインチューニングイメージをビルド
docker compose build vla_finetune
```

これにより、以下を含むDockerイメージが作成されます:
- CUDA 12.4
- CUDA対応のPyTorch 2.2.0
- OpenVLAの依存関係
- Flash Attention 2(ビルドが成功した場合)

### 2. インタラクティブコンテナの実行

```bash
# インタラクティブコンテナを起動
docker compose --profile vla run --rm vla_finetune

# コンテナ内でデータセット読み込みをテスト
python3 vla/crane_x7_dataset.py data/tfrecord_logs

# ファインチューニングを実行
cd vla
python3 finetune.py
```

### 3. ヘルパースクリプトの使用

```bash
# データセット読み込みのテスト
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh test-dataset

# シングルGPUトレーニング
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train

# マルチGPUトレーニング(2 GPU)
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train-multi-gpu 2
```

## 高度な使用方法

### カスタムトレーニングパラメータ

```bash
docker compose --profile vla run --rm vla_finetune bash -c "
  cd vla && python3 finetune.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --lora_rank 64 \
    --use_wandb \
    --wandb_project my-crane-x7
"
```

### マルチGPUトレーニング

```bash
docker compose --profile vla run --rm vla_finetune bash -c "
  cd vla && torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune.py \
    --batch_size 8 \
    --learning_rate 5e-4
"
```

### 追加ディレクトリのマウント

`docker-compose.yml`を編集してボリュームマウントを追加:

```yaml
volumes:
  - type: bind
    source: "./my_custom_data"
    target: "/workspace/custom_data"
```

## コンテナ内のディレクトリ構造

```
/workspace/
├── vla/                    # ファインチューニングスクリプト(マウント)
│   ├── finetune.py
│   ├── crane_x7_dataset.py
│   ├── finetune_config.py
│   └── README.md
├── data/                   # トレーニングデータ(マウント)
│   └── tfrecord_logs/
│       ├── episode_0000_*/
│       ├── episode_0001_*/
│       └── ...
└── outputs/                # チェックポイントとログ(マウント)
    └── crane_x7_finetune/
```

## GPU設定

docker-compose.ymlは利用可能な全GPUを使用するように設定されています:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all          # 全GPUを使用
          capabilities: [gpu]
```

特定のGPUを使用するには、`count`を変更:
- `count: 1` - 1つのGPUを使用
- `count: 2` - 2つのGPUを使用
- `count: all` - 利用可能な全GPUを使用

または`CUDA_VISIBLE_DEVICES`環境変数を使用:

```bash
docker compose --profile vla run --rm \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  vla_finetune
```

## 共有メモリの設定

コンテナはPyTorch DataLoaderのマルチプロセッシングのために16GBの共有メモリ(`shm_size: '16gb'`)で設定されています。共有メモリエラーが発生した場合は、`docker-compose.yml`でこの値を増やしてください:

```yaml
shm_size: '32gb'  # 必要に応じて増やす
```

## トレーニングの監視

### Weights & Biasesの使用

```bash
# W&Bにログイン(初回セットアップのみ)
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

### ログの表示

トレーニングログは標準出力に出力されます。ログをファイルに保存するには:

```bash
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train 2>&1 | tee training.log
```

## トラブルシューティング

### メモリ不足

1. `finetune_config.py`またはCLIでバッチサイズを減らす:
   ```bash
   --batch_size 4
   ```

2. 勾配チェックポインティングを有効化:
   ```python
   gradient_checkpointing: bool = True
   ```

3. LoRAランクを減らす:
   ```bash
   --lora_rank 16
   ```

### Flash Attentionのビルド失敗

Flash Attention 2はオプションです。ビルドが失敗してもコンテナは継続して動作します。トレーニングは少し遅くなりますが、機能は問題ありません。

コンテナ内で手動インストールする場合:

```bash
docker compose --profile vla run --rm vla_finetune bash
# コンテナ内で:
pip3 install packaging ninja
pip3 install flash-attn==2.5.5 --no-build-isolation
```

### CUDAバージョンの不一致

DockerfileはCUDA 12.4を使用しています。ホストに異なるCUDAバージョンがある場合は、Dockerfileのベースイメージを変更してください:

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS vla  # CUDAバージョンを変更
```

その後、再ビルド:
```bash
docker compose build vla_finetune --no-cache
```

### パーミッションの問題

マウントされたボリュームでパーミッションエラーが発生した場合:

```bash
# 現在のユーザーとしてコンテナを実行
docker compose --profile vla run --rm --user $(id -u):$(id -g) vla_finetune
```

## 開発ワークフロー

1. **データ収集**: ROS2コンテナを使用してデモンストレーションデータを収集
   ```bash
   docker compose --profile real up  # または --profile sim
   ```

2. **ファインチューニング**: VLAコンテナに切り替えてトレーニング
   ```bash
   docker compose --profile vla run --rm vla_finetune
   ```

3. **推論**: ファインチューニングされたモデルをロボット制御に読み込み
   ```bash
   # (実装はデプロイ設定に依存)
   ```

## リソース要件

### 最小要件

- GPU: 16GB VRAMのNVIDIA GPU(例: V100、RTX 4090)
- RAM: 32GBのシステムRAM
- ストレージ: 50GBの空き容量

### 推奨要件

- GPU: NVIDIA A100 40GB/80GB
- RAM: 64GB以上のシステムRAM
- ストレージ: 100GB以上の空き容量(データセットとチェックポイント用)

### マルチGPUスケーリング

- 1x A100 (40GB): バッチサイズ ~8-12(LoRA使用時)
- 2x A100 (40GB): バッチサイズ ~16-24(LoRA使用時)
- 4x A100 (40GB): バッチサイズ ~32-48(LoRA使用時)

## 次のステップ

ファインチューニング完了後:

1. チェックポイントは`outputs/crane_x7_finetune/checkpoint-*/`に保存されます
2. ファインチューニングされたモデルを推論用に読み込みます(`vla/README.md`を参照)
3. ロボット制御スタックと統合します
