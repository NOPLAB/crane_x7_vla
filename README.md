# CRANE-X7

ロボット設計制作論実習3

## 概要

VLAを使ってマニュピレータを制御する。

## 必須

- Native Linux
- Docker

## リポジトリのクローン

```bash
git clone --recursive https://github.com/NOPLAB/crane_x7_vla
```

## 実行方法

### 1. `.env`の作成

.env.templateからコピーして作成

各環境変数の説明
- `USB_DEVICE` ホストに認識されているのUSBデバイスのパス

### 2. X11の許可

```bash
xhost +
```

### 3. 実行

実機の場合
```bash
docker compose --profile real up
```

シミュレータ(Gazebo)の場合
```bash
docker compose --profile sim up
```

## VLAファインチューニング

### 1. VLA Dockerイメージのビルド

```bash
docker compose build vla_finetune
```

### 2. データセットの確認

```bash
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh test-dataset
```

### 3. ファインチューニングの実行

シングルGPUの場合:
```bash
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train
```

マルチGPU（2台）の場合:
```bash
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train-multi-gpu 2
```

対話型シェルで実行:
```bash
docker compose --profile vla run --rm vla_finetune bash

# コンテナ内で
cd vla
python3 finetune.py --batch_size 16 --learning_rate 1e-4
```

### 4. 詳細なドキュメント

- [VLAファインチューニングの詳細](vla/README.md)
- [Docker使用方法](vla/docker_usage.md)

### 5. 出力

ファインチューニングの結果は`outputs/crane_x7_finetune/`に保存されます:
```
outputs/
└── crane_x7_finetune/
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── checkpoint-1500/
```

## 開発

### Docker

1. ビルド

```bash
./scripts/build.sh
```

2. 実行

実機で実行
```bash
./scripts/run.sh real

# colcon build --symlink-install
# source install/setup.bash
```

シミュレータ(Gazebo)で実行
```bash
./scripts/run.sh sim

# colcon build --symlink-install
# source install/setup.bash
```

## プロジェクト構成

```
crane_x7_vla/
├── ros2/                       # ROS2パッケージ
│   └── src/
│       ├── crane_x7_ros/       # CRANE-X7公式パッケージ
│       ├── crane_x7_description/
│       └── crane_x7_log/       # データロギング用パッケージ
├── vla/                        # VLAファインチューニング
│   ├── finetune.py            # メインスクリプト
│   ├── crane_x7_dataset.py    # データセットローダー
│   ├── finetune_config.py     # 設定
│   ├── README.md              # 詳細ドキュメント
│   ├── docker_usage.md        # Docker使用方法
│   └── requirements.txt       # Python依存関係
├── data/                       # ロボットのデモンストレーションデータ
│   └── tfrecord_logs/
│       ├── episode_0000_*/
│       └── episode_0001_*/
├── outputs/                    # ファインチューニング出力
│   └── crane_x7_finetune/
│       └── checkpoint-*/
├── scripts/
│   └── docker/
│       └── vla_finetune.sh    # VLA実行ヘルパースクリプト
├── Dockerfile                  # マルチステージDockerfile
│   ├── base: ROS2環境
│   ├── dev: ROS2開発環境
│   └── vla: VLAファインチューニング環境（ROS2と独立）
└── docker-compose.yml
    ├── real: 実機制御
    ├── sim: Gazeboシミュレーション
    └── vla: VLAファインチューニング
```

## 参考情報

### RT Corporation (CRANE-X7)
- https://github.com/rt-net/crane_x7
- https://github.com/rt-net/crane_x7_ros
- https://github.com/rt-net/crane_x7_Hardware
- https://github.com/rt-net/crane_x7_samples

### OpenVLA
- [OpenVLA GitHub](https://github.com/openvla/openvla)

