# CRANE-X7 VLA

## 概要

CRANE-X7ロボットアームとVLAを使用した制御プログラムです。

**主な機能:**

- CRANE-X7の実機制御とGazeboシミュレーション
- RLDS形式でのデモンストレーションデータ収集
- 言語インストラクション対応のデータロギング
- OpenVLAモデルのファインチューニング
- テレオペレーションモード（キネステティックティーチング）
- RealSenseカメラ対応（RGB + 深度画像）

## 必要なもの

- Native Linux
- Docker

## リポジトリのクローン

```bash
git clone --recursive https://github.com/NOPLAB/crane_x7_vla
```

## 実行方法

### 1. `.env`の作成

`ros2/.env.template`をコピーして`ros2/.env`を作成します。

```bash
cd ros2
cp .env.template .env
# 必要に応じて編集
```

各環境変数の説明:

- `USB_DEVICE`: リーダーロボットのUSBデバイスパス、デフォルトは `/dev/ttyUSB0`
- `USB_DEVICE_FOLLOWER`: フォロワーロボットのUSBデバイスパス、デフォルトは `/dev/ttyUSB1`
- `DISPLAY`: X11ディスプレイ、デフォルトは `:0`

### 2. X11の許可

```bash
xhost +
```

### 3. 実行

#### 基本制御モード

実機:
```bash
docker compose -f ros2/docker-compose.yml --profile real up
```

シミュレーション:
```bash
docker compose -f ros2/docker-compose.yml --profile sim up
```

実機 + カメラビューワー（RealSense映像表示）:
```bash
docker compose -f ros2/docker-compose.yml --profile real-viewer up
```

#### テレオペレーションモード

手動でロボットを動かしてデモンストレーションを記録します。

リーダーモードのみ - 手動教示、記録なし:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-leader up
```

リーダーモード + データロガー - 手動教示、記録あり:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-leader-logger up
```

リーダーモード + データロガー + カメラビューワー - 手動教示、記録あり、映像表示:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-leader-viewer up
```

フォロワーモードのみ - 2台のロボットが必要:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-follower up
```

フォロワーモード + カメラビューワー - 2台のロボット、フォロワー側の映像表示:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-follower-viewer up
```

フォロワーモード + データロガー - 模倣記録、2台のロボットが必要:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-follower-logger up
```

リーダー + フォロワー同時実行:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop up
```

リーダー + フォロワー + データロガー:
```bash
docker compose -f ros2/docker-compose.yml --profile teleop-logger up
```

## VLAファインチューニング

### 1. VLA Dockerイメージのビルド

```bash
docker compose -f ros2/docker-compose.yml build vla_finetune
```

### 2. データセットの確認

```bash
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune \
  /workspace/ros2/scripts/docker/vla_finetune.sh test-dataset
```

### 3. ファインチューニングの実行

シングルGPU:
```bash
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune \
  /workspace/ros2/scripts/docker/vla_finetune.sh train
```

マルチGPU - 2台を使う場合:
```bash
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune \
  /workspace/ros2/scripts/docker/vla_finetune.sh train-multi-gpu 2
```

対話型シェルで実行:
```bash
docker compose -f ros2/docker-compose.yml --profile vla run --rm vla_finetune bash

# コンテナ内で
cd vla
python3 finetune.py --batch_size 16 --learning_rate 1e-4
```

### 4. 詳細ドキュメント

- [VLAファインチューニングの詳細](vla/README.md)
- [Docker使用方法](docs/DOCKER_USAGE.md)
- [プロジェクト仕様](docs/SPEC.md)
- [CLAUDE.md](CLAUDE.md) - Claude Codeへの指示

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
./ros2/scripts/build.sh
```

2. 実行

実機:
```bash
./ros2/scripts/run.sh real

# colcon build --symlink-install
# source install/setup.bash
```

シミュレーション:
```bash
./ros2/scripts/run.sh sim

# colcon build --symlink-install
# source install/setup.bash
```

## プロジェクト構成

```
crane_x7_vla/
├── ros2/                          # ROS 2ワークスペース
│   └── src/
│       ├── crane_x7_ros/          # CRANE-X7公式パッケージ - Gitサブモジュール
│       │   ├── crane_x7_control/  # ハードウェア制御インターフェース
│       │   ├── crane_x7_examples/ # サンプルプログラム
│       │   ├── crane_x7_gazebo/   # Gazeboシミュレーション
│       │   └── crane_x7_moveit_config/ # MoveIt2設定
│       ├── crane_x7_description/  # URDFロボットモデル - Gitサブモジュール
│       ├── crane_x7_log/          # データロギングパッケージ - RLDS形式対応
│       │   ├── crane_x7_log/      # ROS 2ノード実装
│       │   │   ├── data_logger.py # メインデータ収集ノード
│       │   │   ├── episode_saver.py # エピソード保存 NPZ/TFRecord
│       │   │   └── tfrecord_writer.py # TFRecord変換
│       │   └── launch/            # 起動ファイル
│       │       ├── real_with_logger.launch.py # 実機 + ロガー
│       │       ├── demo_with_logger.launch.py # シミュレーション + ロガー
│       │       ├── teleop_leader_with_logger.launch.py # テレオペ + ロガー
│       │       └── data_logger.launch.py # スタンドアローンロガー
│       └── crane_x7_teleop/       # テレオペレーションパッケージ
│           ├── src/               # C++実装
│           │   └── teleop_hardware_node.cpp # トルクOFF制御ノード
│           └── launch/            # 起動ファイル
│               ├── teleop_leader.launch.py # リーダーロボット
│               └── teleop_follower.launch.py # フォロワーロボット
├── vla/                           # VLAファインチューニング
│   ├── crane_x7_dataset.py        # CRANE-X7データセットローダー
│   ├── finetune_config.py         # ファインチューニング設定
│   ├── finetune.py                # カスタム学習スクリプト
│   ├── test_crane_x7_loader.py    # RLDS形式データ検証スクリプト
│   ├── requirements.txt           # Python依存関係
│   ├── README.md                  # VLAドキュメント
│   └── openvla/                   # OpenVLA本体 - Gitサブモジュール
│       ├── prismatic/             # Prismatic VLMライブラリ
│       │   └── vla/datasets/rlds/oxe/  # CRANE-X7データセット統合
│       ├── vla-scripts/           # 公式学習・デプロイスクリプト
│       └── experiments/           # ロボット評価実験
├── data/                          # データ保存ディレクトリ
│   └── tfrecord_logs/             # 収集されたエピソード
│       ├── episode_0000_YYYYMMDD_HHMMSS/
│       │   └── episode_data.tfrecord
│       ├── episode_0001_YYYYMMDD_HHMMSS/
│       │   └── episode_data.tfrecord
│       └── dataset_statistics.json # データセット統計情報
├── ros2/                          # ROS 2ワークスペース（Docker設定含む）
│   ├── Dockerfile                 # マルチステージDockerfile
│   │   ├── base: ROS 2基本環境
│   │   ├── dev: ROS 2開発環境
│   │   └── vla: VLA学習環境、ROS 2と独立
│   ├── docker-compose.yml         # Docker Composeプロファイル
│   │   ├── real: 実機制御
│   │   ├── sim: Gazeboシミュレーション
│   │   ├── teleop-*: テレオペレーションモード
│   │   └── vla: VLAファインチューニング
│   ├── .env.template              # 環境変数テンプレート
│   ├── .dockerignore              # Dockerビルド除外設定
│   ├── scripts/                   # ユーティリティスクリプト
│   │   ├── build.sh               # Dockerビルドスクリプト
│   │   ├── run.sh                 # Docker実行スクリプト
│   │   ├── read_tfrecord.py       # TFRecord読み込みツール
│   │   └── docker/
│   │       ├── build-entrypoint.sh    # ビルド用エントリーポイント
│   │       ├── run-entrypoint.sh      # 実行用エントリーポイント
│   │       ├── entrypoint.sh          # 開発用エントリーポイント
│   │       └── vla_finetune.sh        # VLA実行ヘルパー
│   └── src/                       # 上記のROS 2パッケージ
├── outputs/                       # モデル出力ディレクトリ
│   └── crane_x7_finetune/         # ファインチューニング結果
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── checkpoint-1500/
├── docs/                          # ドキュメント
│   ├── DOCKER_USAGE.md            # Docker使用方法
│   └── SPEC.md                    # プロジェクト仕様
├── CLAUDE.md                      # Claude Code指示
└── README.md                      # 本ファイル
```

## データ収集ワークフロー

### 1. デモンストレーション収集

テレオペレーションモードでロボットを手動操作してデータを収集します:

```bash
docker compose -f ros2/docker-compose.yml --profile teleop-leader-logger up
```

**収集されるデータ:**
- ロボットの関節状態（7関節 + グリッパー）
- RGB画像（RealSense D435使用時）
- 深度画像（オプション）
- タスクの言語インストラクション
- タイムスタンプ

**保存形式:**
- RLDS形式のTFRecordファイル
- 自動的に `data/tfrecord_logs/` に保存
- エピソード長、収集レート、保存形式は設定ファイルで調整可能

### 2. 言語インストラクションの設定

各エピソードにタスクの説明を紐付けます:

```bash
# 別のターミナルで実行
ros2 topic pub /task/language_instruction std_msgs/String \
  "data: '赤いキューブを掴んで青いコンテナに入れる'"
```

### 3. データセット統計の確認

収集後、データセット統計が自動的に計算されます:

```bash
cat data/tfrecord_logs/dataset_statistics.json
```

### 4. VLAファインチューニング

事前学習済みOpenVLAモデルをファインチューニングします:

```bash
docker compose --profile vla run --rm vla_finetune \
  /workspace/scripts/docker/vla_finetune.sh train
```

- LoRAおよびフルファインチューニングをサポート
- 詳細は[vla/README.md](vla/README.md)を参照

### 5. デプロイ

ファインチューニング済みモデルをデプロイしてROS 2と統合します:

```bash
# REST API経由でデプロイ
python vla/openvla/vla-scripts/deploy.py
```

## ライセンス

### このリポジトリのオリジナルコード

- **プロジェクト全体**: MIT License - Copyright 2025 nop
- **crane_x7_log**: MIT License
- **crane_x7_vla**: MIT License
- **crane_x7_teleop**: MIT License
- **VLAファインチューニングスクリプト**: MIT License

### 外部/サードパーティパッケージ - Gitサブモジュール

- **crane_x7_ros** - RT Corporation: Apache License 2.0
- **crane_x7_description** - RT Corporation: RT Corporation非商用ライセンス
  - 研究・内部使用のみ許可
  - 商用利用にはRT Corporationからの事前許可が必要
- **OpenVLA**: MIT License - コード部分
  - 事前学習済みモデルには別途制限あり、例えばLlama-2ライセンスなど

**重要**: RT Corporationのパッケージ `crane_x7_ros` と `crane_x7_description` は、このリポジトリのオリジナルコードとは異なるライセンスです。使用前に各LICENSEファイルを確認してください。

## 参考情報

### RT Corporation (CRANE-X7)

- [CRANE-X7公式](https://github.com/rt-net/crane_x7)
- [CRANE-X7 ROS 2パッケージ](https://github.com/rt-net/crane_x7_ros)
- [CRANE-X7 ハードウェア](https://github.com/rt-net/crane_x7_Hardware)
- [CRANE-X7 サンプルコード](https://github.com/rt-net/crane_x7_samples)

### OpenVLA

- [OpenVLA公式サイト](https://openvla.github.io/)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA論文](https://arxiv.org/abs/2406.09246)
- [HuggingFaceモデル](https://huggingface.co/openvla)

### Open X-Embodiment

- [Open X-Embodimentプロジェクト](https://robotics-transformer-x.github.io/)

