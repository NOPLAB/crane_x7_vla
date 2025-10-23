# CRANE-X7 VLA

## 概要

CRANE-X7とVLAを使用した制御プログラムです。

**主な機能:**

- CRANE-X7の実機制御とGazeboシミュレーション
- デモンストレーションデータの収集（Open X-Embodiment形式）
- OpenVLAモデルのファインチューニング
- テレオペレーション（キネステティックティーチング）モード

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

各環境変数の説明:

- `USB_DEVICE`: リーダーロボットのUSBデバイスパス（デフォルト: `/dev/ttyUSB0`）
- `USB_DEVICE_FOLLOWER`: フォロワーロボットのUSBデバイスパス（デフォルト: `/dev/ttyUSB1`）
- `DISPLAY`: X11ディスプレイ（デフォルト: `:0`）

### 2. X11の許可

```bash
xhost +
```

### 3. 実行

#### 基本制御モード

実機の場合:
```bash
docker compose --profile real up
```

シミュレータ(Gazebo)の場合:
```bash
docker compose --profile sim up
```

#### テレオペレーション（キネステティックティーチング）モード

手動でロボットを動かしてデモンストレーションを記録します。

リーダーモードのみ（手動教示、記録なし）:
```bash
docker compose --profile teleop-leader up
```

リーダーモード + データロガー（手動教示、記録あり）:
```bash
docker compose --profile teleop-leader-logger up
```

フォロワーモードのみ（2台のロボットが必要）:
```bash
docker compose --profile teleop-follower up
```

フォロワーモード + データロガー（模倣記録、2台のロボットが必要）:
```bash
docker compose --profile teleop-follower-logger up
```

リーダー + フォロワー同時実行:
```bash
docker compose --profile teleop up
```

リーダー + フォロワー + データロガー:
```bash
docker compose --profile teleop-logger up
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
├── ros2/                          # ROS 2ワークスペース
│   └── src/
│       ├── crane_x7_ros/          # CRANE-X7公式パッケージ（Gitサブモジュール）
│       │   ├── crane_x7_control/  # ハードウェア制御インターフェース
│       │   ├── crane_x7_examples/ # サンプルプログラム
│       │   ├── crane_x7_gazebo/   # Gazeboシミュレーション
│       │   └── crane_x7_moveit_config/ # MoveIt2設定
│       ├── crane_x7_description/  # URDFロボットモデル（Gitサブモジュール）
│       └── crane_x7_log/          # データロギングパッケージ
│           ├── crane_x7_log/      # ROS 2ノード実装
│           │   ├── data_logger.py # メインデータ収集ノード
│           │   ├── episode_saver.py # エピソード保存（NPZ/TFRecord）
│           │   └── tfrecord_writer.py # TFRecord変換
│           └── launch/            # 起動ファイル
│               ├── real_with_logger.launch.py # 実機 + ロガー
│               ├── demo_with_logger.launch.py # シミュレーション + ロガー
│               └── data_logger.launch.py # スタンドアローンロガー
├── vla/                           # VLAファインチューニング
│   ├── crane_x7_dataset.py        # CRANE-X7データセットローダー
│   ├── finetune_config.py         # ファインチューニング設定
│   ├── finetune.py                # カスタム学習スクリプト
│   ├── requirements.txt           # Python依存関係
│   ├── README.md                  # VLAドキュメント
│   └── openvla/                   # OpenVLA本体（Gitサブモジュール）
│       ├── prismatic/             # Prismatic VLMライブラリ
│       ├── vla-scripts/           # 公式学習・デプロイスクリプト
│       └── experiments/           # ロボット評価実験
├── data/                          # データ保存ディレクトリ
│   └── tfrecord_logs/             # 収集されたエピソード
│       ├── episode_0000_YYYYMMDD_HHMMSS/
│       │   └── episode_data.tfrecord
│       └── episode_0001_YYYYMMDD_HHMMSS/
│           └── episode_data.tfrecord
├── outputs/                       # モデル出力ディレクトリ
│   └── crane_x7_finetune/         # ファインチューニング結果
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── checkpoint-1500/
├── scripts/                       # ユーティリティスクリプト
│   ├── build.sh                   # Dockerビルドスクリプト
│   ├── run.sh                     # Docker実行スクリプト
│   ├── docker/
│   │   └── vla_finetune.sh        # VLA実行ヘルパー
│   └── read_tfrecord.py           # TFRecord読み込みツール
├── docs/                          # ドキュメント
│   ├── DOCKER_USAGE.md            # Docker使用方法
│   └── SPEC.md                    # プロジェクト仕様
├── Dockerfile                     # マルチステージDockerfile
│   ├── base: ROS 2基本環境
│   ├── dev: ROS 2開発環境
│   └── vla_finetune: VLA学習環境（ROS 2と独立）
├── docker-compose.yml             # Docker Composeプロファイル
│   ├── real: 実機制御
│   ├── sim: Gazeboシミュレーション
│   ├── teleop-*: テレオペレーションモード
│   └── vla: VLAファインチューニング
├── CLAUDE.md                      # Claude Code指示
├── .env.template                  # 環境変数テンプレート
└── README.md                      # 本ファイル
```

## データ収集ワークフロー

1. **デモンストレーション収集**: テレオペレーションモードでロボットを手動操作
   ```bash
   docker compose --profile teleop-leader-logger up
   ```
   - エピソードは自動的に `data/tfrecord_logs/` に保存されます
   - エピソード長、保存形式（NPZ/TFRecord）は起動パラメータで設定可能

2. **データ形式変換**（NPZ使用時）:
   ```bash
   python3 -m crane_x7_log.tfrecord_writer episode_data.npz episode_data.tfrecord
   ```

3. **VLAファインチューニング**:
   - `vla/openvla/` ディレクトリで事前学習済みOpenVLAモデルをファインチューニング
   - LoRAおよびフルファインチューニングをサポート
   - 詳細は[vla/README.md](vla/README.md)を参照

4. **デプロイ**:
   - ファインチューニング済みモデルをREST API経由でデプロイ（`vla/openvla/vla-scripts/deploy.py`）
   - ROS 2制御スタックと統合してクローズドループマニピュレーション

## ライセンス

### このリポジトリのオリジナルコード

- **プロジェクト全体**: MIT License (Copyright 2025 nop)
- **crane_x7_log**: MIT License
- **crane_x7_vla**: MIT License
- **crane_x7_teleop**: MIT License
- **VLAファインチューニングスクリプト**: MIT License

### 外部/サードパーティパッケージ（Gitサブモジュール）

- **crane_x7_ros** (RT Corporation): Apache License 2.0
- **crane_x7_description** (RT Corporation): RT Corporation非商用ライセンス
  - 研究・内部使用のみ許可
  - 商用利用にはRT Corporationからの事前許可が必要
- **OpenVLA**: MIT License (コード)
  - 事前学習済みモデルには別途制限あり（例: Llama-2ライセンス）

**重要**: RT Corporationのパッケージ（`crane_x7_ros`, `crane_x7_description`）は、このリポジトリのオリジナルコードとは異なるライセンスです。使用前に各LICENSEファイルを確認してください。

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

