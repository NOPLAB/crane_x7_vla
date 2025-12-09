# CRANE-X7 ROS 2

CRANE-X7ロボットアームのROS 2 Humbleベース制御環境。実機制御、Gazeboシミュレーション、Liftシミュレーション、VLA推論をサポート。

## 目次

- [クイックスタート](#クイックスタート)
- [環境設定](#環境設定)
- [利用可能なプロファイル](#利用可能なプロファイル)
- [使用例](#使用例)
- [ROS 2パッケージ](#ros-2パッケージ)
- [ディレクトリ構成](#ディレクトリ構成)
- [トラブルシューティング](#トラブルシューティング)

## クイックスタート

```bash
# プロジェクトルートから実行
cd crane_x7_vla

# 環境設定ファイルの作成
cp .env.template .env
# .envを編集してUSBデバイス、APIキー等を設定

# 実機制御
docker compose --profile real up

# シミュレーション
docker compose --profile sim up
```

## 環境設定

### .env（プロジェクトルート）

```bash
# ディスプレイ設定
DISPLAY=:0                          # X11ディスプレイ（WSL2: :0）

# ROS 2 Domain ID
ROS_DOMAIN_ID=42                    # 0-101の範囲、同じネットワーク内で統一

# USBデバイス
USB_DEVICE=/dev/ttyUSB0             # リーダーロボット
USB_DEVICE_FOLLOWER=/dev/ttyUSB1    # フォロワーロボット

# RealSense D435カメラ
CAMERA_SERIAL=                      # カメラシリアル番号（省略可）
CAMERA2_SERIAL=                     # 2台目カメラ
USE_D435=false                      # D435を使用するか
USE_VIEWER=false                    # RVizを表示するか

# Gemini API
GEMINI_API_KEY=                     # Google Gemini Robotics-ER APIキー

# VLA推論
HF_TOKEN=                           # Hugging Faceトークン
HF_CACHE_DIR=${HOME}/.cache/huggingface  # モデルキャッシュ
VLA_MODEL_PATH=                     # ファインチューニング済みモデルパス
VLA_TASK_INSTRUCTION=               # タスク指示（自然言語）
VLA_DEVICE=cuda                     # cuda / cpu

# Liftシミュレーション（統一シミュレータ抽象化）
LIFT_SIMULATOR=maniskill            # maniskill / genesis
LIFT_BACKEND=cpu                    # gpu / cpu
LIFT_RENDER_MODE=none               # rgb_array / human / none
```

### Lift環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `LIFT_SIMULATOR` | シミュレータ（`maniskill`/`genesis`） | `maniskill` |
| `LIFT_BACKEND` | バックエンド（`gpu`/`cpu`） | `cpu` |
| `LIFT_RENDER_MODE` | レンダリング（`rgb_array`/`human`/`none`） | `none` |

**注意**: `rgb_array`と`human`モードはGPU必須。GPUなし環境では`cpu` + `none`を使用。

## 利用可能なプロファイル

| プロファイル | 説明 | コマンド |
|-------------|------|---------|
| `real` | 実機制御 | `docker compose --profile real up` |
| `sim` | Gazeboシミュレーション | `docker compose --profile sim up` |
| `teleop` | テレオペ（Leader+Follower） | `docker compose --profile teleop up` |
| `log` | テレオペ + カメラ + データロガー | `docker compose --profile log up` |
| `gemini` | 実機 + Gemini API | `docker compose --profile gemini up` |
| `gemini-sim` | シミュレーション + Gemini | `docker compose --profile gemini-sim up` |
| `vla` | 実機 + VLA推論（GPU） | `docker compose --profile vla up` |
| `vla-sim` | シミュレーション + VLA推論 | `docker compose --profile vla-sim up` |
| `lift` | Liftシミュレーション | `docker compose --profile lift up` |
| `lift-vla` | Lift + VLA推論 | `docker compose --profile lift-vla up` |
| `lift-logger` | Lift + データロギング | `docker compose --profile lift-logger up` |
| `rosbridge` | rosbridgeサーバー（実機） | `docker compose --profile rosbridge up` |
| `rosbridge-sim` | rosbridgeサーバー（シミュ） | `docker compose --profile rosbridge-sim up` |
| `remote-inference` | リモートGPU推論 | `docker compose --profile remote-inference up` |
| `lerobot` | LeRobot開発シェル | `docker compose --profile lerobot up` |
| `lerobot-train` | LeRobotトレーニング | `docker compose --profile lerobot-train up` |

## 使用例

### 実機制御

```bash
# 基本起動
docker compose --profile real up

# D435カメラ + RVizビューア付き
USE_D435=true USE_VIEWER=true docker compose --profile real up
```

### シミュレーション

```bash
# Gazebo起動
docker compose --profile sim up

# RVizビューア付き
USE_VIEWER=true docker compose --profile sim up
```

### テレオペレーション

2台のCRANE-X7を使用したLeader-Followerテレオペレーション。

```bash
# USBデバイスの確認
ls -la /dev/ttyUSB*

# .envでデバイスを設定
# USB_DEVICE=/dev/ttyUSB0        # リーダー
# USB_DEVICE_FOLLOWER=/dev/ttyUSB1  # フォロワー

# テレオペのみ（カメラなし）
docker compose --profile teleop up

# テレオペ + カメラ + データロガー
docker compose --profile log up

# ビューア表示を切り替え
USE_VIEWER=false docker compose --profile log up
```

### VLA推論（ローカルGPU）

ローカルGPUでVLA推論を実行。

```bash
# .envで設定
# VLA_MODEL_PATH=/workspace/vla/outputs/checkpoint-xxx
# VLA_TASK_INSTRUCTION="pick up the red object"
# VLA_DEVICE=cuda

# 実機 + VLA
docker compose --profile vla up

# シミュレーション + VLA
docker compose --profile vla-sim up
```

### Gemini API統合

Google Gemini Robotics-ER APIを使用したタスク実行。

```bash
# .envでAPIキーを設定
# GEMINI_API_KEY=your-api-key

# 実機
docker compose --profile gemini up

# シミュレーション
docker compose --profile gemini-sim up
```

### Liftシミュレーション

統一シミュレータ抽象化レイヤー（ManiSkill、Genesis対応）。

```bash
# ManiSkill（デフォルト）
docker compose --profile lift up

# Genesisシミュレータ使用
LIFT_SIMULATOR=genesis LIFT_BACKEND=gpu docker compose --profile lift up

# Lift + VLA推論
docker compose --profile lift-vla up

# Lift + データロギング
docker compose --profile lift-logger up
```

## ROS 2パッケージ

### crane_x7_bringup

統合launchファイルパッケージ。各種起動構成をまとめて管理。

| launchファイル | 説明 |
|---------------|------|
| `real.launch.py` | 実機制御（MoveIt2 + ハードウェア + ロガー） |
| `sim.launch.py` | Gazeboシミュレーション + ロガー |
| `teleop.launch.py` | テレオペ（リーダー + フォロワー） |
| `data_collection.launch.py` | カメラ + データロガー（テレオペと併用） |
| `gemini_real.launch.py` | Gemini API（実機） |
| `gemini_sim.launch.py` | Gemini API（シミュレーション） |
| `vla_real.launch.py` | VLA推論（実機） |
| `vla_sim.launch.py` | VLA推論（Gazebo） |
| `rosbridge_real.launch.py` | 実機 + rosbridge（リモートVLA用） |
| `rosbridge_sim.launch.py` | Gazebo + rosbridge（リモートVLA用） |
| `lift.launch.py` | Liftシミュレーション |
| `lift_vla.launch.py` | Lift + VLA推論 |
| `lift_logger.launch.py` | Lift + データロガー |

```bash
# 使用例
ros2 launch crane_x7_bringup real.launch.py use_d435:=true
ros2 launch crane_x7_bringup teleop.launch.py
ros2 launch crane_x7_bringup data_collection.launch.py  # 別プロセスで起動
```

### crane_x7_log

データロギングパッケージ。TFRecord形式でOXE互換データを収集。

| launchファイル | 説明 |
|---------------|------|
| `data_logger.launch.py` | データロガーノード単体 |
| `camera_viewer.launch.py` | カメラビューア（RViz）単体 |

**出力データ形式**:
- `observation/state`: 関節位置（float32、8次元）
- `observation/image`: JPEGエンコードRGB画像
- `action`: 次状態（action[t] = state[t+1]）

### crane_x7_teleop

Leader-Followerテレオペレーションパッケージ（C++実装）。

| launchファイル | 説明 |
|---------------|------|
| `teleop_leader.launch.py` | リーダーノード単体（トルクOFF） |
| `teleop_follower.launch.py` | フォロワーノード単体（トルクON） |

```bash
# 個別起動
ros2 launch crane_x7_teleop teleop_leader.launch.py
ros2 launch crane_x7_teleop teleop_follower.launch.py
```

### crane_x7_vla

VLA推論ノード。OpenVLAファインチューニング済みモデルを使用。

| launchファイル | 説明 |
|---------------|------|
| `vla_control.launch.py` | VLA推論 + ロボットコントローラ |
| `vla_inference_only.launch.py` | 推論ノードのみ（リモートGPU用） |

```bash
# VLA推論のみ
ros2 launch crane_x7_vla vla_inference_only.launch.py \
  model_path:=/path/to/checkpoint \
  task_instruction:="pick up the object"
```

**Topics**:
- Subscribe: `/camera/color/image_raw`, `/joint_states`
- Publish: `/vla/predicted_action`

### crane_x7_gemini

Google Gemini Robotics-ER API統合パッケージ。

| launchファイル | 説明 |
|---------------|------|
| `trajectory_planner.launch.py` | Geminiプランナーノード単体 |

### crane_x7_sim_gazebo

カスタムGazeboシミュレーション環境。

| launchファイル | 説明 |
|---------------|------|
| `pick_and_place.launch.py` | Pick & Place環境 |

### crane_x7_lift

Liftシミュレーション統合パッケージ（統一シミュレータ抽象化）。

| launchファイル | 説明 |
|---------------|------|
| `sim.launch.py` | Liftシミュレータノード |

## ディレクトリ構成

```
crane_x7_vla/
├── docker-compose.yml             # 統一Docker Compose設定
├── .env                           # 環境設定（.env.templateからコピー）
├── .env.template                  # 環境設定テンプレート
├── docker/                        # Docker環境
│   ├── Dockerfile.ros2            # ROS 2統合環境
│   ├── Dockerfile.inference       # リモートGPU推論
│   ├── Dockerfile.vlarl           # VLA-RL学習
│   ├── Dockerfile.lerobot         # LeRobot統合
│   ├── entrypoint-ros2.sh         # ROS 2用エントリーポイント
│   ├── entrypoint-inference.sh    # 推論用エントリーポイント
│   └── wait-for-peer.sh           # Tailscale待機スクリプト
│
├── ros2/                          # ROS 2ワークスペース
│   └── src/                       # ROS 2パッケージ
│       ├── crane_x7_ros/          # RT Corporation公式（サブモジュール）
│       ├── crane_x7_description/  # URDFモデル（サブモジュール）
│       ├── crane_x7_bringup/      # 統合launchファイル
│       ├── crane_x7_log/          # データロギング
│       ├── crane_x7_teleop/       # テレオペレーション
│       ├── crane_x7_vla/          # VLA推論ノード
│       ├── crane_x7_gemini/       # Gemini API統合
│       ├── crane_x7_sim_gazebo/   # Gazebo環境
│       └── crane_x7_lift/         # Lift統合（統一シミュレータ）
│
├── requirements.txt               # Python依存関係
└── rosdep_packages.txt            # ROS 2パッケージ依存
```

## ワークスペースビルド

コンテナ内でのROS 2ワークスペースビルド：

```bash
cd /workspace/ros2
colcon build --symlink-install
source install/setup.bash

# 特定パッケージのみ
colcon build --packages-select crane_x7_bringup

# テスト実行
colcon test --packages-select crane_x7_log
colcon test-result --verbose
```

## トラブルシューティング

### USBデバイスが認識されない

```bash
# デバイス確認
ls -la /dev/ttyUSB*

# 権限確認
sudo chmod 666 /dev/ttyUSB0

# udevルール追加（永続化）
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/99-ftdi.rules
sudo udevadm control --reload-rules
```

### RealSenseカメラが検出されない

```bash
# コンテナ内で確認
rs-enumerate-devices -s

# USB 3.0ポートに接続しているか確認
lsusb | grep Intel
```

### VLA推論が遅い

```bash
# GPU使用確認
nvidia-smi

# CUDAデバイス設定
VLA_DEVICE=cuda docker compose --profile vla up
```

### X11ディスプレイエラー

```bash
# ホストで許可
xhost +local:docker

# WSL2の場合
export DISPLAY=:0
```

### Liftシミュレータエラー

```bash
# GPU使用時にレンダリングが必要な場合
LIFT_BACKEND=gpu LIFT_RENDER_MODE=rgb_array docker compose --profile lift up

# GPUなしの場合はnoneモードを使用
LIFT_BACKEND=cpu LIFT_RENDER_MODE=none docker compose --profile lift up
```

## ライセンス

- **オリジナルコード**: MIT License (Copyright 2025 nop)
- **crane_x7_ros**: Apache License 2.0
- **crane_x7_description**: RT Corporation非商用ライセンス
