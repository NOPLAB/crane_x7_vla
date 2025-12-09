# CRANE-X7 ROS 2

CRANE-X7ロボットアームのROS 2 Humbleベース制御環境。実機制御、Gazeboシミュレーション、VLA推論、Tailscale経由のリモートGPU推論をサポート。

## 目次

- [クイックスタート](#クイックスタート)
- [環境設定](#環境設定)
- [Docker（ローカル実行）](#dockerローカル実行)
- [Docker-Remote（リモート推論）](#docker-remoteリモート推論)
- [ROS 2パッケージ](#ros-2パッケージ)
- [トラブルシューティング](#トラブルシューティング)

## クイックスタート

```bash
cd ros2/docker

# 環境設定ファイルの作成
cp .env.template .env
# .envを編集してUSBデバイス、APIキー等を設定

# 実機制御
docker compose --profile real up

# シミュレーション
docker compose --profile sim up
```

## 環境設定

### docker/.env

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
```

### docker-remote/.env（リモート推論用）

```bash
# Tailscale VPN
TS_AUTHKEY=tskey-auth-xxxxx         # 必須: Tailscale認証キー
TS_USERSPACE=false                  # TUNデバイス使用

# rosbridge WebSocket
ROSBRIDGE_PORT=9090                 # WebSocketポート
```

## Docker（ローカル実行）

ローカルマシンでの実行。GPUが搭載されている場合はVLA推論もローカルで実行可能。

### ディレクトリ

```bash
cd ros2/docker
```

### 利用可能なプロファイル

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
| `maniskill` | ManiSkillシミュレーション | `docker compose --profile maniskill up` |
| `maniskill-vla` | ManiSkill + VLA推論 | `docker compose --profile maniskill-vla up` |
| `maniskill-logger` | ManiSkill + データロギング | `docker compose --profile maniskill-logger up` |

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

## Docker-Remote（リモート推論）

TailscaleVPN経由でリモートGPUサーバーのVLA推論を利用。ロボット側にGPUがなくても高性能な推論が可能。

### アーキテクチャ

```
┌─────────────────────────────────────┐
│         リモートGPUサーバー           │
│  ┌─────────────────────────────┐   │
│  │   VLA推論コンテナ            │   │
│  │   - OpenVLA/OpenPI          │   │
│  │   - roslibpy (WebSocket)    │   │
│  └──────────┬──────────────────┘   │
│             │ Tailscale VPN         │
└─────────────┼───────────────────────┘
              │ TCP:9090 (rosbridge)
              │
┌─────────────┼───────────────────────┐
│             ▼                       │
│  ┌─────────────────────────────┐   │
│  │   ローカルロボット側          │   │
│  │   - Tailscale sidecar       │   │
│  │   - rosbridge_server        │   │
│  │   - CRANE-X7制御             │   │
│  │   - RealSense D435          │   │
│  └─────────────────────────────┘   │
│            ロボット側マシン          │
└─────────────────────────────────────┘
```

### セットアップ

#### 1. Tailscale認証キーの取得

1. [Tailscale Admin Console](https://login.tailscale.com/admin/settings/keys) にアクセス
2. "Generate auth key" をクリック
3. "Reusable" と "Ephemeral" にチェック
4. 生成されたキーを控える

#### 2. ローカル側（ロボット）の設定

```bash
cd ros2/docker-remote

# 環境設定
cp .env.template .env

# .envを編集
# TS_AUTHKEY=tskey-auth-xxxxx  # Tailscale認証キー
# USB_DEVICE=/dev/ttyUSB0
# ROSBRIDGE_PORT=9090
```

#### 3. リモートGPUサーバーの設定

リモートサーバーでVLA推論コンテナを準備：

```bash
# vla/ディレクトリでDockerイメージをビルド
cd vla
docker build -f Dockerfile.openvla -t crane_x7_vla_openvla .
```

### 実行手順

#### ステップ1: ローカル側でrosbridge起動

```bash
cd ros2/docker-remote

# 実機 + rosbridge
docker compose --profile real up

# または、シミュレーション + rosbridge
docker compose --profile sim up

# RVizビューア付き
docker compose --profile real-viewer up
docker compose --profile sim-viewer up
```

起動後、Tailscaleネットワークに `crane-x7-local` として参加。

#### ステップ2: リモートGPUサーバーでVLA推論

```bash
# リモートサーバーで実行
docker run --gpus all -it --rm \
  -e TS_AUTHKEY=tskey-auth-xxxxx \
  -e ROSBRIDGE_HOST=crane-x7-local \
  -e ROSBRIDGE_PORT=9090 \
  -e VLA_MODEL_PATH=openvla/openvla-7b \
  -e VLA_TASK_INSTRUCTION="pick up the red block" \
  -e VLA_DEVICE=cuda \
  -e HF_TOKEN=hf_xxxxx \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --cap-add NET_ADMIN \
  --device /dev/net/tun \
  crane_x7_vla_openvla \
  python -m crane_x7_vla.inference.rosbridge_client
```

### プロファイル一覧（docker-remote）

| プロファイル | 説明 | コマンド |
|-------------|------|---------|
| `real` | 実機 + rosbridge | `docker compose --profile real up` |
| `real-viewer` | 実機 + rosbridge + RViz | `docker compose --profile real-viewer up` |
| `sim` | シミュレーション + rosbridge | `docker compose --profile sim up` |
| `sim-viewer` | シミュレーション + rosbridge + RViz | `docker compose --profile sim-viewer up` |

### 通信フロー

1. **画像・関節状態の送信**: ローカル → rosbridge → WebSocket → リモート
2. **VLA推論実行**: リモートGPUでモデル推論
3. **アクション受信**: リモート → WebSocket → rosbridge → ローカル
4. **ロボット制御**: 予測アクションに基づいて軌跡実行

### Tailscaleホスト名

| ホスト名 | 場所 | 役割 |
|---------|------|------|
| `crane-x7-local` | ローカル（ロボット側） | rosbridge_server |
| `crane-x7-remote` | リモートGPUサーバー | VLA推論 |

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
| `maniskill.launch.py` | ManiSkillシミュレーション |
| `maniskill_vla.launch.py` | ManiSkill + VLA推論 |
| `maniskill_logger.launch.py` | ManiSkill + データロガー |

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

### crane_x7_sim_maniskill

ManiSkillシミュレーション統合パッケージ。

| launchファイル | 説明 |
|---------------|------|
| `sim_only.launch.py` | ManiSkill環境単体 |

## ディレクトリ構成

```
ros2/
├── docker/                     # ローカル実行用Docker
│   ├── docker-compose.yml      # メインコンポーズファイル
│   ├── Dockerfile              # ROS 2 Humble + VLAイメージ
│   ├── entrypoint.sh           # 起動スクリプト
│   └── .env.template           # 環境設定テンプレート
│
├── docker-remote/              # リモート推論用Docker
│   ├── docker-compose.yml      # Tailscale + rosbridge構成
│   ├── Dockerfile              # 軽量VLA推論イメージ
│   ├── entrypoint-rosbridge.sh # rosbridge起動スクリプト
│   ├── wait-for-peer.sh        # Tailscaleピア待機
│   └── .env.template           # リモート環境設定
│
├── src/                        # ROS 2パッケージ
│   ├── crane_x7_ros/           # RT Corporation公式（サブモジュール）
│   ├── crane_x7_description/   # URDFモデル（サブモジュール）
│   ├── crane_x7_bringup/       # 統合launchファイル
│   ├── crane_x7_log/           # データロギング
│   ├── crane_x7_teleop/        # テレオペレーション
│   ├── crane_x7_vla/           # VLA推論ノード
│   ├── crane_x7_gemini/        # Gemini API統合
│   ├── crane_x7_sim_gazebo/    # Gazebo環境
│   └── crane_x7_sim_maniskill/ # ManiSkill統合
│
├── requirements.txt            # Python依存関係
└── rosdep_packages.txt         # ROS 2パッケージ依存
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

### Tailscale接続に失敗する

```bash
# 認証キーの有効期限を確認
# https://login.tailscale.com/admin/settings/keys

# コンテナ内でステータス確認
tailscale status

# ログ確認
docker compose logs tailscale-local
```

### VLA推論が遅い

```bash
# GPU使用確認
nvidia-smi

# CUDAデバイス設定
VLA_DEVICE=cuda docker compose --profile vla up

# バッチサイズ調整（モデル依存）
```

### rosbridge接続エラー

```bash
# ポート確認
netstat -tlnp | grep 9090

# Tailscale経由の接続テスト
tailscale ping crane-x7-local

# WebSocket接続テスト
python3 -c "import roslibpy; c = roslibpy.Ros('ws://crane-x7-local:9090'); c.run()"
```

### X11ディスプレイエラー

```bash
# ホストで許可
xhost +local:docker

# WSL2の場合
export DISPLAY=:0
```

## ライセンス

- **オリジナルコード**: MIT License (Copyright 2025 nop)
- **crane_x7_ros**: Apache License 2.0
- **crane_x7_description**: RT Corporation非商用ライセンス
