# crane_x7_log

VLAファインチューニング用のOXE（Open X-Embodiment）フォーマットでCRANE-X7ロボットアームのデータを記録するパッケージ。

## 概要

このパッケージは、OpenVLAのようなVision-Language-Action（VLA）モデルのファインチューニングに適した、Open X-Embodimentデータセット仕様に準拠した形式でCRANE-X7のマニピュレーションデータを記録するROS 2ノードを提供します。

## 機能

- **マルチモーダルデータ記録**: 関節状態、RGB画像、深度画像（オプション）
- **OXEフォーマット互換性**: VLA学習用のTFRecord出力形式
- **設定可能なエピソード**: エピソード長とデータ収集レートの調整可能
- **自動保存**: 目標長に達すると自動的にエピソードを保存

## ノード

### oxe_logger

ロボットとカメラのトピックをサブスクライブするメインデータ収集ノード。

**サブスクライブするトピック:**

- `/joint_states` (sensor_msgs/JointState): ロボット関節位置
- `/camera/color/image_raw` (sensor_msgs/Image): RGBカメラフィード
- `/camera/color/camera_info` (sensor_msgs/CameraInfo): カメラキャリブレーション
- `/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image): 深度画像（オプション）

**パラメータ:**

- `output_dir` (string, デフォルト: `/workspace/data/oxe_logs`): 出力ディレクトリ
- `episode_length` (int, デフォルト: 100): エピソードあたりのステップ数
- `use_camera` (bool, デフォルト: true): カメラ記録を有効化
- `use_depth` (bool, デフォルト: false): 深度記録を有効化

## 使い方

### ビルド

```bash
cd /workspace/ros2
colcon build --packages-select crane_x7_log --symlink-install
source install/setup.bash
```

### 実機ロボットでの実行

```bash
# データロガー付きでロボット制御を起動
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

または別々に起動：

```bash
# ロボット制御を起動
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0 use_d435:=true

# 別のターミナルでロガーを起動
ros2 launch crane_x7_log data_logger.launch.py
```

### テレオペレーションでの実行（キネステティックティーチング）

#### Leaderモード（手動教示）

Leaderロボット（手動移動のためトルクOFF）をデータロガー付きで起動：

```bash
ros2 launch crane_x7_log teleop_leader_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  output_dir:=/workspace/data/teleop_demonstrations
```

Leaderロボットは手動で動かしてタスクをデモンストレーションできます。関節状態は `/joint_states` と `/teleop/leader/state` にパブリッシュされ、データロガーによって自動的に記録されます。

#### Followerモード（模倣記録）

Leaderの動きに追従する2台目のCRANE-X7がある場合：

```bash
# このlaunchファイルはcrane_x7_teleopパッケージにあります
ros2 launch crane_x7_teleop teleop_with_logger.launch.py \
  port_name:=/dev/ttyUSB1 \
  output_dir:=/workspace/data/follower_demonstrations
```

### シミュレーションでの実行

```bash
# データロガー付きでGazeboシミュレーションを起動
ros2 launch crane_x7_log demo_with_logger.launch.py
```

または別々に起動：

```bash
# Gazeboシミュレーションを起動
ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py

# 別のターミナルでロガーを起動（カメラなし）
ros2 launch crane_x7_log data_logger.launch.py output_dir:=/workspace/data/sim_logs
```

### カスタムパラメータ

```bash
ros2 launch crane_x7_log real_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  output_dir:=/path/to/data \
  use_d435:=true
```

すべてのlaunchファイルは以下の共通パラメータを受け付けます：

- `output_dir`: ログデータを保存するディレクトリ（デフォルト: `/workspace/data/tfrecord_logs`）
- `config_file`: ロガー設定ファイルへのパス（オプション）

teleop leaderモード用：

- `port_name`: LeaderロボットのUSBポート（デフォルト: `/dev/ttyUSB0`）
- `use_d435`: RealSense D435カメラを有効化（デフォルト: `false`）

## データフォーマット

### NPZフォーマット（中間形式）

エピソードは最初に圧縮されたNumPyアーカイブ（`.npz`）として保存されます：

```
episode_0000_20250102_120000/
  └── episode_data.npz
      ├── states: (N, 8) - 7つのアーム関節 + 1つのグリッパー
      ├── actions: (N, 8) - 次の状態（1ステップシフト）
      ├── timestamps: (N,) - UNIXタイムスタンプ
      ├── images: (N, H, W, 3) - RGB画像（有効時）
      └── depths: (N, H, W) - 深度画像（有効時）
```

### TFRecordフォーマット（OXE互換）

NPZをTFRecordに変換：

```bash
python3 -m crane_x7_log.oxe_writer episode_data.npz episode_data.tfrecord
```

TFRecordの特徴量：

- `observation/state`: float32 関節位置
- `observation/image`: JPEGエンコードされたRGB画像
- `observation/depth`: float32 深度配列
- `observation/timestamp`: float32 タイムスタンプ
- `action`: float32 目標関節位置

## 依存関係

- ROS 2 Humble
- Pythonパッケージ：
  - rclpy
  - sensor_msgs
  - cv_bridge
  - numpy
  - opencv-python
  - tensorflow

## ライセンス

MIT License
