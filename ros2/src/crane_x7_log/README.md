# crane_x7_log

OpenVLA等のVLAモデルファインチューニング用に、RLDS（Robot Learning Dataset Standard）形式でCRANE-X7ロボットアームのデータを記録するパッケージ。

## 概要

このパッケージは、OpenVLAのようなVision-Language-Action（VLA）モデルのファインチューニングに適した、**RLDS（Robot Learning Dataset Standard）形式**でCRANE-X7のマニピュレーションデータを記録するROS 2ノードを提供します。Open X-Embodimentプロジェクトで使用される標準形式に準拠しています。

## 機能

- **マルチモーダルデータ記録**: 関節状態、RGB画像、深度画像（オプション）
- **RLDS形式互換性**: OpenVLA学習に直接使用可能なTFRecord出力
- **言語インストラクション対応**: タスク記述をエピソードに関連付け
- **データセット統計自動計算**: アクション正規化用の統計情報（mean/std/min/max/percentiles）
- **設定可能なエピソード**: エピソード長とデータ収集レートの調整可能
- **自動保存**: 目標長に達すると自動的にエピソードを保存

## ノード

### oxe_logger

ロボットとカメラのトピックをサブスクライブするメインデータ収集ノード。

**サブスクライブするトピック:**

- `/joint_states` (sensor_msgs/JointState): ロボット関節位置
- `/task/language_instruction` (std_msgs/String): タスクの自然言語記述（VLA必須）
- `/camera/color/image_raw` (sensor_msgs/Image): RGBカメラフィード
- `/camera/color/camera_info` (sensor_msgs/CameraInfo): カメラキャリブレーション
- `/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image): 深度画像（オプション）

**主要パラメータ:**

- `output_dir` (string, デフォルト: `/workspace/data/oxe_logs`): 出力ディレクトリ
- `episode_length` (int, デフォルト: 100): エピソードあたりのステップ数
- `save_format` (string, デフォルト: `tfrecord`): 保存形式（`npz` または `tfrecord`）
- `use_camera` (bool, デフォルト: true): カメラ記録を有効化
- `use_depth` (bool, デフォルト: false): 深度記録を有効化

**RLDS設定パラメータ:**

- `dataset_name` (string, デフォルト: `crane_x7`): データセット識別子
- `language_instruction_topic` (string, デフォルト: `/task/language_instruction`): 言語指示トピック
- `default_language_instruction` (string, デフォルト: `manipulate the object`): デフォルトタスク記述
- `compute_dataset_statistics` (bool, デフォルト: `true`): データセット統計を計算
- `statistics_output_path` (string, デフォルト: `/workspace/data/logs/dataset_statistics.json`): 統計ファイル出力先

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

このパッケージは**RLDS（Robot Learning Dataset Standard）形式**でデータを出力します。これはOpenVLAが直接読み込める形式です。

### TFRecordフォーマット（RLDS互換、推奨）

デフォルト設定（`save_format: tfrecord`）では、エピソードはRLDS形式のTFRecordとして保存されます：

```
episode_0000_20250127_120000/
  └── episode_data.tfrecord
```

**TFRecord特徴量（ステップごと）:**

| フィールド名 | 型 | 説明 |
|------------|-----|------|
| `observation/proprio` | float32[] | 8次元固有受容覚（7関節 + 1グリッパー） |
| `observation/image_primary` | bytes | JPEGエンコードRGB画像 |
| `observation/depth_primary` | bytes | float32深度配列（オプション） |
| `observation/timestep` | int64 | ステップインデックス（0から開始） |
| `task/language_instruction` | bytes | タスクの自然言語記述（例: "赤いキューブを掴む"） |
| `action` | float32[] | 8次元アクション（次状態 = state[t+1]） |
| `dataset_name` | bytes | データセット識別子（例: "crane_x7"） |

### NPZフォーマット（デバッグ用）

`save_format: npz` に設定すると、圧縮されたNumPyアーカイブとして保存されます：

```
episode_0000_20250127_120000/
  └── episode_data.npz
      ├── states: (N, 8) - 7つのアーム関節 + 1つのグリッパー
      ├── actions: (N, 8) - 次の状態（1ステップシフト）
      ├── timestamps: (N,) - UNIXタイムスタンプ
      ├── images: (N, H, W, 3) - RGB画像（有効時）
      └── depths: (N, H, W) - 深度画像（有効時）
```

**NPZからTFRecordへの変換:**

```bash
python3 -m crane_x7_log.tfrecord_writer \
  episode_data.npz \
  episode_data.tfrecord \
  crane_x7 \
  "pick up the red cube"
```

### データセット統計

`compute_dataset_statistics: true` の場合、全エピソード終了時に統計が自動計算されます：

```json
{
  "dataset_name": "crane_x7",
  "num_transitions": 1000,
  "action": {
    "mean": [0.0, 0.5, ...],
    "std": [0.3, 0.2, ...],
    "min": [-1.5, -1.0, ...],
    "max": [1.5, 1.0, ...],
    "q01": [-1.2, -0.8, ...],
    "q99": [1.2, 0.8, ...]
  }
}
```

この統計はOpenVLAの正規化（`BOUNDS_Q99`など）に使用されます。

## 言語インストラクションの設定

VLAモデルは各エピソードに**言語タスク記述**を必要とします。設定方法は3つあります：

### 方法1: ROSトピック経由（推奨）

別のノードから `/task/language_instruction` トピックにパブリッシュ：

```bash
# 別のターミナルで
ros2 topic pub /task/language_instruction std_msgs/String "data: 'pick up the red cube'"
```

または、Pythonスクリプトから：

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

node = rclpy.create_node('instruction_publisher')
pub = node.create_publisher(String, '/task/language_instruction', 10)

msg = String()
msg.data = "pick up the red cube and place it in the blue container"
pub.publish(msg)
```

### 方法2: デフォルトインストラクション

トピックが利用できない場合、`default_language_instruction` パラメータが使用されます：

```yaml
# logger_config.yaml
data_logger:
  ros__parameters:
    default_language_instruction: "grasp and manipulate objects on the table"
```

### 方法3: エピソードごとの手動変更

収集後にNPZファイルを手動で変換：

```bash
python3 -m crane_x7_log.tfrecord_writer \
  episode_0000_20250127_120000/episode_data.npz \
  episode_0000_20250127_120000/episode_data.tfrecord \
  crane_x7 \
  "pick up the blue block"
```

## OpenVLAとの統合

### ステップ1: データ収集

```bash
# 実機でデモンストレーションを収集
ros2 launch crane_x7_log real_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  use_d435:=true

# 別のターミナルで言語インストラクションをパブリッシュ
ros2 topic pub /task/language_instruction std_msgs/String \
  "data: 'pick up the red cube and place it in the target location'"
```

データは `/workspace/data/logs/` に RLDS 形式で保存されます。

### ステップ2: OpenVLA データセット登録

`vla/openvla/prismatic/vla/datasets/rlds/oxe/configs.py` に追加：

```python
OXE_DATASET_CONFIGS = {
    # ... 既存のデータセット ...

    "crane_x7": DatasetConfig(
        name="crane_x7",
        data_dir="/workspace/data/logs",
        image_obs_keys={"primary": "image_primary"},
        depth_obs_keys={"primary": "depth_primary"},
        state_obs_keys=["proprio"],
        language_key="language_instruction",
        action_encoding=ActionEncoding.JOINT_POS,
        state_encoding=StateEncoding.JOINT,
        dataset_statistics="/workspace/data/logs/dataset_statistics.json",
        normalization_type=NormalizationType.BOUNDS_Q99,
    ),
}
```

### ステップ3: OpenVLA ファインチューニング

```bash
cd /workspace/vla/openvla

python prismatic/vla/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /workspace/data \
  --dataset_name crane_x7 \
  --run_root_dir /workspace/runs \
  --adapter_tmp_dir /workspace/adapters \
  --lora_rank 32 \
  --batch_size 16 \
  --learning_rate 5e-4 \
  --num_epochs 100
```

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
