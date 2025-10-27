# crane_x7_log

CRANE-X7ロボットアームの操作データをRLDS（Robot Learning Dataset Standard）形式で記録し、OpenVLAなどのVLAモデルのファインチューニングに活用するためのパッケージです。

## 概要

このパッケージは、CRANE-X7の操作データをOpenVLAをはじめとするVision-Language-Action（VLA）モデルで学習できる形で記録します。出力形式は**RLDS（Robot Learning Dataset Standard）**に準拠しており、Open X-Embodimentプロジェクトで採用されている標準規格に沿っています。

## 機能

- **マルチモーダルデータ記録**: ロボットの関節状態、RGB画像、深度画像（オプション）を同時に記録
- **RLDS形式互換**: OpenVLAの学習で直接利用できるTFRecord形式で出力
- **言語インストラクション対応**: 各エピソードにタスクの説明を紐付け
- **データセット統計自動計算**: アクション正規化に必要な統計情報（平均、標準偏差、最小値、最大値、パーセンタイルなど）を自動算出
- **柔軟なエピソード設定**: エピソード長やデータ収集レートを用途に応じて調整可能
- **自動保存**: 設定したステップ数に達すると、エピソードを自動的に保存
- **音声通知**: エピソード開始・完了・残り時間を音声で通知（TTS使用）

## ノード

### oxe_logger

ロボットの関節状態とカメラ画像をリアルタイムで収集するメインノードです。

**サブスクライブするトピック:**

- `/joint_states` (sensor_msgs/JointState): ロボットの関節角度情報
- `/task/language_instruction` (std_msgs/String): タスクの自然言語による説明（VLAで必須）
- `/camera/color/image_raw` (sensor_msgs/Image): RGBカメラからの映像
- `/camera/color/camera_info` (sensor_msgs/CameraInfo): カメラのキャリブレーション情報
- `/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image): 深度画像（使用する場合）

**主要パラメータ:**

- `output_dir` (string, デフォルト: `/workspace/data/tfrecord_logs`): 記録データの保存先ディレクトリ
- `episode_length` (int, デフォルト: 1000): 1エピソードに含めるステップ数
- `inter_episode_delay` (float, デフォルト: 5.0): エピソード間の待機時間（秒）。環境やロボットのリセットのための時間を確保
- `collection_rate` (float, デフォルト: 30.0): データ収集の周波数（Hz）
- `save_format` (string, デフォルト: `tfrecord`): 保存形式（`npz` または `tfrecord`）
- `use_camera` (bool, デフォルト: true): RGB画像の記録を有効化
- `use_depth` (bool, デフォルト: false): 深度画像の記録を有効化

**RLDS設定パラメータ:**

- `dataset_name` (string, デフォルト: `crane_x7`): データセットを識別するための名前
- `language_instruction_topic` (string, デフォルト: `/task/language_instruction`): タスクの言語指示を受け取るトピック
- `default_language_instruction` (string, デフォルト: `manipulate the object`): トピックが利用できない場合に使用されるデフォルトのタスク記述
- `compute_dataset_statistics` (bool, デフォルト: `true`): アクションの統計情報（平均、標準偏差など）を自動計算
- `statistics_output_path` (string, デフォルト: `/workspace/data/tfrecord_logs/dataset_statistics.json`): 統計情報の保存先ファイルパス

**音声通知パラメータ:**

- `enable_voice_notifications` (bool, デフォルト: `true`): 音声通知を有効化
- `voice_language` (string, デフォルト: `en`): 音声言語（`en`, `ja`など）
- `voice_speed` (int, デフォルト: `150`): 読み上げ速度（単語/分）
- `notify_on_episode_start` (bool, デフォルト: `true`): エピソード開始時に通知
- `notify_on_episode_complete` (bool, デフォルト: `true`): エピソード完了時に通知
- `notify_time_remaining` (bool, デフォルト: `true`): 残り時間を通知
- `time_notification_intervals` (list[int], デフォルト: `[60, 30, 10]`): 残り時間通知のタイミング（秒）

## 使い方

### ビルド

まず、パッケージをビルドしてセットアップします：

```bash
cd /workspace/ros2
colcon build --packages-select crane_x7_log --symlink-install
source install/setup.bash
```

### 実機ロボットでの実行

実機のCRANE-X7でデータ収集を行う場合：

```bash
# ロボット制御とデータロガーを同時に起動
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

ロボット制御とロガーを別々に起動することも可能です：

```bash
# まずロボット制御を起動
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0 use_d435:=true

# 別のターミナルでデータロガーを起動
ros2 launch crane_x7_log data_logger.launch.py
```

### テレオペレーションでの実行（キネステティックティーチング）

#### Leaderモード（手動教示）

Leaderロボット（手動移動のためトルクOFF）をデータロガー付きで起動します：

```bash
ros2 launch crane_x7_log teleop_leader_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  use_d435:=true
```

Leaderロボットは手で直接動かすことでタスクのデモンストレーションが可能です。関節状態は `/joint_states` と `/teleop/leader/state` にパブリッシュされ、データロガーによって自動的に記録されます。

#### Followerモード（模倣記録）

Leaderの動きに追従する2台目のCRANE-X7がある場合は以下のようにします：

```bash
# このlaunchファイルはcrane_x7_teleopパッケージにあります
ros2 launch crane_x7_teleop teleop_with_logger.launch.py \
  port_name:=/dev/ttyUSB1 \
  use_d435:=true
```

### シミュレーションでの実行

Gazeboシミュレーション環境でデータ収集を行う場合：

```bash
# シミュレーションとデータロガーを同時に起動
ros2 launch crane_x7_log demo_with_logger.launch.py
```

シミュレーションとロガーを別々に起動することも可能です：

```bash
# まずGazeboシミュレーションを起動
ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py

# 別のターミナルでデータロガーを起動（カメラなしで動作）
ros2 launch crane_x7_log data_logger.launch.py output_dir:=/workspace/data/sim_logs
```

### カスタムパラメータ

必要に応じてパラメータをカスタマイズできます：

```bash
ros2 launch crane_x7_log real_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  output_dir:=/path/to/data \
  use_d435:=true
```

以下のパラメータがすべてのlaunchファイルで共通して使用できます：

- `output_dir`: 記録データの保存先ディレクトリ（デフォルト: `/workspace/data/tfrecord_logs`）
- `config_file`: カスタム設定ファイルのパス（オプション）

teleop leaderモード固有のパラメータ：

- `port_name`: LeaderロボットのUSBポート（デフォルト: `/dev/ttyUSB0`）
- `use_d435`: RealSense D435カメラの使用を有効化（デフォルト: `false`）

## データフォーマット

このパッケージは**RLDS（Robot Learning Dataset Standard）形式**でデータを出力します。OpenVLAをはじめとするVLAモデルで直接利用できる形式です。

### TFRecordフォーマット（RLDS互換、推奨）

デフォルト設定（`save_format: tfrecord`）を使用すると、エピソードはRLDS準拠のTFRecordとして保存されます：

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

### NPZフォーマット（デバッグ・検証用）

`save_format: npz` に設定することで、データをNumPy形式の圧縮アーカイブとして保存できます：

```
episode_0000_20250127_120000/
  └── episode_data.npz
      ├── states: (N, 8) - 7つのアーム関節 + 1つのグリッパー
      ├── actions: (N, 8) - 次の状態（1ステップシフト）
      ├── timestamps: (N,) - UNIXタイムスタンプ
      ├── images: (N, H, W, 3) - RGB画像（有効時）
      └── depths: (N, H, W) - 深度画像（有効時）
```

**NPZからTFRecordへの変換方法:**

NPZ形式で保存したデータは、後からTFRecord形式に変換できます：

```bash
python3 -m crane_x7_log.tfrecord_writer \
  episode_data.npz \
  episode_data.tfrecord \
  crane_x7 \
  "pick up the red cube"
```

### データセット統計

`compute_dataset_statistics: true` に設定すると、収集したすべてのエピソードから統計情報が自動的に計算されます：

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

この統計情報は、OpenVLAでアクションを正規化する際（`BOUNDS_Q99`など）に使用されます。

## 音声通知

データ収集中のフィードバックを提供するため、エピソードのイベントを音声で通知する機能が実装されています。

### 通知タイミング

- **エピソード開始**: 新しいエピソードの記録が開始されたとき
- **エピソード完了**: エピソードが完了し、データが保存されたとき
- **残り時間**: エピソード終了前の特定の時間（デフォルト: 60秒前、30秒前、10秒前）

### カスタマイズ

音声通知は設定ファイルでカスタマイズできます：

```yaml
# logger_config.yaml
data_logger:
  ros__parameters:
    # 音声通知を無効化
    enable_voice_notifications: false

    # 日本語で通知（espeak-ngが日本語に対応している場合）
    voice_language: "ja"

    # 読み上げ速度を調整
    voice_speed: 120  # ゆっくり読む

    # 残り時間の通知タイミングを変更
    time_notification_intervals: [120, 60, 30, 10]  # 2分前、1分前、30秒前、10秒前
```

### Docker設定

音声出力を有効にするため、Docker Composeの設定に以下が含まれています：

- `/dev/snd`: ALSAオーディオデバイス
- `/run/user/1000/pulse`: PulseAudioソケット
- `audio`グループへのアクセス

音声が再生されない場合は、ホスト側のPulseAudioが起動していることを確認してください。

## 言語インストラクションの設定

VLAモデルでは、各エピソードに**タスクの言語記述**が必要です。以下の3つの方法で設定できます：

### 方法1: ROSトピック経由（推奨）

別のノードから `/task/language_instruction` トピックにタスクの説明をパブリッシュします：

```bash
# 別のターミナルで実行
ros2 topic pub /task/language_instruction std_msgs/String "data: 'pick up the red cube'"
```

Pythonスクリプトから送信する場合：

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

### 方法2: デフォルトインストラクションを使用

トピックから情報が得られない場合、`default_language_instruction` パラメータの値が自動的に使用されます：

```yaml
# logger_config.yaml
data_logger:
  ros__parameters:
    default_language_instruction: "grasp and manipulate objects on the table"
```

### 方法3: 収集後に手動で設定

データ収集後、NPZファイルをTFRecordに変換する際にタスク記述を指定できます：

```bash
python3 -m crane_x7_log.tfrecord_writer \
  episode_0000_20250127_120000/episode_data.npz \
  episode_0000_20250127_120000/episode_data.tfrecord \
  crane_x7 \
  "pick up the blue block"
```

## OpenVLAとの統合

### ステップ1: データ収集

実機でデモンストレーションを収集します：

```bash
# ロボット制御とデータロガーを起動
ros2 launch crane_x7_log real_with_logger.launch.py \
  port_name:=/dev/ttyUSB0 \
  use_d435:=true

# 別のターミナルでタスクの説明を送信
ros2 topic pub /task/language_instruction std_msgs/String \
  "data: 'pick up the red cube and place it in the target location'"
```

収集されたデータは `/workspace/data/tfrecord_logs/` にRLDS形式で保存されます。

### ステップ2: OpenVLA データセット登録

収集したデータセットをOpenVLAに登録します。`vla/openvla/prismatic/vla/datasets/rlds/oxe/configs.py` に以下を追加：

```python
OXE_DATASET_CONFIGS = {
    # ... 既存のデータセット ...

    "crane_x7": DatasetConfig(
        name="crane_x7",
        data_dir="/workspace/data/tfrecord_logs",
        image_obs_keys={"primary": "image_primary"},
        depth_obs_keys={"primary": "depth_primary"},
        state_obs_keys=["proprio"],
        language_key="language_instruction",
        action_encoding=ActionEncoding.JOINT_POS,
        state_encoding=StateEncoding.JOINT,
        dataset_statistics="/workspace/data/tfrecord_logs/dataset_statistics.json",
        normalization_type=NormalizationType.BOUNDS_Q99,
    ),
}
```

### ステップ3: OpenVLA ファインチューニング

登録したデータセットを使ってOpenVLAをファインチューニングします：

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

このパッケージを使用するには以下が必要です：

- ROS 2 Humble
- Pythonパッケージ：
  - rclpy（ROS 2 Pythonクライアント）
  - sensor_msgs（センサーメッセージ型）
  - cv_bridge（OpenCVとROS画像の変換）
  - numpy（数値計算）
  - opencv-python（画像処理）
  - tensorflow（TFRecordの読み書き）

## ライセンス

MIT License
