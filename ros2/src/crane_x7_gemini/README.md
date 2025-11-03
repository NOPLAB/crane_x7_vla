# crane_x7_gemini

Google Gemini Robotics-ER API integration for CRANE-X7 robot manipulation.

## 概要

このパッケージは、Google Gemini Robotics-ER 1.5 モデルをCRANE-X7ロボットアームと統合し、ビジョンベースの物体認識とマニピュレーションタスクを実行します。

## 機能

- **物体検出**: Gemini APIを使用したリアルタイム物体検出
- **バウンディングボックス検出**: 2Dバウンディングボックスでの物体位置特定
- **軌道プランニング**: Geminiの推論を使用した自然言語ベースの軌道生成
- **ピックアンドプレース**: シミュレーションおよび実機でのピックアンドプレースタスク

## 前提条件

### APIキーの設定

Google Gemini APIキーが必要です：

```bash
export GEMINI_API_KEY="your-api-key-here"
```

または、起動時にパラメータとして渡すこともできます：

```bash
ros2 launch crane_x7_gemini gemini.launch.py api_key:=your-api-key-here
```

### 依存関係

- ROS 2 Humble
- `google-genai` Python パッケージ
- OpenCV
- MoveIt2

## 使用方法

### 1. Geminiノードのみ起動

カメラ画像から物体を検出するGeminiノードを起動：

```bash
ros2 launch crane_x7_gemini gemini.launch.py
```

パラメータ：
- `api_key`: Gemini APIキー（環境変数`GEMINI_API_KEY`から取得）
- `model_id`: Geminiモデル（デフォルト: `gemini-robotics-er-1.5-preview`）
- `image_topic`: 入力カメラトピック（デフォルト: `/camera/color/image_raw`）
- `output_topic`: 検出結果トピック（デフォルト: `/gemini/detections`）
- `temperature`: モデル温度（デフォルト: `0.5`）
- `thinking_budget`: 推論バジェット（デフォルト: `0`）
- `max_objects`: 最大検出物体数（デフォルト: `10`）

### 2. 実機ロボットとGeminiノード

実機ロボットとGeminiノードを同時に起動：

```bash
ros2 launch crane_x7_gemini gemini_with_robot.launch.py \
  port_name:=/dev/ttyUSB0 \
  use_d435:=true
```

パラメータ：
- `port_name`: CRANE-X7のUSBポート（デフォルト: `/dev/ttyUSB0`）
- `use_d435`: RealSense D435カメラを使用（デフォルト: `true`）
- `api_key`: Gemini APIキー
- `execute_trajectory`: 計画した軌道を実行するかどうか（デフォルト: `true`）

### 3. シミュレーションでピックアンドプレース

GazeboシミュレーションとGeminiノードを使用したピックアンドプレースタスク：

```bash
ros2 launch crane_x7_gemini gemini_pick_and_place_sim.launch.py
```

パラメータ：
- `api_key`: Gemini APIキー
- `auto_start_pick_place`: ピックアンドプレースタスクを自動開始（デフォルト: `false`）
- `example`: ピックアンドプレース例の種類（デフォルト: `pick_and_place`）
  - `pick_and_place`: 基本的なピックアンドプレース
  - `pick_and_place_tf`: TF座標系を使用したピックアンドプレース

#### 自動起動でピックアンドプレースを実行

```bash
ros2 launch crane_x7_gemini gemini_pick_and_place_sim.launch.py \
  auto_start_pick_place:=true
```

#### 手動でピックアンドプレースを実行

シミュレーションとGeminiノードを起動した後、別のターミナルでピックアンドプレースタスクを手動で実行：

```bash
# シミュレーション起動
ros2 launch crane_x7_gemini gemini_pick_and_place_sim.launch.py

# 別のターミナルで
ros2 launch crane_x7_examples example.launch.py \
  example:=pick_and_place \
  use_sim_time:=true
```

### 4. 物体検出のみ

物体検出ノードを個別に起動：

```bash
ros2 launch crane_x7_gemini object_detector.launch.py
```

### 5. 軌道プランニング

自然言語指示から軌道を生成：

```bash
ros2 launch crane_x7_gemini trajectory_planner.launch.py
```

別のターミナルから指示を送信：

```bash
ros2 topic pub /gemini/prompt std_msgs/msg/String \
  "data: 'Pick up the red cube and place it on the table'"
```

## トピック

### Subscribe
- `/camera/color/image_raw` (sensor_msgs/Image): カメラ画像入力
- `/gemini/prompt` (std_msgs/String): 軌道プランニング用の自然言語指示

### Publish
- `/gemini/detections` (std_msgs/String): JSON形式の物体検出結果
- `/gemini/trajectory` (カスタム): 生成された軌道

## 設定

設定ファイル: `config/gemini_config.yaml`

```yaml
gemini_node:
  ros__parameters:
    model_id: "gemini-robotics-er-1.5-preview"
    temperature: 0.5
    thinking_budget: 0
    max_objects: 10
```

## 例

### 物体検出の例

```python
# Geminiノードが起動している状態で
ros2 topic echo /gemini/detections
```

出力例：
```json
[
  {
    "point": [500, 300],
    "label": "red cube"
  },
  {
    "point": [600, 400],
    "label": "blue cylinder"
  }
]
```

### バウンディングボックス検出の例

GeminiノードのAPIを使用：
```python
from crane_x7_gemini.gemini_node import GeminiNode
node = GeminiNode()
boxes = node.get_bounding_boxes()
```

出力例：
```json
[
  {
    "box_2d": [200, 150, 600, 450],
    "label": "red cube"
  }
]
```

## ライセンス

MIT License - Copyright 2025 nop

## 参考資料

- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [CRANE-X7 Documentation](https://github.com/rt-net/crane_x7_ros)
