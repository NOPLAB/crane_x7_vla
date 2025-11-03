# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## 概要

このリポジトリには、CRANE-X7ロボットアームを制御するためのROS 2 Humbleコード、およびビジョンベースのマニピュレーションタスクのためのOpenVLA（Vision-Language-Action）統合が含まれています。このプロジェクトは実機とGazeboシミュレーションの両方をサポートしています。

## アーキテクチャ

### 主要コンポーネント

1. **ROS 2ワークスペース** (`ros2/`)
   - `crane_x7_ros/`: RT CorporationのCRANE-X7用公式ROS 2パッケージ
     - `crane_x7_control`: ハードウェア制御インターフェースとUSB通信
     - `crane_x7_examples`: ロボットの機能を示すサンプルプログラム
     - `crane_x7_gazebo`: Gazeboシミュレーション環境
     - `crane_x7_moveit_config`: モーションプランニング用MoveIt2設定
   - `crane_x7_description/`: URDF/xacroロボットモデル定義
   - `crane_x7_log/`: VLAトレーニング用データロギングパッケージ
     - OXE互換形式でロボットマニピュレーションエピソードを収集
     - NPZおよびTFRecord両方の出力形式をサポート
     - 関節状態、RGB画像、オプションのデプス情報をキャプチャ
   - `crane_x7_gemini/`: Google Gemini Robotics-ER API統合パッケージ
     - ビジョンベースの物体検出と認識
     - 自然言語指示からの軌道プランニング
     - 実機とシミュレーションの両方をサポート

2. **OpenVLA** (`vla/openvla/`)
   - ロボットマニピュレーション用Vision-Language-Actionモデル
   - Prismatic VLMsに基づく
   - Embodied AIタスクのファインチューニングとデプロイメントをサポート
   - Open X-Embodimentデータセットミックスで訓練

3. **Docker環境**
   - `base`（本番環境）と`dev`（開発環境）ターゲットを持つマルチステージDockerfile
   - ROS Humbleベースイメージ（Ubuntu 22.04）
   - GUIアプリケーション（RViz、Gazebo）用X11フォワーディング

### 開発モード

リポジトリはdocker-composeプロファイルで制御される複数の実行モードをサポートしています：
- **real**: 物理的なCRANE-X7にUSB経由で接続（`/dev/ttyUSB0`）
- **real-viewer**: カメラビューア付き実機（RealSense D435ストリームを表示）
- **sim**: ハードウェアなしでGazeboシミュレーションを実行
- **gemini**: 実機ロボットでGemini Robotics-ER APIを使用した物体検出とマニピュレーション
- **gemini-sim**: GazeboシミュレーションでGemini APIを使用したピックアンドプレースタスク

## よく使うコマンド

### Docker開発

Dockerイメージのビルド：
```bash
./ros2/scripts/build.sh
```

インタラクティブな開発用コンテナの実行（実機ロボットハードウェア使用）：
```bash
./ros2/scripts/run.sh real
```

シミュレーションでの実行：
```bash
./ros2/scripts/run.sh sim
```

コンテナ内でROSパッケージをビルド：
```bash
cd /workspace/ros2
colcon build --symlink-install
source install/setup.bash
```

特定のパッケージをビルド：
```bash
colcon build --packages-select crane_x7_log --symlink-install
source install/setup.bash
```

### Docker Compose（クイックスタート）

`ros2/`ディレクトリ内の`.env.template`から`.env`を作成し、以下を設定：
- `USB_DEVICE`: USBデバイスパス（デフォルト：`/dev/ttyUSB0`）
- `USB_DEVICE_FOLLOWER`: フォロワーロボット用USBデバイスパス（デフォルト：`/dev/ttyUSB1`）
- `DISPLAY`: X11ディスプレイ（デフォルト：`:0`）

```bash
# テンプレートから.envファイルを作成
cd ros2
cp .env.template .env
# 必要に応じて.envを編集
```

実機ロボットで実行：
```bash
docker compose -f ros2/docker-compose.yml --profile real up
```

シミュレーションで実行：
```bash
docker compose -f ros2/docker-compose.yml --profile sim up
```

実機ロボットとカメラビューアで実行（RealSense D435ストリームを表示）：
```bash
docker compose -f ros2/docker-compose.yml --profile real-viewer up
```

テレオペレーション（動作教示）で実行：
```bash
# リーダーモードのみ（記録なしの手動教示）
docker compose -f ros2/docker-compose.yml --profile teleop-leader up

# データロガー付きリーダーモード（記録ありの手動教示）
docker compose -f ros2/docker-compose.yml --profile teleop-leader-logger up

# データロガーとカメラビューア付きリーダーモード（記録とビデオ表示ありの手動教示）
docker compose -f ros2/docker-compose.yml --profile teleop-leader-viewer up

# フォロワーモードのみ（2台のロボットが必要）
docker compose -f ros2/docker-compose.yml --profile teleop-follower up

# カメラビューア付きフォロワーモード（ビデオ表示付きフォロワーロボット、2台のロボットが必要）
docker compose -f ros2/docker-compose.yml --profile teleop-follower-viewer up

# データロガー付きフォロワーモード（模倣記録、2台のロボットが必要）
docker compose -f ros2/docker-compose.yml --profile teleop-follower-logger up

# リーダーとフォロワーを同時実行
docker compose -f ros2/docker-compose.yml --profile teleop up

# データロガー付きでリーダーとフォロワーを同時実行
docker compose -f ros2/docker-compose.yml --profile teleop-logger up

# カメラビューア付きフォロワー（フォロワー側カメラ表示）
docker compose -f ros2/docker-compose.yml --profile teleop-viewer up
```

Gemini統合で実行：
```bash
# 実機ロボットでGeminiノードとロボット制御を起動
docker compose -f ros2/docker-compose.yml --profile gemini up

# GazeboシミュレーションでGeminiノードとピックアンドプレースタスクを起動
docker compose -f ros2/docker-compose.yml --profile gemini-sim up
```

注意: Geminiプロファイルを使用する前に、`.env`ファイルに`GEMINI_API_KEY`を設定してください。

### ROS 2起動コマンド

実機ロボットでデモを起動（コンテナ内）：
```bash
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0
```

Gazeboシミュレーションを起動：
```bash
ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py
```

サンプルプログラムを実行（別のターミナルで）：
```bash
ros2 launch crane_x7_examples example.launch.py example:='gripper_control'
```

RVizでロボットモデルを表示：
```bash
ros2 launch crane_x7_description display.launch.py
```

RealSense D435カメラマウントの場合、起動コマンドに`use_d435:=true`を追加してください。

RealSenseカメラストリームを表示（スタンドアロン）：
```bash
ros2 launch crane_x7_log camera_viewer.launch.py
```

カスタムトピックでRealSenseカメラストリームを表示：
```bash
ros2 launch crane_x7_log camera_viewer.launch.py image_topic:=/camera/depth/image_rect_raw
```

### データロギング

データロガー付きでロボット制御を起動（実機ロボット）：
```bash
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

データロガー付きでロボット制御を起動（シミュレーション）：
```bash
ros2 launch crane_x7_log demo_with_logger.launch.py
```

スタンドアロンデータロガー（ロボットがすでに実行中の場合）：
```bash
ros2 launch crane_x7_log data_logger.launch.py output_dir:=/workspace/data/tfrecord_logs
```

NPZエピソードをTFRecord形式に変換：
```bash
python3 -m crane_x7_log.tfrecord_writer episode_data.npz episode_data.tfrecord
```

### ROS 2ビルドシステム

ワークスペースはcolconビルドシステムを使用します：
- `colcon build --symlink-install`: シンボリックリンク付きですべてのパッケージをビルド（開発時推奨）
- `colcon build --packages-select <package_name>`: 特定のパッケージをビルド
- `source install/setup.bash`: ビルド後にワークスペースをソース

## 主要なアーキテクチャの詳細

### 起動フロー

**実機ロボット**：
- Docker Compose：`crane_x7_log/real_with_logger.launch.py`を実行
  - `crane_x7_examples/demo.launch.py`（MoveIt2 + ハードウェア制御）を含む
  - OXEデータ収集用のデータロガーノードを追加
- 手動：`ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0`
  - MoveIt2（`crane_x7_moveit_config`）とハードウェアコントローラ（`crane_x7_control`）を起動

**シミュレーション**：
- Docker Compose：`crane_x7_log/demo_with_logger.launch.py`を実行
  - データロガー付きGazeboシミュレーションを含む
- 手動：`ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py`
  - ロボットモデルとMoveIt2付きでGazeboを起動

### USBデバイスアクセス

実機ロボットはDynamixelサーボへのUSBアクセスが必要です。docker-compose設定：
- ホストデバイス`$USB_DEVICE`をコンテナ内の`/dev/ttyUSB0`にマッピング
- USB権限設定については`crane_x7_control/README.md`を参照

### X11ディスプレイ

WSLとネイティブLinuxの両方が異なるボリュームマウントでサポートされています：
- WSL：`/tmp/.X11-unix`と`/mnt/wslg`をマウント
- Linux：`/tmp/.X11-unix`をマウント（`xhost +`が必要）

### データロギングアーキテクチャ

`crane_x7_log`パッケージはOXE互換のデータ収集パイプラインを実装しています：

**データフロー**：
1. **サブスクリプション**：`data_logger`ノードは以下をサブスクライブ：
   - `/joint_states`: 7つのアーム関節 + 1つのグリッパー状態
   - `/camera/color/image_raw`: RGBカメラフィード（オプション）
   - `/camera/aligned_depth_to_color/image_raw`: デプス画像（オプション）
2. **バッファリング**：エピソード長に達するまでステップをメモリにバッファ
3. **アクション割り当て**：`action[t] = state[t+1]`（次状態予測形式）
4. **保存**：エピソードをNPZまたはTFRecord形式でディスクに保存

**主要コンポーネント**：
- `data_logger.py`: マルチモーダルデータを収集するメインROS 2ノード
- `episode_saver.py`: エピソードの永続化を処理（NPZ/TFRecord）
- `tfrecord_writer.py`: エピソードデータをVLAトレーニング用TFRecord形式に変換
- `image_processor.py`: 画像エンコードと処理ユーティリティ
- `config_manager.py`: 設定の読み込みと検証

**出力形式**（NPZ）：
```
episode_0000_YYYYMMDD_HHMMSS/
  └── episode_data.npz
      ├── states: (N, 8)      # 関節位置
      ├── actions: (N, 8)     # 次状態（1つシフト）
      ├── timestamps: (N,)    # UNIXタイムスタンプ
      ├── images: (N, H, W, 3) # RGB画像（オプション）
      └── depths: (N, H, W)   # デプス画像（オプション）
```

**出力形式**（TFRecord）：
- `observation/state`: 関節位置（float32）
- `observation/image`: JPEGエンコードRGB画像（bytes）
- `observation/depth`: デプス配列（bytes、float32）
- `observation/timestamp`: UNIXタイムスタンプ（float32）
- `action`: ターゲット関節位置（float32）

## VLAトレーニングワークフロー

1. **データ収集**：`crane_x7_log`を使用してデモンストレーションエピソードを収集
   - ロガー付きでロボットを実行：`docker compose -f ros2/docker-compose.yml --profile real up`
   - エピソードは自動的に`/workspace/data/tfrecord_logs`に保存されます
   - 起動パラメータでエピソード長、保存形式（NPZ/TFRecord）を設定

2. **データ形式変換**（NPZを使用する場合）：
   ```bash
   python3 -m crane_x7_log.tfrecord_writer episode_data.npz episode_data.tfrecord
   ```

3. **OpenVLAのファインチューニング**：
   - `vla/openvla/`ディレクトリに配置
   - 収集したTFRecordデータを使用して事前訓練済みOpenVLAモデルをファインチューニング
   - HuggingFace PEFT経由でLoRAおよび完全ファインチューニングをサポート
   - 詳細なファインチューニング手順については`vla/openvla/README.md`を参照

4. **デプロイメント**：
   - REST API（`vla-scripts/deploy.py`）経由でファインチューニング済みモデルをデプロイ
   - クローズドループマニピュレーション用にROS 2制御スタックと統合

## ライセンスに関する注記

### このリポジトリ（オリジナルコード）

- **プロジェクトルートおよびオリジナルコード**：MITライセンス（Copyright 2025 nop）
- **crane_x7_log**：MITライセンス
- **crane_x7_vla**：MITライセンス
- **crane_x7_teleop**：MITライセンス
- **VLAファインチューニングスクリプト**：MITライセンス

### 外部/サードパーティパッケージ（Gitサブモジュール）

- **crane_x7_ros**（RT Corporation）：Apache License 2.0
- **crane_x7_description**（RT Corporation）：RT Corporation非商用ライセンス
  - 研究および内部使用のみ
  - 商用利用にはRT Corporationの事前許可が必要
- **OpenVLA**：MITライセンス（コード）
  - 事前訓練済みモデルには追加の制限がある場合があります（例：Llama-2ライセンス）

**重要**：RT Corporationパッケージ（`crane_x7_ros`、`crane_x7_description`）は、このリポジトリのオリジナルコードとは異なるライセンスを持っています。使用前に各LICENSEファイルを確認してください。

## 参考資料

- RT Corporation CRANE-X7リソース：
  - https://github.com/rt-net/crane_x7
  - https://github.com/rt-net/crane_x7_ros
  - https://github.com/rt-net/crane_x7_description
- OpenVLAプロジェクト：https://openvla.github.io/
- Open X-Embodiment：https://robotics-transformer-x.github.io/
