# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## 概要

このリポジトリには、CRANE-X7ロボットアームを制御するためのROS 2 Humbleコード、およびビジョンベースのマニピュレーションタスクのためのOpenVLA（Vision-Language-Action）統合が含まれています。このプロジェクトは実機とGazeboシミュレーションの両方をサポートしています。

## 詳細ドキュメント

- [README.md](README.md) - プロジェクト概要とクイックスタート
- [docs/ROS2_DOCKER.md](docs/ROS2_DOCKER.md) - ROS 2 Docker環境の詳細ガイド
- [docs/VLA_DOCKER.md](docs/VLA_DOCKER.md) - VLAトレーニング環境の詳細ガイド
- [vla/README.md](vla/README.md) - VLAファインチューニング詳細
- [sim/README.md](sim/README.md) - ManiSkillシミュレータ詳細
- [ros2/src/crane_x7_gemini/README.md](ros2/src/crane_x7_gemini/README.md) - Gemini API統合詳細

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
   - `crane_x7_teleop/`: テレオペレーションパッケージ
     - トルクOFFモードで手動教示を実現
     - リーダー/フォロワーアーキテクチャで模倣学習をサポート
     - データロガーと統合してデモンストレーションを記録
   - `crane_x7_vla/`: VLA推論パッケージ
     - ファインチューニング済みVLAモデルのROS 2統合
     - リアルタイム制御用の推論ノード
   - `crane_x7_sim_gazebo/`: Gazeboシミュレーション環境
     - テーブルとオブジェクト付きシミュレーション環境
     - 実機と同じインターフェースでテスト可能

2. **OpenVLA** (`vla/openvla/`)
   - ロボットマニピュレーション用Vision-Language-Actionモデル
   - Prismatic VLMsに基づく
   - Embodied AIタスクのファインチューニングとデプロイメントをサポート
   - Open X-Embodimentデータセットミックスで訓練

3. **Docker環境**
   - **ROS 2 Docker** (`ros2/Dockerfile`)
     - `base`: ROS 2 Humble + CRANE-X7パッケージ（本番環境）
     - `dev`: 開発ツール追加版（推奨）
     - Ubuntu 22.04ベース、X11フォワーディング対応
   - **VLA Docker** (`vla/Dockerfile`)
     - `base`: CUDA 12.1 + Python 3.11 + OpenVLA + OpenPI + crane_x7_vla
     - `dev`: Jupyter、TensorBoard等の開発ツール追加版
     - マルチGPU対応、LoRAファインチューニング最適化済み

4. **ManiSkillシミュレーション** (`sim/`)
   - ManiSkill（SAPIEN物理エンジン）ベースのCRANE-X7シミュレーション環境
   - Dreamerスタイルのワールドモデルトレーニング対応
   - ピックアンドプレースタスク環境
   - ハンドカメラ統合（640x480 RGB）
   - 8次元行動空間（アーム7関節 + グリッパー1関節）

### Gazeboシミュレーション環境

このプロジェクトには2つのGazeboシミュレーション環境があります：

1. **crane_x7_gazebo**（`crane_x7_ros/`サブモジュール内）
   - RT Corporation公式シミュレーション
   - 基本的なGazebo環境

2. **crane_x7_sim_gazebo**（カスタムパッケージ）
   - VLAトレーニング用拡張環境
   - テーブルとオブジェクト配置
   - データ収集に最適化
   - 起動: `ros2 launch crane_x7_sim_gazebo crane_x7_with_table.launch.py`

### 開発モード

リポジトリはdocker-composeプロファイルで制御される複数の実行モードをサポートしています：
- **real**: 物理的なCRANE-X7にUSB経由で接続（`/dev/ttyUSB0`）、データロガー付き
- **real-viewer**: カメラビューア付き実機（RealSense D435ストリームを表示）
- **sim**: ハードウェアなしでGazeboシミュレーションを実行、データロガー付き
- **teleop-leader**: リーダーロボットでトルクOFFモード（手動教示）
- **teleop-leader-logger**: リーダーロボット + データロガー（記録あり）
- **teleop-leader-viewer**: リーダーロボット + データロガー + カメラビューア
- **teleop-follower**: フォロワーロボット（リーダーの動きを模倣、2台必要）
- **teleop-follower-viewer**: フォロワーロボット + カメラビューア
- **teleop-follower-logger**: フォロワーロボット + データロガー（記録あり）
- **teleop**: リーダー + フォロワーを同時起動（2台必要）
- **teleop-logger**: リーダー + フォロワー + データロガー（2台必要）
- **teleop-viewer**: リーダー + フォロワー + カメラビューア（フォロワー側）
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
- `USB_DEVICE`: リーダーロボットのUSBデバイスパス（デフォルト：`/dev/ttyUSB0`）
- `USB_DEVICE_FOLLOWER`: フォロワーロボット用USBデバイスパス（デフォルト：`/dev/ttyUSB1`）
- `DISPLAY`: X11ディスプレイ（デフォルト：`:0`）
- `ROS_DOMAIN_ID`: ROS 2 Domain ID（複数ロボット使用時は同じ値を設定、デフォルト：`42`）
- `GEMINI_API_KEY`: Google Gemini APIキー（geminiプロファイル使用時に必要）
- `CLEAN_BUILD`: ビルド前にクリーンするか（`true`/`false`、デフォルト：`false`）
- `USER_ID`, `GROUP_ID`, `USERNAME`: ホストユーザーとの権限整合のため（デフォルト：1000/1000/ros2user）

```bash
# テンプレートから.envファイルを作成
cd ros2
cp .env.template .env
# 必要に応じて.envを編集（特にGEMINI_API_KEYなど）
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
ros2 launch crane_x7_sim_gazebo crane_x7_with_table.launch.py
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

### VLA推論とデプロイメント

ファインチューニング済みモデルをロボット制御に統合：

実機でVLA推論を実行：
```bash
ros2 launch crane_x7_vla real_with_vla.launch.py \
  model_path:=/workspace/outputs/crane_x7_openvla/checkpoint-5000 \
  port_name:=/dev/ttyUSB0
```

シミュレーションでVLA推論を実行：
```bash
ros2 launch crane_x7_vla sim_with_vla.launch.py \
  model_path:=/workspace/outputs/crane_x7_openvla/checkpoint-5000
```

VLA制御のみを起動（ロボットが既に起動している場合）：
```bash
ros2 launch crane_x7_vla vla_control.launch.py \
  model_path:=/workspace/outputs/crane_x7_openvla/checkpoint-5000
```

### ManiSkillシミュレーション

ManiSkillベースのシミュレーション環境を使用：
```bash
# 関節動作テスト
python sim/src/scripts/joint_test.py

# MJCFモデルテスト
python sim/src/scripts/mjcf_test.py

# トレーニング
python sim/src/scripts/train.py
```

### ROS 2ビルドシステム

ワークスペースはcolconビルドシステムを使用します：
- `colcon build --symlink-install`: シンボリックリンク付きですべてのパッケージをビルド（開発時推奨）
- `colcon build --packages-select <package_name>`: 特定のパッケージをビルド
- `source install/setup.bash`: ビルド後にワークスペースをソース

### テスト実行

ROS 2パッケージのテストを実行：
```bash
# コンテナ内で
cd /workspace/ros2

# すべてのカスタムパッケージをテスト
colcon test --packages-select crane_x7_log crane_x7_vla crane_x7_gemini crane_x7_teleop
colcon test-result --verbose

# 特定のパッケージをテスト
colcon test --packages-select crane_x7_log
colcon test-result --all
```

VLAデータセットの検証：
```bash
# OpenVLAコンテナ内で
python3 /workspace/vla/test_crane_x7_loader.py
```

### プロジェクトナビゲーション

主要ディレクトリ：
- `ros2/src/` - ROS 2パッケージソース
- `vla/src/crane_x7_vla/` - VLAトレーニングCLI実装
- `sim/src/` - ManiSkillシミュレータ実装
- `data/tfrecord_logs/` - 収集されたエピソードデータ
- `outputs/` - トレーニング済みモデルとチェックポイント

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
- 手動：`ros2 launch crane_x7_sim_gazebo crane_x7_with_table.launch.py`
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

### 重要：OpenVLAとOpenPIの依存関係分離

OpenVLAとOpenPIは互いに競合する依存関係を持つため、**別々のDockerイメージとrequirementsファイル**を使用します：

| バックエンド | Dockerfile | Requirements | PyTorch | Transformers | 主な特徴 |
|------------|-----------|--------------|---------|--------------|---------|
| **OpenVLA** | `vla/Dockerfile.openvla` | `vla/requirements-openvla.txt` | 2.2.0 | 4.40.1 | Prismatic VLM、単一ステップアクション |
| **OpenPI** | `vla/Dockerfile.openpi` | `vla/requirements-openpi.txt` | 2.7.1 | 4.53.2 | JAX/Flax、アクションチャンク、Python 3.11+ |

### 1. データ収集

`crane_x7_log`を使用してデモンストレーションエピソードを収集：

```bash
# テレオペレーション（手動教示）でデータ収集
docker compose -f ros2/docker-compose.yml --profile teleop-leader-logger up

# エピソードは自動的に data/tfrecord_logs に保存されます
# 起動パラメータでエピソード長、保存形式（NPZ/TFRecord）を設定可能

# 言語インストラクションをパブリッシュ
ros2 topic pub /task/language_instruction std_msgs/String "data: 'タスクの説明'"
```

### 2. OpenVLAのファインチューニング

**環境構築**：
```bash
# OpenVLA用Dockerイメージをビルド
docker compose -f ros2/docker-compose.yml build vla_openvla

# インタラクティブコンテナを起動
docker compose -f ros2/docker-compose.yml run --rm vla_openvla
```

**コンテナ内でトレーニング**：
```bash
# シングルGPU
cd /workspace/vla
python -m crane_x7_vla.training.cli train \
  --backend openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --num-epochs 100

# マルチGPU（例：2台）
torchrun --nproc_per_node=2 -m crane_x7_vla.training.cli train \
  --backend openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla \
  --batch-size 8 \
  --learning-rate 5e-4 \
  --num-epochs 100
```

チェックポイントは`/workspace/outputs/crane_x7_openvla/`に保存されます。

### 3. OpenPIのファインチューニング

**環境構築**：
```bash
# OpenPI用Dockerイメージをビルド
docker compose -f ros2/docker-compose.yml build vla_openpi

# インタラクティブコンテナを起動
docker compose -f ros2/docker-compose.yml run --rm vla_openpi
```

**コンテナ内でトレーニング**：
```bash
# OpenPIトレーニング（JAXベース）
cd /workspace/vla
python -m crane_x7_vla.training.cli train \
  --backend openpi \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openpi \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --num-epochs 100
```

チェックポイントは`/workspace/outputs/crane_x7_openpi/`に保存されます。

### 4. データセット検証

**OpenVLA**：
```bash
docker compose -f ros2/docker-compose.yml run --rm vla_openvla \
  python3 /workspace/vla/test_crane_x7_loader.py
```

**OpenPI**：
```bash
docker compose -f ros2/docker-compose.yml run --rm vla_openpi \
  python3 /workspace/vla/test_crane_x7_loader.py
```

### 5. デプロイメント

- REST API（`vla/src/openvla/vla-scripts/deploy.py`）経由でファインチューニング済みモデルをデプロイ
- クローズドループマニピュレーション用にROS 2制御スタックと統合

### 環境変数設定

`ros2/.env`ファイルで以下を設定：
```bash
# Hugging Face token (モデルダウンロード用)
HF_TOKEN=your_huggingface_token

# Weights & Biases API key (ロギング用、オプション)
WANDB_API_KEY=your_wandb_api_key

# GPU設定
CUDA_VISIBLE_DEVICES=0  # 使用するGPU ID
```

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
