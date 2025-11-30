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

### Docker Compose（クイックスタート）

`ros2/`ディレクトリ内の`.env.template`から`.env`を作成し、以下を設定：
- `USB_DEVICE`: リーダーロボットのUSBデバイスパス（デフォルト：`/dev/ttyUSB0`）
- `USB_DEVICE_FOLLOWER`: フォロワーロボット用USBデバイスパス（デフォルト：`/dev/ttyUSB1`）
- `DISPLAY`: X11ディスプレイ（デフォルト：`:0`）
- `ROS_DOMAIN_ID`: ROS 2 Domain ID（複数ロボット使用時は同じ値を設定、デフォルト：`42`）
- `GEMINI_API_KEY`: Google Gemini APIキー（geminiプロファイル使用時に必要）
- `USER_ID`, `GROUP_ID`, `USERNAME`: ホストユーザーとの権限整合のため（デフォルト：1000/1000/ros2user）

### データロギング

`crane_x7_log`パッケージは実機・シミュレーション両方でデータ収集が可能。
詳細は [docs/ROS2_DOCKER.md](docs/ROS2_DOCKER.md) を参照。

### VLA推論とデプロイメント

ファインチューニング済みモデルは`crane_x7_vla`パッケージで実機・シミュレーションに統合可能。
詳細は [docs/VLA_DOCKER.md](docs/VLA_DOCKER.md) を参照。

### ManiSkillシミュレーション

ManiSkillベースのシミュレーション環境。詳細は [sim/README.md](sim/README.md) を参照。

### ROS 2ビルドシステム

ワークスペースはcolconビルドシステムを使用。詳細は [docs/ROS2_DOCKER.md](docs/ROS2_DOCKER.md) を参照。

### テスト実行

ROS 2パッケージのテストはcolconで実行。VLAデータセットの検証スクリプトも利用可能。
詳細は各パッケージのREADMEを参照。

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

**シミュレーション**：
- Docker Compose：`crane_x7_log/demo_with_logger.launch.py`を実行
  - データロガー付きGazeboシミュレーションを含む

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
| **OpenVLA** | `vla/Dockerfile.openvla` | `vla/requirements-openvla.txt` | 2.5.1 | 4.57.3 | Prismatic VLM、単一ステップアクション |
| **OpenPI** | `vla/Dockerfile.openpi` | `vla/requirements-openpi.txt` | 2.7.1 | 4.53.2 | JAX/Flax、アクションチャンク、Python 3.11+ |

### 1. データ収集

`crane_x7_log`を使用してデモンストレーションエピソードを収集。エピソードは`data/tfrecord_logs`に保存される。
詳細は [docs/ROS2_DOCKER.md](docs/ROS2_DOCKER.md) を参照。

### 2. OpenVLAのファインチューニング

OpenVLA用Dockerイメージでトレーニング。シングルGPU/マルチGPU両対応。
チェックポイントは`/workspace/outputs/crane_x7_openvla/`に保存される。
詳細は [docs/VLA_DOCKER.md](docs/VLA_DOCKER.md) および [vla/README.md](vla/README.md) を参照。

### 3. OpenPIのファインチューニング

OpenPI用Dockerイメージでトレーニング（JAXベース）。
チェックポイントは`/workspace/outputs/crane_x7_openpi/`に保存される。
詳細は [docs/VLA_DOCKER.md](docs/VLA_DOCKER.md) および [vla/README.md](vla/README.md) を参照。

### 4. データセット検証

各バックエンド用のDockerコンテナ内でデータセット検証スクリプトを実行可能。
詳細は [vla/README.md](vla/README.md) を参照。

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

## 注意事項

- 必ず日本語で応答すること

