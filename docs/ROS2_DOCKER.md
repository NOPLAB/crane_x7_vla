# ROS 2 Docker環境

このガイドでは、CRANE-X7ロボットアームを制御するためのROS 2 Humble環境のセットアップと使用方法を説明します。

## 概要

このプロジェクトでは、ROS 2 Humble（Ubuntu 22.04）ベースのDocker環境を提供しています。実機ロボットとGazeboシミュレーションの両方をサポートしており、データ収集、テレオペレーション、デモプログラムの実行が可能です。

## 前提条件

### ハードウェア要件

**実機ロボット使用時：**
- CRANE-X7ロボットアーム
- USBケーブル（Dynamixelサーボ接続用）
- Intel RealSense D435カメラ（オプション、データ収集時推奨）

**シミュレーション使用時：**
- X11対応のディスプレイ（Gazebo、RViz表示用）
- 推奨: 4GB以上のRAM

### ソフトウェア要件

- Docker (>= 20.10)
- Docker Compose (>= 2.0)
- X11サーバー（GUIアプリケーション表示用）
  - Linux: ネイティブサポート
  - WSL2: WSLg（Windows 11に組み込み）

## クイックスタート

### 1. 環境設定

`.env`ファイルをテンプレートから作成：

```bash
cd ros2
cp .env.template .env
# 必要に応じて.envを編集
```

主な設定項目：
- `USB_DEVICE`: リーダーロボット用USBデバイスパス（デフォルト: `/dev/ttyUSB0`）
- `USB_DEVICE_FOLLOWER`: フォロワーロボット用USBデバイスパス（デフォルト: `/dev/ttyUSB1`）
- `DISPLAY`: X11ディスプレイ（デフォルト: `:0`）

### 2. Dockerイメージのビルド

```bash
# プロジェクトルートから実行
cd ros2
./scripts/build.sh
```

これにより、以下を含むDockerイメージが作成されます：
- ROS 2 Humble（Ubuntu 22.04ベース）
- MoveIt2モーションプランニングフレームワーク
- Gazeboシミュレーター
- CRANE-X7制御パッケージ
- Intel RealSenseドライバー（オプション）

### 3. 実行モードの選択

このプロジェクトでは、docker-composeプロファイルで複数の実行モードをサポートしています：

#### 実機ロボット制御

```bash
# 基本的な実機制御
docker compose --profile real up

# カメラビューア付き実機制御（RealSense D435ストリーム表示）
docker compose --profile real-viewer up
```

#### シミュレーション

```bash
# Gazeboシミュレーション
docker compose --profile sim up
```

#### テレオペレーション（動作教示）

```bash
# リーダーモードのみ（記録なしの手動教示）
docker compose --profile teleop-leader up

# データロガー付きリーダーモード（記録ありの手動教示）
docker compose --profile teleop-leader-logger up

# データロガーとカメラビューア付きリーダーモード
docker compose --profile teleop-leader-viewer up

# フォロワーモードのみ（2台のロボットが必要）
docker compose --profile teleop-follower up

# カメラビューア付きフォロワーモード
docker compose --profile teleop-follower-viewer up

# データロガー付きフォロワーモード（模倣記録、2台のロボットが必要）
docker compose --profile teleop-follower-logger up

# リーダーとフォロワーを同時実行
docker compose --profile teleop up

# データロガー付きでリーダーとフォロワーを同時実行
docker compose --profile teleop-logger up

# カメラビューア付きフォロワー（フォロワー側カメラ表示）
docker compose --profile teleop-viewer up
```

## インタラクティブな使用方法

コンテナ内でROS 2コマンドを直接実行する場合：

### 1. コンテナの起動

```bash
# 開発用にインタラクティブコンテナを起動
./scripts/run.sh real    # 実機ロボット
# または
./scripts/run.sh sim     # シミュレーション
```

### 2. コンテナ内でのビルド

```bash
# コンテナ内で実行
cd /workspace/ros2
colcon build --symlink-install
source install/setup.bash
```

特定のパッケージのみをビルド：

```bash
colcon build --packages-select crane_x7_log --symlink-install
source install/setup.bash
```

### 3. ROS 2起動コマンド

#### 実機ロボット

基本的なデモ起動：

```bash
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0
```

RealSense D435カメラマウント使用時：

```bash
ros2 launch crane_x7_examples demo.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

データロガー付きで起動：

```bash
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

#### シミュレーション

Gazeboシミュレーション起動：

```bash
ros2 launch crane_x7_gazebo crane_x7_with_table.launch.py
```

データロガー付きシミュレーション：

```bash
ros2 launch crane_x7_log demo_with_logger.launch.py
```

#### サンプルプログラムの実行

別のターミナルでコンテナに接続：

```bash
docker exec -it <container_name> bash
```

サンプルプログラムを実行：

```bash
ros2 launch crane_x7_examples example.launch.py example:='gripper_control'
```

利用可能なサンプル：
- `gripper_control`: グリッパーの開閉
- `pose_groupstate`: 事前定義姿勢への移動
- `joint_values`: 関節角度指定
- `pick_and_place`: ピックアンドプレースデモ

#### ビジュアライゼーション

RVizでロボットモデルを表示：

```bash
ros2 launch crane_x7_description display.launch.py
```

RealSenseカメラストリームを表示：

```bash
# デフォルトトピック（カラー画像）
ros2 launch crane_x7_log camera_viewer.launch.py

# カスタムトピック（デプス画像）
ros2 launch crane_x7_log camera_viewer.launch.py image_topic:=/camera/depth/image_rect_raw
```

## データロギング

CRANE-X7データセット用にVLAトレーニング互換のデータロギング機能を提供しています。

### データロガーの起動

#### 実機ロボット

```bash
# データロガー付きで実機制御を起動
docker compose --profile real up
```

または手動起動：

```bash
ros2 launch crane_x7_log real_with_logger.launch.py port_name:=/dev/ttyUSB0 use_d435:=true
```

#### シミュレーション

```bash
# データロガー付きでシミュレーションを起動
docker compose --profile sim up
```

または手動起動：

```bash
ros2 launch crane_x7_log demo_with_logger.launch.py
```

#### スタンドアロンデータロガー

ロボットがすでに実行中の場合：

```bash
ros2 launch crane_x7_log data_logger.launch.py output_dir:=/workspace/data/tfrecord_logs
```

### データ形式

データロガーはOXE（Open X-Embodiment）互換形式でエピソードを保存します：

#### 出力ディレクトリ構造

```
data/tfrecord_logs/
├── episode_0000_YYYYMMDD_HHMMSS/
│   └── episode_data.npz
├── episode_0001_YYYYMMDD_HHMMSS/
│   └── episode_data.npz
└── ...
```

#### NPZ形式の内容

```python
episode_data.npz:
  - states: (N, 8)      # 関節位置（7アーム + 1グリッパー）
  - actions: (N, 8)     # 次状態（1ステップシフト）
  - timestamps: (N,)    # UNIXタイムスタンプ
  - images: (N, H, W, 3) # RGB画像（オプション）
  - depths: (N, H, W)   # デプス画像（オプション）
```

### データ変換

NPZエピソードをTFRecord形式に変換（VLAトレーニング用）：

```bash
python3 -m crane_x7_log.tfrecord_writer episode_data.npz episode_data.tfrecord
```

## USBデバイス設定

実機ロボットを使用する場合、Dynamixelサーボへのアクセス許可が必要です。

### デバイスの確認

```bash
ls -l /dev/ttyUSB*
```

### udevルールの設定（Linux）

永続的なアクセス許可を設定：

```bash
# udevルールファイルを作成
sudo nano /etc/udev/rules.d/99-dynamixel.rules
```

以下の内容を追加：

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", MODE="0666"
```

ルールをリロード：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 一時的なアクセス許可

```bash
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1  # フォロワーロボット使用時
```

## X11ディスプレイ設定

GUIアプリケーション（Gazebo、RViz）を表示するためのX11設定。

### Linux（ネイティブ）

```bash
# X11接続を許可
xhost +local:docker

# 環境変数を設定（.envファイル）
DISPLAY=:0
```

### WSL2（Windows）

Windows 11のWSLgは自動的に設定されます。追加設定は不要です。

`.env`ファイルのデフォルト設定：

```bash
DISPLAY=:0
```

### トラブルシューティング

X11接続エラーが発生した場合：

```bash
# ホストで実行
echo $DISPLAY

# Dockerコンテナで実行
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix alpine env | grep DISPLAY
```

## コンテナ内のディレクトリ構造

```
/workspace/
├── ros2/
│   ├── crane_x7_ros/          # RT Corporation公式パッケージ
│   │   ├── crane_x7_control/  # ハードウェア制御
│   │   ├── crane_x7_examples/ # サンプルプログラム
│   │   ├── crane_x7_gazebo/   # シミュレーション
│   │   └── crane_x7_moveit_config/  # MoveIt2設定
│   ├── crane_x7_description/  # URDFモデル定義
│   ├── crane_x7_log/          # データロギングパッケージ
│   └── crane_x7_teleop/       # テレオペレーションパッケージ
├── data/                      # データセット（マウント）
│   └── tfrecord_logs/
└── scripts/                   # ヘルパースクリプト
```

## トラブルシューティング

### USBデバイスが見つからない

**症状:**
```
Error: Could not open port /dev/ttyUSB0
```

**解決策:**

1. デバイスが接続されているか確認：
   ```bash
   ls -l /dev/ttyUSB*
   ```

2. udevルールを設定（前述の「USBデバイス設定」を参照）

3. `.env`ファイルで正しいデバイスパスを確認

### Gazebo/RVizが表示されない

**症状:**
GUIウィンドウが表示されない

**解決策:**

1. X11アクセスを許可：
   ```bash
   xhost +local:docker
   ```

2. `DISPLAY`環境変数を確認：
   ```bash
   echo $DISPLAY
   ```

3. WSL2の場合、WSLgが有効か確認

### colconビルドエラー

**症状:**
```
CMake Error: ...
```

**解決策:**

1. 依存関係を更新：
   ```bash
   sudo apt update
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

2. ビルドディレクトリをクリーンアップ：
   ```bash
   rm -rf build/ install/ log/
   colcon build --symlink-install
   ```

### カメラが検出されない

**症状:**
```
No RealSense devices detected
```

**解決策:**

1. カメラが接続されているか確認：
   ```bash
   lsusb | grep Intel
   ```

2. docker-compose.ymlでUSBデバイスが正しくマウントされているか確認

3. RealSenseドライバーを再インストール（コンテナ内）

### データロガーが動作しない

**症状:**
エピソードが保存されない

**解決策:**

1. 出力ディレクトリのパーミッションを確認：
   ```bash
   ls -ld data/tfrecord_logs/
   ```

2. ロガーノードが実行中か確認：
   ```bash
   ros2 node list | grep data_logger
   ```

3. ログを確認：
   ```bash
   ros2 node info /data_logger_node
   ```

## 高度な使用方法

### カスタムパラメータでの起動

```bash
ros2 launch crane_x7_examples demo.launch.py \
    port_name:=/dev/ttyUSB0 \
    use_d435:=true \
    rviz_config:=/path/to/custom.rviz
```

### 複数のロボット制御

2台のCRANE-X7を同時に制御：

```bash
# 端末1: リーダーロボット
docker compose --profile teleop-leader up

# 端末2: フォロワーロボット
docker compose --profile teleop-follower up
```

### MoveIt2でのモーションプランニング

Pythonスクリプトでカスタムモーションを実行：

```python
from moveit_msgs.msg import MoveGroupAction
# MoveIt2 APIを使用してカスタムプランニング
```

### ROSトピックの監視

```bash
# 利用可能なトピックのリスト
ros2 topic list

# 特定のトピックをエコー
ros2 topic echo /joint_states

# トピックの周波数を確認
ros2 topic hz /joint_states

# トピックの情報を表示
ros2 topic info /joint_states
```

### ROSサービスの呼び出し

```bash
# 利用可能なサービスのリスト
ros2 service list

# サービスを呼び出し
ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers
```

## 開発ワークフロー

推奨される開発フローは以下の通りです：

### 1. 開発環境のセットアップ

```bash
# インタラクティブコンテナを起動
./scripts/run.sh real

# コンテナ内でパッケージをビルド
cd /workspace/ros2
colcon build --symlink-install
source install/setup.bash
```

### 2. コードの編集

ホストマシンでコードを編集（変更は自動的にコンテナと同期）：

```bash
# ホストで編集
vim ros2/crane_x7_log/crane_x7_log/data_logger.py
```

### 3. テストとデバッグ

```bash
# コンテナ内でノードを個別に実行
ros2 run crane_x7_log data_logger_node

# またはlaunchファイルで実行
ros2 launch crane_x7_log data_logger.launch.py
```

### 4. データ収集とトレーニング

```bash
# ステップ1: データ収集
docker compose --profile real up

# ステップ2: VLAトレーニング（別のガイドを参照）
docker compose --profile vla run --rm vla_finetune
```

## パフォーマンス最適化

### リアルタイム性の向上

```bash
# コンテナをprivilegedモードで実行（docker-compose.yml）
privileged: true

# RTプライオリティを設定
ulimit -r 99
```

### メモリ使用量の削減

```bash
# 不要なノードを無効化
# launch ファイルで不要なノードをコメントアウト
```

## クリーンアップ

### コンテナの停止と削除

```bash
# すべてのコンテナを停止
docker compose down

# ボリュームも削除
docker compose down -v
```

### イメージの削除

```bash
# ROS2イメージを削除
docker rmi $(docker images -q '*ros2*')
```

### ビルドキャッシュのクリーンアップ

```bash
# ホストのビルドディレクトリをクリーンアップ
cd ros2
rm -rf build/ install/ log/
```

## 参考資料

### CRANE-X7関連
- [RT Corporation CRANE-X7公式リポジトリ](https://github.com/rt-net/crane_x7)
- [crane_x7_ros（ROS 2パッケージ）](https://github.com/rt-net/crane_x7_ros)
- [crane_x7_description](https://github.com/rt-net/crane_x7_description)

### ROS 2関連
- [ROS 2 Humbleドキュメント](https://docs.ros.org/en/humble/)
- [MoveIt2ドキュメント](https://moveit.picknik.ai/humble/index.html)
- [Gazeboシミュレーター](https://gazebosim.org/)

### データセットとVLA
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [OpenVLAプロジェクト](https://openvla.github.io/)

### Docker関連
- [Docker公式ドキュメント](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
