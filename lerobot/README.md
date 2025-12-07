# LeRobot CRANE-X7 Integration

[LeRobot](https://github.com/huggingface/lerobot)を使用したCRANE-X7ロボットアームのテレオペレーションと模倣学習環境。

## 概要

このディレクトリは、CRANE-X7ロボットアーム用のLeRobot統合環境を提供します。

### 機能

- **テレオペレーション**: リーダー/フォロワー構成でのデモンストレーション収集
- **データ収集**: LeRobot形式でのデータセット保存
- **トレーニング**: ACT / Diffusion Policyによる模倣学習
- **推論**: 学習済みポリシーによるリアルタイム制御

### 対応ハードウェア

- **ロボット**: CRANE-X7（7DoFアーム + グリッパー）
- **モーター**: Dynamixel XM430-W350 / XM540-W270
- **カメラ**: Intel RealSense D435 / USBウェブカメラ

## クイックスタート

### 1. 環境設定

```bash
cd lerobot
cp .env.template .env
# .envファイルを編集して設定
```

### 2. Dockerイメージのビルド

```bash
docker compose build
```

### 3. キャリブレーション

```bash
# ロボットのキャリブレーション
docker compose --profile calibrate up

# テレオペレーター（リーダーアーム）のキャリブレーション
CALIBRATE_TYPE=teleop docker compose --profile calibrate up
```

### 4. テレオペレーション＋データ収集

```bash
# 環境変数を設定
export TASK_INSTRUCTION="pick up the red block"
export NUM_EPISODES=50

# データ収集開始
docker compose --profile teleop up
```

### 5. トレーニング

```bash
# ACTポリシーのトレーニング
docker compose --profile train up

# Diffusion Policyのトレーニング
docker compose --profile train-diffusion up
```

### 6. 推論

```bash
export POLICY_PATH=/workspace/outputs/act_crane_x7_dataset/checkpoints/last/pretrained_model

docker compose --profile inference up
```

## ディレクトリ構造

```
lerobot/
├── Dockerfile                      # Docker環境定義
├── docker-compose.yml              # サービス定義
├── .env.template                   # 環境変数テンプレート
├── pyproject.toml                  # Pythonパッケージ定義
├── requirements.txt                # 依存関係
│
├── lerobot_robot_crane_x7/         # Robotパッケージ
│   ├── __init__.py
│   ├── config_crane_x7.py          # 設定クラス
│   └── crane_x7.py                 # Robot実装
│
├── lerobot_teleoperator_crane_x7/  # Teleoperatorパッケージ
│   ├── __init__.py
│   ├── config_crane_x7_teleop.py   # 設定クラス
│   └── crane_x7_teleop.py          # Teleoperator実装
│
├── configs/                        # ポリシー設定
│   ├── act_crane_x7.yaml           # ACT設定
│   └── diffusion_crane_x7.yaml     # Diffusion Policy設定
│
├── scripts/                        # ユーティリティ
│   ├── calibrate.py                # キャリブレーション
│   ├── find_motors.py              # モーター検出
│   ├── inference.py                # 推論実行
│   └── convert_tfrecord_to_lerobot.py  # データ変換
│
├── calibration/                    # キャリブレーションデータ
└── outputs/                        # トレーニング出力
```

## 設定

### USBデバイス

```bash
# デバイスの確認
ls /dev/ttyUSB*

# 権限設定
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

### 環境変数

| 変数 | 説明 | デフォルト |
|------|------|------------|
| `USB_DEVICE_LEADER` | リーダーアームのポート | `/dev/ttyUSB0` |
| `USB_DEVICE_FOLLOWER` | フォロワーアームのポート | `/dev/ttyUSB1` |
| `TASK_INSTRUCTION` | タスクの説明 | `pick up the object` |
| `NUM_EPISODES` | 収集するエピソード数 | `10` |
| `BATCH_SIZE` | トレーニングバッチサイズ | `64` |
| `TRAIN_STEPS` | トレーニングステップ数 | `100000` |

## 既存データの変換

TFRecord形式のデータをLeRobot形式に変換：

```bash
docker compose run --rm dev python /workspace/lerobot/scripts/convert_tfrecord_to_lerobot.py \
    --tfrecord-dir /workspace/data/tfrecord_logs \
    --output-dir /workspace/data/lerobot_datasets/converted \
    --task "pick up the object"
```

## トラブルシューティング

### USBデバイスが認識されない

```bash
# udevルールを追加
sudo cp 99-dynamixel.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### モーターが応答しない

```bash
# モーター検出スクリプトを実行
docker compose run --rm dev python /workspace/lerobot/scripts/find_motors.py --port /dev/ttyUSB0
```

### カメラが認識されない

```bash
# RealSenseデバイスの確認
rs-enumerate-devices

# Video4Linuxデバイスの確認
v4l2-ctl --list-devices
```

## ライセンス

MIT License - Copyright 2025 nop

## 参考資料

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot - Bring Your Own Hardware](https://huggingface.co/docs/lerobot/integrate_hardware)
- [CRANE-X7 Documentation](https://github.com/rt-net/crane_x7_ros)
