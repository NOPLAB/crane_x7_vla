# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

CRANE-X7ロボットアームの制御とVLAファインチューニングのためのリポジトリ。ROS 2 Humbleベースで実機とGazeboシミュレーションをサポート。

## 主要コマンド

### ROS 2環境（Docker）

```bash
# docker-compose.ymlの場所: ros2/docker/docker-compose.yml
cd ros2/docker

# 実機制御
docker compose --profile real up

# Gazeboシミュレーション
docker compose --profile sim up

# テレオペレーション（リーダー + フォロワー）
docker compose --profile teleop up

# テレオペレーション + カメラビューア
docker compose --profile teleop-viewer up

# Gemini API統合（実機）
docker compose --profile gemini up

# Gemini API統合（シミュレーション）
docker compose --profile gemini-sim up

# VLA推論（実機）
docker compose --profile vla up

# VLA推論（シミュレーション）
docker compose --profile vla-sim up
```

### VLAファインチューニング

```bash
cd vla

# OpenVLA用Dockerイメージビルド
docker build -f Dockerfile.openvla -t crane_x7_vla_openvla .

# OpenPI用Dockerイメージビルド
docker build -f Dockerfile.openpi -t crane_x7_vla_openpi .

# トレーニング実行（コンテナ内）
python -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla \
  --training-batch-size 16

# マルチGPU
torchrun --nproc_per_node=2 -m crane_x7_vla.training.cli train openvla ...

# 設定ファイル生成
python -m crane_x7_vla.training.cli config --backend openvla --output my_config.yaml

# LoRAマージ
python -m crane_x7_vla.scripts.merge_lora \
  --adapter_path /workspace/outputs/crane_x7_openvla/lora_adapters \
  --output_path /workspace/outputs/crane_x7_openvla_merged
```

### Slurmクラスター

```bash
cd slurm
pip install -e .

# ジョブ投下
slurm-submit submit jobs/train.sh

# W&B Sweep
slurm-submit sweep start examples/sweeps/sweep_openvla.yaml --max-runs 10
```

### LeRobot統合

```bash
cd lerobot

# キャリブレーション
docker compose --profile calibrate up

# テレオペ + データ収集
docker compose --profile teleop up

# トレーニング
docker compose --profile train up
```

### ROS 2ワークスペースビルド（コンテナ内）

```bash
cd /workspace/ros2
colcon build --symlink-install
source install/setup.bash

# 特定パッケージのみ
colcon build --packages-select crane_x7_log

# テスト実行
colcon test --packages-select crane_x7_log
colcon test-result --verbose
```

## ディレクトリ構成

```
crane_x7_vla/
├── ros2/                          # ROS 2ワークスペース
│   ├── docker/                    # Docker環境
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── src/
│       ├── crane_x7_ros/          # RT Corporation公式パッケージ（サブモジュール）
│       ├── crane_x7_description/  # URDFロボットモデル（サブモジュール）
│       ├── crane_x7_log/          # データロギング（RLDS/TFRecord）
│       ├── crane_x7_teleop/       # テレオペレーション
│       ├── crane_x7_vla/          # VLA推論ノード
│       ├── crane_x7_gemini/       # Gemini API統合
│       └── crane_x7_sim_gazebo/   # カスタムGazebo環境
├── vla/                           # VLAファインチューニング
│   ├── Dockerfile.openvla         # OpenVLA用Docker
│   ├── Dockerfile.openpi          # OpenPI用Docker
│   ├── configs/                   # 設定ファイル
│   └── src/
│       ├── crane_x7_vla/          # 統一トレーニングCLI
│       ├── openvla/               # OpenVLAサブモジュール
│       └── openpi/                # OpenPIサブモジュール
├── sim/                           # ManiSkillシミュレータ
│   └── src/
│       ├── crane_x7/              # ロボット定義（MJCF）
│       └── environments/          # タスク環境
├── lerobot/                       # LeRobot統合
│   ├── lerobot_robot_crane_x7/    # Robotプラグイン
│   ├── lerobot_teleoperator_crane_x7/  # Teleoperatorプラグイン
│   └── configs/                   # ポリシー設定（ACT, Diffusion）
├── slurm/                         # Slurmジョブ投下ツール
│   └── src/slurm_submit/
├── data/                          # データ保存
│   └── tfrecord_logs/             # 収集エピソード
└── scripts/                       # ユーティリティ
```

## アーキテクチャ詳細

### OpenVLA vs OpenPI

OpenVLAとOpenPIは依存関係が競合するため、**別々のDockerイメージ**を使用：

| バックエンド | Dockerfile | Python | PyTorch | 状態 |
|------------|-----------|--------|---------|------|
| OpenVLA | `Dockerfile.openvla` | 3.10 | 2.5.1 | 実装済み |
| OpenPI | `Dockerfile.openpi` | 3.11 | 2.7.1 | 未実装 |

### データフォーマット

**TFRecord出力**（crane_x7_log）:
- `observation/state`: 関節位置（float32、8次元）
- `observation/image`: JPEGエンコードRGB画像
- `action`: 次状態（`action[t] = state[t+1]`形式）

**CRANE-X7関節**:
- アーム: 7自由度
- グリッパー: 1自由度（2フィンガー連動）

### 起動フロー

- **実機**: `crane_x7_log/real.launch.py` → MoveIt2 + ハードウェア制御
- **シミュレーション**: `crane_x7_log/sim.launch.py` → Gazebo

### 環境変数（ros2/docker/.env）

```bash
USB_DEVICE=/dev/ttyUSB0           # リーダーロボット
USB_DEVICE_FOLLOWER=/dev/ttyUSB1  # フォロワーロボット
DISPLAY=:0                        # X11ディスプレイ
ROS_DOMAIN_ID=42                  # ROS 2 Domain ID
GEMINI_API_KEY=                   # Gemini APIキー
HF_TOKEN=                         # Hugging Faceトークン
VLA_MODEL_PATH=                   # VLAモデルパス
VLA_TASK_INSTRUCTION=             # タスク指示
```

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）
- **crane_x7_ros**（サブモジュール）: Apache License 2.0
- **crane_x7_description**（サブモジュール）: RT Corporation非商用ライセンス（研究・内部使用のみ）

## 参考資料

- [vla/README.md](vla/README.md) - VLAファインチューニング詳細
- [sim/README.md](sim/README.md) - ManiSkillシミュレータ
- [slurm/README.md](slurm/README.md) - Slurmジョブ投下ツール
- [lerobot/README.md](lerobot/README.md) - LeRobot統合
- [ros2/src/crane_x7_gemini/README.md](ros2/src/crane_x7_gemini/README.md) - Gemini API統合

## 注意事項

- 必ず日本語で応答すること
