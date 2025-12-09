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

# テレオペレーション（リーダー + フォロワーのみ）
docker compose --profile teleop up

# テレオペレーション + カメラ + データロガー
docker compose --profile log up

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
│       ├── crane_x7_sim_gazebo/   # カスタムGazebo環境
│       └── crane_x7_bringup/      # 統合launchファイル
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

| バックエンド | Dockerfile           | Python | PyTorch | 状態     |
| ------------ | -------------------- | ------ | ------- | -------- |
| OpenVLA      | `Dockerfile.openvla` | 3.10   | 2.5.1   | 実装済み |
| OpenPI       | `Dockerfile.openpi`  | 3.11   | 2.7.1   | 未実装   |

### データフォーマット

**TFRecord出力**（crane_x7_log）:
- `observation/state`: 関節位置（float32、8次元）
- `observation/image`: JPEGエンコードRGB画像
- `action`: 次状態（`action[t] = state[t+1]`形式）

**CRANE-X7関節**:
- アーム: 7自由度
- グリッパー: 1自由度（2フィンガー連動）

### 起動フロー

**crane_x7_bringup**パッケージで各種起動をまとめて管理：

| launchファイル | 説明 |
|---------------|------|
| `real.launch.py` | 実機制御（MoveIt2 + ハードウェア + ロガー） |
| `sim.launch.py` | Gazeboシミュレーション + ロガー |
| `teleop.launch.py` | テレオペ（リーダー + フォロワー） |
| `data_collection.launch.py` | カメラ + データロガー（テレオペと併用） |
| `gemini_real.launch.py` | Gemini API（実機） |
| `gemini_sim.launch.py` | Gemini API（シミュレーション） |
| `vla_real.launch.py` | VLA推論（実機） |
| `vla_sim.launch.py` | VLA推論（Gazebo） |
| `rosbridge_real.launch.py` | 実機 + rosbridge（リモートVLA用） |
| `rosbridge_sim.launch.py` | Gazebo + rosbridge（リモートVLA用） |
| `maniskill.launch.py` | ManiSkillシミュレーション |
| `maniskill_vla.launch.py` | ManiSkill + VLA推論 |
| `maniskill_logger.launch.py` | ManiSkill + データロガー |

使用例:
```bash
ros2 launch crane_x7_bringup real.launch.py use_d435:=true
ros2 launch crane_x7_bringup teleop.launch.py  # リーダー+フォロワーのみ
ros2 launch crane_x7_bringup data_collection.launch.py  # カメラ+ロガー（別プロセス）
```

**各パッケージの基本launch**（bringupから参照）:

| パッケージ | launchファイル | 説明 |
|-----------|---------------|------|
| crane_x7_log | `data_logger.launch.py` | データロガーノード単体 |
| crane_x7_log | `camera_viewer.launch.py` | カメラビューア単体 |
| crane_x7_teleop | `teleop_leader.launch.py` | リーダーノード単体 |
| crane_x7_teleop | `teleop_follower.launch.py` | フォロワーノード単体 |
| crane_x7_vla | `vla_control.launch.py` | VLAノード群 |
| crane_x7_vla | `vla_inference_only.launch.py` | 推論ノードのみ（リモートGPU用） |
| crane_x7_gemini | `trajectory_planner.launch.py` | Geminiプランナーノード |
| crane_x7_sim_gazebo | `pick_and_place.launch.py` | Gazebo環境 |
| crane_x7_sim_maniskill | `sim_only.launch.py` | ManiSkill環境 |

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
- 作業の最後にCLAUDE.mdと各ディレクトリにあるREADME.mdを更新すること
