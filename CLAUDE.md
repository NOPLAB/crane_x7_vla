# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

CRANE-X7ロボットアームの制御とVLAファインチューニングのためのリポジトリ。ROS 2 Humbleベースで実機とGazeboシミュレーションをサポート。

## 主要コマンド

### ROS 2環境（Docker）

```bash
# docker-compose.ymlはプロジェクトルートに配置
# プロジェクトルートから実行

# 実機制御
docker compose --profile real up

# Gazeboシミュレーション
docker compose --profile sim up

# テレオペレーション（リーダー + フォロワーのみ）
docker compose --profile teleop up

# テレオペレーション + カメラ + データロガー
docker compose --profile log up

# VLA推論（実機）
docker compose --profile vla up

# VLA推論（シミュレーション）
docker compose --profile vla-sim up

# Liftシミュレーション（統一シミュレータ抽象化）
docker compose --profile lift up

# Lift + VLA推論
docker compose --profile lift-vla up

# Lift + データロガー
docker compose --profile lift-logger up

# VLA-RLトレーニング
docker compose --profile vla-rl up

# VLA-RL開発モード（インタラクティブシェル）
docker compose --profile vla-rl-dev up

# rosbridgeサーバー（実機、リモート推論用）
docker compose --profile rosbridge up

# rosbridgeサーバー（シミュレーション）
docker compose --profile rosbridge-sim up

# リモートGPU推論クライアント
docker compose --profile remote-inference up

# LeRobot開発シェル
docker compose --profile lerobot up

# LeRobotトレーニング
docker compose --profile lerobot-train up
```

### Lift環境変数

`.env`（プロジェクトルート）で以下を設定可能:

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `LIFT_SIMULATOR` | シミュレータ（`maniskill`/`genesis`/`isaacsim`） | `maniskill` |
| `LIFT_BACKEND` | シミュレーションバックエンド（`gpu`/`cpu`） | `cpu` |
| `LIFT_RENDER_MODE` | レンダリングモード（`rgb_array`/`human`/`none`） | `none` |

GPU使用時:
```bash
# .envに設定
LIFT_SIMULATOR=maniskill
LIFT_BACKEND=gpu
LIFT_RENDER_MODE=rgb_array
```

**注意**: `rgb_array`と`human`モードはGPU必須。GPUなし環境では`cpu` + `none`を使用。

### VLAファインチューニング

```bash
cd vla

# Dockerイメージビルド（全バックエンド含む）
docker build -t crane_x7_vla .

# トレーニング実行（コンテナ内）
python -m crane_x7_vla.training.cli train openvla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_openvla \
  --training-batch-size 16

# MiniVLAトレーニング（VQ Action Chunking + Multi-image）
python -m crane_x7_vla.training.cli train minivla \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_minivla \
  --training-batch-size 32 \
  --vq-enabled \
  --multi-image-enabled

# マルチGPU
torchrun --nproc_per_node=2 -m crane_x7_vla.training.cli train openvla ...

# Pi0トレーニング（PaliGemma + Expert Gemma）
python -m crane_x7_vla.training.cli train pi0 \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_pi0

# Pi0.5トレーニング（adaRMSNorm + Discrete State）
python -m crane_x7_vla.training.cli train pi0.5 \
  --data-root /workspace/data/tfrecord_logs \
  --experiment-name crane_x7_pi05

# 設定ファイル生成
python -m crane_x7_vla.training.cli config --backend openvla --output my_config.yaml
python -m crane_x7_vla.training.cli config --backend minivla --output minivla_config.yaml
python -m crane_x7_vla.training.cli config --backend pi0 --output pi0_config.yaml
python -m crane_x7_vla.training.cli config --backend pi0.5 --output pi05_config.yaml

# W&B Sweepエージェント（ハイパーパラメータチューニング）
python -m crane_x7_vla.training.cli agent pi0.5 \
  --sweep-id <SWEEP_ID> \
  --entity <WANDB_ENTITY> \
  --project <WANDB_PROJECT> \
  --data-root /workspace/data/tfrecord_logs \
  --output-dir /workspace/outputs/checkpoints \
  --max-steps 10000

# LoRAマージ
python -m crane_x7_vla.scripts.merge_lora \
  --adapter_path /workspace/outputs/crane_x7_openvla/lora_adapters \
  --output_path /workspace/outputs/crane_x7_openvla_merged
```

### Python環境管理（uv）

各モジュールは[uv](https://docs.astral.sh/uv/)でPython依存関係を管理:

```bash
# 依存関係インストール
cd lifter && uv sync     # lifter
cd sim && uv sync        # シミュレータ
cd vla-rl && uv sync     # VLA-RL
cd vla && uv sync        # VLA（Docker環境用、Python 3.10-3.13が必要）
cd lerobot && uv sync    # LeRobot（Python 3.11-3.13が必要）

# 開発依存関係も含める
uv sync --group dev

# パッケージ実行
uv run lifter --help
uv run crane-x7-vla-rl --help
```

**注意**: vla/lerobotはnumpy<2.0.0制約のためPython 3.14では動作しません。Docker環境での使用を推奨。

### VLA-RL強化学習

```bash
cd vla-rl

# インストール（uv）
uv sync

# トレーニング（SFTチェックポイントから）
python -m crane_x7_vla_rl.training.cli train \
  --sft-checkpoint /workspace/vla/outputs/checkpoint \
  --simulator maniskill \
  --env-id PickPlace-CRANE-X7 \
  --num-parallel-envs 4 \
  --use-wandb

# トレーニング（事前学習モデルから）
python -m crane_x7_vla_rl.training.cli train \
  --pretrained openvla/openvla-7b

# 評価
python -m crane_x7_vla_rl.training.cli evaluate \
  --checkpoint outputs/crane_x7_vla_rl/checkpoint_best \
  --num-episodes 20

# 設定ファイル生成
python -m crane_x7_vla_rl.training.cli config --output my_config.yaml
```

### Slurmクラスター (lifter)

```bash
cd lifter
uv sync

# ジョブ投下
lifter submit jobs/train.sh

# W&B Sweep（リモートSlurmクラスター）
lifter sweep start examples/sweeps/sweep_openvla.yaml --max-runs 10

# W&B Sweep（ローカル実行）
lifter sweep start examples/sweeps/sweep_openvla.yaml \
  --local \
  --template examples/templates_local/openvla_sweep.sh \
  --max-runs 5

# ローカルSweep再開
lifter sweep resume <SWEEP_ID> \
  --local \
  --template examples/templates_local/openvla_sweep.sh
```

### LeRobot統合

```bash
# Docker Compose経由（推奨）
docker compose --profile lerobot up      # 開発シェル
docker compose --profile lerobot-train up # トレーニング

# または手動ビルド
docker build -f docker/Dockerfile.lerobot -t crane_x7_lerobot .

# 詳細はlerobot/README.mdを参照
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
├── docker-compose.yml             # 統一Docker Compose設定
├── .env                           # 環境変数設定
├── .env.template                  # 環境変数テンプレート
├── docker/                        # Docker環境
│   ├── Dockerfile.ros2            # ROS 2統合環境
│   ├── Dockerfile.remote-inference # リモートGPU推論
│   ├── Dockerfile.remote-vla-rl   # リモートVLA-RLトレーニング
│   ├── Dockerfile.vla-rl          # VLA-RL学習
│   ├── Dockerfile.lerobot         # LeRobot統合
│   ├── entrypoint-ros2.sh         # ROS 2用エントリーポイント
│   ├── entrypoint-remote-inference.sh # 推論用エントリーポイント
│   ├── entrypoint-remote-vla-rl.sh # VLA-RL用エントリーポイント
│   └── wait-for-peer.sh           # Tailscale待機スクリプト
├── ros2/                          # ROS 2ワークスペース
│   └── src/
│       ├── crane_x7_ros/          # RT Corporation公式パッケージ（サブモジュール）
│       ├── crane_x7_description/  # URDFロボットモデル（サブモジュール）
│       ├── crane_x7_log/          # データロギング（RLDS/TFRecord）
│       ├── crane_x7_teleop/       # テレオペレーション
│       ├── crane_x7_vla/          # VLA推論ノード
│       ├── crane_x7_gemini/       # Gemini API統合（物体検出、軌道プランニング）
│       ├── crane_x7_sim_gazebo/   # カスタムGazebo環境
│       ├── crane_x7_lift/         # 統一シミュレータROS 2インターフェース
│       └── crane_x7_bringup/      # 統合launchファイル
├── vla/                           # VLAファインチューニング
│   ├── pyproject.toml             # uv依存関係定義
│   ├── uv.lock                    # ロックファイル
│   ├── Dockerfile                 # 統一Dockerfile
│   ├── configs/                   # 設定ファイル
│   └── src/
│       ├── crane_x7_vla/          # 統一トレーニングCLI
│       │   ├── action_tokenizer/  # VQ Action Tokenizer（MiniVLA用）
│       │   └── backends/          # バックエンド実装
│       ├── openvla/               # OpenVLAサブモジュール
│       └── openpi/                # OpenPIサブモジュール
├── sim/                           # シミュレータ（lift抽象化）
│   ├── pyproject.toml             # uv依存関係定義
│   ├── uv.lock                    # ロックファイル
│   └── src/
│       ├── lift/                  # 統一シミュレータインターフェース
│       ├── robot/                 # 共有ロボットアセット（MJCF、メッシュ）
│       ├── lift_maniskill/        # ManiSkill実装
│       ├── lift_genesis/          # Genesis実装
│       └── lift_isaacsim/         # Isaac Sim実装（スケルトン）
├── vla-rl/                        # VLA強化学習（SimpleVLA-RL方式）
│   ├── pyproject.toml             # uv依存関係定義
│   ├── uv.lock                    # ロックファイル
│   ├── configs/                   # 設定ファイル
│   └── src/crane_x7_vla_rl/
│       ├── training/              # VLARLTrainer, CLI
│       ├── algorithms/            # PPO独自実装, GAE
│       ├── rollout/               # 並列ロールアウト管理
│       ├── environments/          # liftシミュレータラッパー
│       ├── rewards/               # バイナリ報酬関数
│       └── vla/                   # OpenVLAアダプター
├── lerobot/                       # LeRobot統合
│   ├── pyproject.toml             # uv依存関係定義
│   ├── uv.lock                    # ロックファイル
│   ├── lerobot_robot_crane_x7/    # Robotプラグイン
│   ├── lerobot_teleoperator_crane_x7/  # Teleoperatorプラグイン
│   └── configs/                   # ポリシー設定（ACT, Diffusion）
├── lifter/                        # Slurmジョブ投下ツール
│   ├── pyproject.toml             # uv依存関係定義
│   ├── uv.lock                    # ロックファイル
│   └── src/lifter/
├── data/                          # データ保存
│   └── tfrecord_logs/             # 収集エピソード
└── scripts/                       # ユーティリティスクリプト
    ├── data/                      # データ処理スクリプト
    └── update-instruction.sh      # タスク指示更新
```

## アーキテクチャ詳細

### VLAバックエンド比較

統一Dockerfile（`vla/Dockerfile`）に全バックエンドの依存関係を含む：

| バックエンド | パラメータ | 特徴 | 状態 |
|-------------|-----------|------|------|
| OpenVLA | ~7B | Prismatic VLMベース | 実装済み |
| OpenVLA-OFT | ~7B | L1 Regression + Action Chunking + FiLM | 実装済み |
| MiniVLA | ~1B | Qwen 2.5 + VQ Action Chunking | 実装済み |
| Pi0 | ~2.3B | PaliGemma + Expert Gemma + Flow Matching | 実装済み |
| Pi0.5 | ~2.3B | Pi0 + adaRMSNorm + Discrete State | 実装済み |

共通環境: **CUDA 12.8** / **Python 3.11** / **PyTorch 2.9.1**

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
| `vla_real.launch.py` | VLA推論（実機） |
| `vla_sim.launch.py` | VLA推論（Gazebo） |
| `gemini_real.launch.py` | Gemini API推論（実機） |
| `gemini_sim.launch.py` | Gemini API推論（Gazebo） |
| `rosbridge_real.launch.py` | 実機 + rosbridge（リモートVLA用） |
| `rosbridge_sim.launch.py` | Gazebo + rosbridge（リモートVLA用） |
| `lift.launch.py` | Liftシミュレーション（統一抽象化） |
| `lift_vla.launch.py` | Lift + VLA推論 |
| `lift_logger.launch.py` | Lift + データロガー |

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
| crane_x7_sim_gazebo | `pick_and_place.launch.py` | Gazebo環境 |
| crane_x7_lift | `sim.launch.py` | Lift統一シミュレータ |

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）
- **crane_x7_ros**（サブモジュール）: Apache License 2.0
- **crane_x7_description**（サブモジュール）: RT Corporation非商用ライセンス（研究・内部使用のみ）

## 参考資料

- [docs/README.md](docs/README.md) - ドキュメントトップページ
- [docs/ros2.md](docs/ros2.md) - ROS 2環境詳細
- [docs/gemini.md](docs/gemini.md) - Gemini API統合（物体検出、軌道プランニング）
- [docs/vla.md](docs/vla.md) - VLAファインチューニング詳細
- [docs/vla-rl.md](docs/vla-rl.md) - VLA強化学習（SimpleVLA-RL方式）
- [docs/remote.md](docs/remote.md) - リモートGPU推論・VLA-RLトレーニング（Vast.ai、Runpod）
- [docs/sim.md](docs/sim.md) - Liftシミュレータ抽象化
- [docs/lifter.md](docs/lifter.md) - lifter (Slurmジョブ投下ツール)
- [docs/lerobot.md](docs/lerobot.md) - LeRobot統合

## 注意事項

- 必ず日本語で応答すること
- 作業の合間にCLAUDE.mdとdocs/下にあるドキュメント更新すること
