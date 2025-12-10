# CRANE-X7 VLA ドキュメント

CRANE-X7ロボットアームの制御とVLAファインチューニングのためのリポジトリ。

## ドキュメント一覧

### ROS 2環境

| ドキュメント | 説明 |
|-------------|------|
| [ros2.md](ros2.md) | ROS 2 Humble環境（実機制御、Gazebo、Docker Composeプロファイル） |
| [gemini.md](gemini.md) | Gemini API統合（物体検出、軌道プランニング） |

### トレーニング

| ドキュメント | 説明 |
|-------------|------|
| [vla.md](vla.md) | VLAファインチューニング（OpenVLA、MiniVLA、OpenPI） |
| [vla-rl.md](vla-rl.md) | VLA強化学習（SimpleVLA-RL方式、PPO） |
| [lerobot.md](lerobot.md) | LeRobot統合（ACT、Diffusion Policy） |

### シミュレーション

| ドキュメント | 説明 |
|-------------|------|
| [sim.md](sim.md) | Liftシミュレータ抽象化（ManiSkill、Genesis） |

### ツール

| ドキュメント | 説明 |
|-------------|------|
| [slurm.md](slurm.md) | Slurmジョブ投下ツール（W&B Sweep連携） |

## クイックスタート

### 実機制御

```bash
cd crane_x7_vla
cp .env.template .env
# .envを編集
docker compose --profile real up
```

### シミュレーション

```bash
docker compose --profile sim up
```

### VLAファインチューニング

```bash
cd vla
docker build -f Dockerfile.openvla -t crane_x7_vla_openvla .
# コンテナ内でトレーニング実行
```

### 詳細

各ドキュメントを参照してください。

## 外部ドキュメント

- [crane_x7_ros](https://github.com/rt-net/crane_x7_ros) - RT Corporation公式ROS 2パッケージ
- [OpenVLA](https://github.com/openvla/openvla) - OpenVLA公式リポジトリ
- [OpenPI](https://github.com/Physical-Intelligence/openpi) - OpenPI公式リポジトリ
- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face LeRobot

## ライセンス

- **オリジナルコード**: MIT License（Copyright 2025 nop）
- **crane_x7_ros**: Apache License 2.0
- **crane_x7_description**: RT Corporation非商用ライセンス
