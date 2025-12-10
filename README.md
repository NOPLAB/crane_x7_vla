# CRANE-X7 VLA

## 概要

CRANE-X7ロボットアームとVLAを使用した制御プログラムです。

**主な機能:**

- CRANE-X7の実機制御とGazeboシミュレーション
- RLDS形式でのデモンストレーションデータ収集
- 言語インストラクション対応のデータロギング
- OpenVLAモデルのファインチューニング
- テレオペレーションモード（キネステティックティーチング）
- RealSenseカメラ対応（RGB + 深度画像）

## ドキュメント

詳細なドキュメントは [docs/](docs/README.md) を参照してください。

| ドキュメント | 説明 |
|-------------|------|
| [docs/ros2.md](docs/ros2.md) | ROS 2環境（実機制御、Gazebo、Docker Composeプロファイル） |
| [docs/vla.md](docs/vla.md) | VLAファインチューニング（OpenVLA、MiniVLA、OpenPI） |
| [docs/vla-rl.md](docs/vla-rl.md) | VLA強化学習（SimpleVLA-RL方式、PPO） |
| [docs/sim.md](docs/sim.md) | Liftシミュレータ抽象化（ManiSkill、Genesis） |
| [docs/lerobot.md](docs/lerobot.md) | LeRobot統合（ACT、Diffusion Policy） |
| [docs/gemini.md](docs/gemini.md) | Gemini API統合 |
| [docs/slurm.md](docs/slurm.md) | Slurmジョブ投下ツール |

## ディレクトリ構成

| ディレクトリ | 説明 |
|-------------|------|
| `ros2/` | ROS 2ワークスペース。CRANE-X7の実機制御、Gazeboシミュレーション、テレオペレーション、データロギング（RLDS/TFRecord形式）、VLA推論ノード、Gemini API統合を含む。 |
| `vla/` | VLAファインチューニング環境。OpenVLAおよびOpenPIを用いたモデルトレーニング、LoRAアダプター管理、設定ファイル生成を行う。 |
| `vla-rl/` | VLA強化学習。SimpleVLA-RL方式でPPOを使用したVLAモデルのファインチューニング。 |
| `sim/` | Liftシミュレータ統合。ManiSkill、Genesisなど複数シミュレータの統一抽象化レイヤー。 |
| `lerobot/` | LeRobot統合。CRANE-X7用のRobotプラグイン、Teleoperatorプラグイン、ACT/Diffusionポリシー設定を含む。 |
| `slurm/` | Slurmクラスター向けジョブ投下ツール。W&B Sweepによるハイパーパラメータ探索もサポート。 |

## 必要なもの

- Native Linux
- Docker

## リポジトリのクローン

```bash
git clone --recursive https://github.com/NOPLAB/crane_x7_vla
```

## ライセンス

### このリポジトリのオリジナルコード

- **プロジェクト全体**: MIT License - Copyright 2025 nop

### 外部/サードパーティパッケージ - Gitサブモジュール

- **crane_x7_ros** - RT Corporation: Apache License 2.0
- **crane_x7_description** - RT Corporation: RT Corporation非商用ライセンス
  - 研究・内部使用のみ許可
  - 商用利用にはRT Corporationからの事前許可が必要
- **OpenVLA**: MIT License - コード部分
  - 事前学習済みモデルには別途制限あり、例えばLlama-2ライセンスなど

**重要**: RT Corporationのパッケージ `crane_x7_ros` と `crane_x7_description` は、このリポジトリのオリジナルコードとは異なるライセンスです。使用前に各LICENSEファイルを確認してください。

## 参考情報

### RT Corporation (CRANE-X7)

- [CRANE-X7公式](https://github.com/rt-net/crane_x7)
- [CRANE-X7 ROS 2パッケージ](https://github.com/rt-net/crane_x7_ros)
- [CRANE-X7 ハードウェア](https://github.com/rt-net/crane_x7_Hardware)
- [CRANE-X7 サンプルコード](https://github.com/rt-net/crane_x7_samples)

### OpenVLA

- [OpenVLA公式サイト](https://openvla.github.io/)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA論文](https://arxiv.org/abs/2406.09246)
- [HuggingFaceモデル](https://huggingface.co/openvla)

### Open X-Embodiment

- [Open X-Embodimentプロジェクト](https://robotics-transformer-x.github.io/)

---

## 著作権

Copyright (c) 2025 nop

このREADME.md、およびこのリポジトリのオリジナルコード（crane_x7_log、crane_x7_vla、crane_x7_teleop、VLAファインチューニングスクリプト等）はMITライセンスの下で提供されています。詳細は上記のライセンスセクションを参照してください。
