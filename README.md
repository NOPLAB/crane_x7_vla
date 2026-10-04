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
| [docs/vla.md](docs/vla.md) | VLAファインチューニング（OpenVLA、MiniVLA、Pi0/Pi0.5） |
| [docs/vla-rl.md](docs/vla-rl.md) | VLA強化学習（OpenVLA PPO実験コード、Pi0.5報酬重み付きflow matching、Usimシミュレータ） |
| [docs/lerobot.md](docs/lerobot.md) | LeRobot統合（ACT、Diffusion Policy） |
| [docs/remote.md](docs/remote.md) | リモートGPU推論（Vast.ai、Runpod） |

VLA学習CLIの設定生成・読み込みと各バックエンドの実行確認状況は [docs/vla.md](docs/vla.md) を参照してください。

Pi0.5 のシミュレータ内 RL ファインチューニングは、SFT済みチェックポイントを入力として `crane_x7_vla_rl.training.pi05_rwfm` から実行します。実行条件と評価方法は [docs/vla-rl.md](docs/vla-rl.md) を参照してください。

## ディレクトリ構成

| ディレクトリ | 説明 |
|-------------|------|
| `ros2/` | ROS 2ワークスペース。CRANE-X7の実機制御、Gazeboシミュレーション、テレオペレーション、データロギング（RLDS/TFRecord形式）、VLA推論ノードを含む |
| `vla/` | VLAファインチューニング環境と `src/crane_x7_vla_rl/` のポリシーコード。usimシミュレータとロボット資産は外部の `usim==0.1.0` が提供。 |
| `lerobot/` | LeRobot統合。CRANE-X7用のRobotプラグイン、Teleoperatorプラグイン、ACT/Diffusionポリシー設定を含む。 |

## 必要なもの

- Native Linux
- Docker

## リポジトリのクローン

```bash
git clone --recursive https://github.com/NOPLAB/crane_x7_vla
git clone https://github.com/NOPLAB/usim usim
```

`usim` と `crane_x7_vla` は同じ親ディレクトリに配置します。`vla/` の
`uv sync --extra sim-maniskill` は `../../usim` をeditableでインストールします。
公開wheelの依存指定は `usim==0.1.0` で、ローカルパスを含みません。
ROSの実行処理はcoreの `usim.bridges.simulation_ros2` が所有し、外部の
`usim/ros2/src/usim_sim` が `usim_sim_node` を提供します。CRANE用設定と
`usim.launch.py`、`usim_logger.launch.py`、`usim_vla.launch.py` は
`crane_x7_bringup` にあります。互換パッケージはありません。
バックエンドは `usim/packages/{genesis,maniskill,isaacsim,gazebo}` にある
独立したPython distributionです。coreは `usim`、ManiSkillは `usim_maniskill`、
Genesisは `usim_genesis`、Isaac Simは `usim_isaacsim`、
ロボット資産は `usim.robots.crane_x7` からimportします。
ネイティブ・Docker手順は [docs/ros2.md](docs/ros2.md) と
[docs/vla.md](docs/vla.md) を参照してください。

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
