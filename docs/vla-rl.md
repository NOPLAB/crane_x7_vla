# VLA-RL 強化学習

VLAモデルを強化学習でファインチューニングするためのフレームワーク。SimpleVLA-RL方式を採用し、PPOアルゴリズムとLiftシミュレータを統合。

## 目次

- [クイックスタート](#クイックスタート)
- [インストール](#インストール)
- [CLI引数一覧](#cli引数一覧)
- [設定ファイル](#設定ファイル)
- [アルゴリズム詳細](#アルゴリズム詳細)
- [ディレクトリ構成](#ディレクトリ構成)
- [トラブルシューティング](#トラブルシューティング)

## クイックスタート

```bash
# プロジェクトルートから実行
cd crane_x7_vla

# VLA-RLトレーニング
docker compose --profile vla-rl up

# VLA-RL開発モード（インタラクティブシェル）
docker compose --profile vla-rl-dev up
```

### 手動実行

```bash
cd vla-rl
pip install -e .

# SFTチェックポイントからトレーニング
python -m crane_x7_vla_rl.training.cli train \
  --sft-checkpoint /workspace/vla/outputs/checkpoint \
  --experiment-name crane_x7_vla_rl

# 事前学習モデルからトレーニング
python -m crane_x7_vla_rl.training.cli train \
  --pretrained openvla/openvla-7b \
  --simulator maniskill

# 評価
python -m crane_x7_vla_rl.training.cli evaluate \
  --checkpoint outputs/crane_x7_vla_rl/checkpoint_best \
  --num-episodes 20

# 設定ファイル生成
python -m crane_x7_vla_rl.training.cli config --output my_config.yaml
```

## インストール

```bash
cd vla-rl
pip install -e .
```

### 必要環境

- Python 3.10+
- PyTorch 2.5.1+
- CUDA 12.x（GPU使用時）
- Liftシミュレータ（`sim/`からインストール）

## CLI引数一覧

### trainコマンド

```bash
python -m crane_x7_vla_rl.training.cli train [OPTIONS]
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--sft-checkpoint` | str | - | SFTチェックポイントのパス（`--pretrained`と排他） |
| `--pretrained` | str | `openvla/openvla-7b` | HuggingFaceモデルIDまたはパス |
| `--config` | str | - | YAML設定ファイルのパス |
| `--experiment-name` | str | `crane_x7_vla_rl` | 実験名（ログ、チェックポイント用） |
| `--output-dir` | str | `outputs` | 出力ディレクトリ |
| `--seed` | int | 42 | 乱数シード |
| `--simulator` | str | `maniskill` | シミュレータ（`maniskill`/`genesis`/`isaacsim`） |
| `--env-id` | str | `PickPlace-CRANE-X7` | 環境ID |
| `--backend` | str | `cpu` | シミュレーションバックエンド（`cpu`/`gpu`） |
| `--num-parallel-envs` | int | 4 | 並列環境数 |
| `--render` | flag | - | リアルタイム可視化を有効化 |
| `--num-updates` | int | 1000 | PPO更新回数 |
| `--num-rollouts` | int | 8 | 更新あたりのロールアウト数 |
| `--learning-rate` | float | 1e-5 | PPO学習率 |
| `--num-epochs` | int | 4 | 更新あたりのPPOエポック数 |
| `--instruction` | str | `pick up the object...` | タスク指示（自然言語） |
| `--use-wandb` | flag | - | W&Bログを有効化 |
| `--wandb-project` | str | `crane-x7-vla-rl` | W&Bプロジェクト名 |
| `--resume` | str | - | 再開するチェックポイントのパス |

### evaluateコマンド

```bash
python -m crane_x7_vla_rl.training.cli evaluate [OPTIONS]
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--checkpoint` | str | **必須** | チェックポイントディレクトリのパス |
| `--num-episodes` | int | 20 | 評価エピソード数 |
| `--simulator` | str | `maniskill` | シミュレータ |
| `--env-id` | str | `PickPlace-CRANE-X7` | 環境ID |
| `--seed` | int | 42 | 乱数シード |
| `--render` | flag | - | 評価エピソードをレンダリング |

### configコマンド

```bash
python -m crane_x7_vla_rl.training.cli config [OPTIONS]
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--output`, `-o` | str | `config.yaml` | 出力ファイルパス |
| `--template` | str | `default` | テンプレート（`default`/`fast`/`full`） |

テンプレートの違い:

| テンプレート | num_updates | num_parallel_envs | 用途 |
|-------------|-------------|-------------------|------|
| `fast` | 100 | 2 | 動作確認、デバッグ |
| `default` | 1000 | 4 | 標準的なトレーニング |
| `full` | 5000 | 8 | 本番トレーニング |

## 設定ファイル

### default.yaml

```yaml
experiment_name: crane_x7_vla_rl
seed: 42
device: cuda

# モデル設定
pretrained_checkpoint: openvla/openvla-7b
sft_checkpoint: null  # SFTモデル使用時に指定

# LoRA設定
lora_rank: 32
lora_alpha: 16

# トレーニング
num_updates: 1000
language_instruction: "pick up the object and place it"

# ロギング
use_wandb: false
wandb_project: crane-x7-vla-rl
log_interval: 10
eval_interval: 50
save_interval: 100
num_eval_episodes: 10

# ロールアウト設定
rollout:
  simulator: maniskill
  env_id: PickPlace-CRANE-X7
  backend: cpu
  render_mode: rgb_array
  num_parallel_envs: 4
  num_rollouts_per_update: 8
  max_steps: 100
  use_binary_reward: true
  temperature: 1.0

# PPO設定
ppo:
  learning_rate: 1.0e-5
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  num_epochs: 4
  minibatch_size: 32
  max_grad_norm: 0.5
  target_kl: 0.02
  clip_value_loss: true
  value_clip_range: 0.2
```

### 設定ファイルの使用

```bash
# 設定ファイルを生成
python -m crane_x7_vla_rl.training.cli config --output my_config.yaml

# 設定ファイルを使用してトレーニング
python -m crane_x7_vla_rl.training.cli train --config my_config.yaml

# CLI引数で上書き
python -m crane_x7_vla_rl.training.cli train \
  --config my_config.yaml \
  --num-updates 2000 \
  --use-wandb
```

## アルゴリズム詳細

### SimpleVLA-RL方式

VLAモデル（OpenVLA）をPPOで強化学習するフレームワーク。主な特徴:

1. **LoRA（Low-Rank Adaptation）**: VLMの重みを効率的に更新
2. **バイナリ報酬**: タスク成功時のみ報酬+1
3. **並列ロールアウト**: 複数環境で同時にデータ収集

### PPO（Proximal Policy Optimization）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `learning_rate` | 1e-5 | 学習率 |
| `gamma` | 0.99 | 割引率 |
| `gae_lambda` | 0.95 | GAE（Generalized Advantage Estimation）のλ |
| `clip_ratio` | 0.2 | クリッピング比率 |
| `value_loss_coef` | 0.5 | 価値損失の係数 |
| `entropy_coef` | 0.01 | エントロピー損失の係数 |
| `num_epochs` | 4 | ミニバッチ更新のエポック数 |
| `target_kl` | 0.02 | KLダイバージェンスの閾値 |

### GAE（Generalized Advantage Estimation）

アドバンテージ推定にGAEを使用し、バイアスとバリアンスのトレードオフを調整:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### バイナリ報酬

シミュレータ環境からの`success`フラグに基づくシンプルな報酬:

```python
reward = 1.0 if info.get("success", False) else 0.0
```

## ディレクトリ構成

```
vla-rl/
├── configs/                    # 設定ファイル
│   └── default.yaml            # デフォルト設定
├── src/crane_x7_vla_rl/
│   ├── training/               # トレーニング
│   │   ├── cli.py              # CLIエントリーポイント
│   │   └── trainer.py          # VLARLTrainer
│   ├── algorithms/             # アルゴリズム
│   │   ├── ppo.py              # PPO実装
│   │   └── advantage.py        # GAE計算
│   ├── rollout/                # ロールアウト管理
│   │   ├── rollout_manager.py  # 並列ロールアウト
│   │   └── trajectory_buffer.py # トラジェクトリバッファ
│   ├── environments/           # 環境ラッパー
│   │   ├── lift_wrapper.py     # Liftシミュレータラッパー
│   │   └── parallel_envs.py    # 並列環境管理
│   ├── rewards/                # 報酬関数
│   │   └── binary_reward.py    # バイナリ報酬
│   ├── vla/                    # VLAアダプター
│   │   └── openvla_adapter.py  # OpenVLAラッパー
│   └── config/                 # 設定クラス
│       ├── base.py             # VLARLConfig
│       ├── ppo_config.py       # PPOConfig
│       └── rollout_config.py   # RolloutConfig
├── setup.py
└── requirements.txt
```

## トラブルシューティング

### GPU/CPUの切り替え

```bash
# CPU使用
python -m crane_x7_vla_rl.training.cli train --backend cpu

# GPU使用
python -m crane_x7_vla_rl.training.cli train --backend gpu
```

**注意**: GPU使用時はLiftシミュレータのGPUバックエンドも必要。

### メモリ不足

並列環境数を減らす:

```bash
python -m crane_x7_vla_rl.training.cli train --num-parallel-envs 2
```

### シミュレータエラー

Liftシミュレータが正しくインストールされているか確認:

```bash
cd sim
pip install -e .

# テスト
python -c "from lift import make_env; env = make_env('maniskill', 'PickPlace-CRANE-X7')"
```

### W&Bログインエラー

```bash
wandb login
# APIキーを入力
```

### チェックポイントの読み込みエラー

```bash
# チェックポイント構造を確認
ls -la outputs/crane_x7_vla_rl/checkpoint_best/
```

必要なファイル:
- `config.yaml`
- `policy_state_dict.pt`
- `optimizer_state_dict.pt`（再開時のみ）
