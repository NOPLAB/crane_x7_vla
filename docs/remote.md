# リモートGPU推論・トレーニング

Vast.ai、Runpod等のクラウドGPUサービスでVLA推論やVLA-RLトレーニングを実行する方法。

## 目次

- [VLA推論（rosbridge経由）](#vla推論rosbridge経由)
- [VLA-RLトレーニング](#vla-rlトレーニング)

---

# VLA推論（rosbridge経由）

ローカルのロボットを制御するリモートVLA推論。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│  リモートGPU (Vast.ai / Runpod)                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Docker Container (crane_x7_remote_inference)               ││
│  │  ┌─────────────────┐    ┌─────────────────────────────────┐││
│  │  │ vla_inference   │    │ Tailscale / SSH Tunnel          │││
│  │  │ _rosbridge.py   │───▶│ → rosbridge WebSocket (9090)    │││
│  │  └─────────────────┘    └─────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │ VPN / トンネル
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ローカル (ロボット / シミュレーション)                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  rosbridge_server (port 9090)                               ││
│  │  ┌───────────────────┐    ┌───────────────────────────────┐││
│  │  │ /camera/color/    │    │ /vla/predicted_action         │││
│  │  │ image_raw/        │───▶│ → robot_controller            │││
│  │  │ compressed        │    │ → FollowJointTrajectory       │││
│  │  └───────────────────┘    └───────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 前提条件

- ファインチューニング済みVLAモデル（ローカルパス or HuggingFace Hub）
- GPU（VRAM 16GB以上推奨、OpenVLA-7B使用時）
- 接続方法: Tailscale VPN または SSHトンネル

## 接続方法

### 方法1: Tailscale VPN（推奨）

Tailscaleは設定が簡単で、NAT越えも自動で行われる。

#### 1. Tailscaleアカウント作成

1. [Tailscale](https://tailscale.com/)でアカウント作成
2. [Admin Console](https://login.tailscale.com/admin/settings/keys)で認証キーを取得
   - Reusable: 有効化（複数インスタンスで使う場合）
   - Ephemeral: 有効化（インスタンス終了時に自動削除）

#### 2. ローカル側のセットアップ

```bash
# Tailscaleインストール (Ubuntu)
curl -fsSL https://tailscale.com/install.sh | sh

# 接続
sudo tailscale up --hostname=crane-x7-local

# 確認
tailscale status
```

#### 3. rosbridgeサーバー起動

```bash
# 実機
docker compose --profile rosbridge up

# または シミュレーション
docker compose --profile rosbridge-sim up
```

### 方法2: SSHトンネル

SSHアクセスが可能な場合はトンネルで接続できる。

#### 1. ローカル側でrosbridgeサーバー起動

```bash
docker compose --profile rosbridge up
```

#### 2. リモートGPU側からSSHトンネル作成

```bash
# Vast.ai/Runpodインスタンス上で実行
# ローカルマシンのSSHサーバーに接続してポートフォワード
ssh -R 9090:localhost:9090 user@your-local-ip

# または autossh で自動再接続
autossh -M 0 -f -N -R 9090:localhost:9090 user@your-local-ip
```

#### 3. 注意事項

- ローカル側でSSHサーバーが必要
- ファイアウォール/ルーターでポート転送設定が必要な場合あり
- 接続が不安定な場合は`autossh`を使用

## モデルの配置

### 方法1: HuggingFace Hub経由（推奨）

起動時に自動ダウンロードされる。初回は時間がかかるが、2回目以降はキャッシュされる。

```bash
# 公開リポジトリ
docker run --gpus all \
  -e VLA_MODEL_PATH="openvla/openvla-7b" \
  ...

# プライベートリポジトリ
docker run --gpus all \
  -e VLA_MODEL_PATH="your-username/crane_x7_openvla" \
  -e HF_TOKEN="hf_xxxxx" \
  ...
```

### 方法2: ネットワークストレージ

事前にモデルをアップロードしておく方法。

#### Vast.ai

```bash
# ボリュームを作成してモデルをアップロード
# Vast.ai WebUIから、または rsync/scp で転送

docker run --gpus all \
  -e VLA_MODEL_PATH="/workspace/models/crane_x7_openvla" \
  -v /path/to/vast/volume:/workspace/models:ro \
  ...
```

#### Runpod

```bash
# Network Volumeを作成してモデルをアップロード
# Runpod WebUIから設定

docker run --gpus all \
  -e VLA_MODEL_PATH="/runpod-volume/models/crane_x7_openvla" \
  -v /runpod-volume:/runpod-volume:ro \
  ...
```

## Dockerイメージの準備

### ビルド

```bash
cd crane_x7_vla
docker build -f docker/Dockerfile.remote-inference -t crane_x7_remote_inference .
```

### Docker Hubにプッシュ

```bash
# タグ付け
docker tag crane_x7_remote_inference <username>/crane_x7_remote_inference:latest

# プッシュ
docker push <username>/crane_x7_remote_inference:latest
```

## 実行方法

### Tailscale VPN + HuggingFace Hub

```bash
docker run --gpus all \
  -e TS_AUTHKEY="tskey-auth-xxxxx" \
  -e TS_HOSTNAME="crane-x7-inference" \
  -e ROSBRIDGE_HOST="crane-x7-local" \
  -e ROSBRIDGE_PORT="9090" \
  -e VLA_MODEL_PATH="openvla/openvla-7b" \
  -e VLA_TASK_INSTRUCTION="pick up the red cube" \
  -e VLA_DEVICE="cuda" \
  -e VLA_INFERENCE_RATE="10.0" \
  <username>/crane_x7_remote_inference:latest
```

### Tailscale VPN + ローカルモデル

```bash
docker run --gpus all \
  -e TS_AUTHKEY="tskey-auth-xxxxx" \
  -e TS_HOSTNAME="crane-x7-inference" \
  -e ROSBRIDGE_HOST="crane-x7-local" \
  -e ROSBRIDGE_PORT="9090" \
  -e VLA_MODEL_PATH="/workspace/models/crane_x7_openvla" \
  -e VLA_TASK_INSTRUCTION="pick up the red cube" \
  -v /path/to/models:/workspace/models:ro \
  <username>/crane_x7_remote_inference:latest
```

### SSHトンネル方式

```bash
# 事前にSSHトンネルを設定済みの場合
docker run --gpus all \
  --network host \
  -e ROSBRIDGE_HOST="localhost" \
  -e ROSBRIDGE_PORT="9090" \
  -e VLA_MODEL_PATH="openvla/openvla-7b" \
  -e VLA_TASK_INSTRUCTION="pick up the red cube" \
  <username>/crane_x7_remote_inference:latest
```

## 環境変数リファレンス

### Tailscale設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `TS_AUTHKEY` | (なし) | Tailscale認証キー。未設定時はTailscaleを使用しない |
| `TS_HOSTNAME` | `crane-x7-inference` | Tailscaleネットワーク上のホスト名 |
| `TS_STATE_DIR` | `/var/lib/tailscale` | Tailscale状態ディレクトリ |

### rosbridge接続

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `ROSBRIDGE_HOST` | `crane-x7-local` | rosbridgeサーバーのホスト名/IP |
| `ROSBRIDGE_PORT` | `9090` | rosbridgeサーバーのポート |
| `PEER_WAIT_TIMEOUT` | `300` | ピア接続待機タイムアウト（秒） |

### VLA推論

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VLA_MODEL_PATH` | (必須) | モデルパス（ローカル or HF Hub ID） |
| `VLA_TASK_INSTRUCTION` | `pick up the object` | タスク指示 |
| `VLA_DEVICE` | `cuda` | 推論デバイス（`cuda` / `cpu`） |
| `VLA_INFERENCE_RATE` | `10.0` | 推論レート（Hz） |
| `VLA_UNNORM_KEY` | `crane_x7` | アクション正規化キー |
| `HF_TOKEN` | (なし) | HuggingFace Hub認証トークン |

## GPUインスタンス選択の目安

| モデル | 最小VRAM | 推奨VRAM | 推奨インスタンス |
|--------|---------|---------|----------------|
| OpenVLA-7B | 16GB | 24GB | RTX 4090, A5000, A6000 |
| OpenVLA-7B (LoRA) | 16GB | 24GB | RTX 4090, A5000, A6000 |
| MiniVLA (~1B) | 8GB | 16GB | RTX 3090, RTX 4080 |

## トラブルシューティング

### Tailscale接続できない

```bash
# コンテナ内で確認
tailscale --socket=/var/lib/tailscale/tailscaled.sock status

# ログ確認
journalctl -u tailscaled
```

原因と対策:
- 認証キーが無効 → Admin Consoleで新しいキーを生成
- ネットワーク制限 → Vast.ai/Runpodのファイアウォール設定を確認

### rosbridgeに接続できない

```bash
# ローカル側でrosbridge起動確認
ros2 topic list  # rosbridgeコンテナ内で実行

# WebSocket接続テスト
python3 -c "import roslibpy; c=roslibpy.Ros('localhost',9090); c.run(); print(c.is_connected)"
```

原因と対策:
- rosbridgeが起動していない → `docker compose --profile rosbridge up`
- Tailscaleピアが見つからない → ローカル側でTailscale接続確認
- ポートがブロック → ファイアウォール設定確認

### GPUが認識されない

```bash
# コンテナ内で確認
nvidia-smi
```

原因と対策:
- `--gpus all`オプション忘れ
- NVIDIA Driverがインストールされていない
- nvidia-container-toolkitが未インストール

### モデルロードエラー

```bash
# HuggingFace Hub認証確認
python3 -c "from huggingface_hub import whoami; print(whoami())"
```

原因と対策:
- プライベートリポジトリにHF_TOKEN未設定
- モデルパスの誤り
- ディスク容量不足（HF Hubキャッシュ用に十分な空きが必要）

### 推論が遅い

原因と対策:
- VRAMが足りずCPUオフロード → より大きなVRAMのGPUを使用
- ネットワーク遅延 → 画像圧縮率を上げる、推論レートを下げる
- `VLA_INFERENCE_RATE`を調整（デフォルト10Hz）

## 動的タスク更新

実行中にタスク指示を変更できる：

```bash
# ローカル側で実行
ros2 topic pub /vla/update_instruction std_msgs/String "data: 'place the cube on the plate'"
```

---

# VLA-RLトレーニング

シミュレータ上でVLAモデルを強化学習でファインチューニングする。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│  リモートGPU (Vast.ai / Runpod)                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Docker Container (crane_x7_vla_rl)                         ││
│  │  ┌──────────────────────────────────────────────────┐      ││
│  │  │  VLARLTrainer                                    │      ││
│  │  │  ┌────────────┐  ┌─────────────┐  ┌───────────┐ │      ││
│  │  │  │ OpenVLA    │  │ PPO         │  │ Lift      │ │      ││
│  │  │  │ (LoRA)     │◀─│ Algorithm   │◀─│ Simulator │ │      ││
│  │  │  └────────────┘  └─────────────┘  └───────────┘ │      ││
│  │  └──────────────────────────────────────────────────┘      ││
│  │                           │                                 ││
│  │                           ▼                                 ││
│  │  ┌──────────────────────────────────────────────────┐      ││
│  │  │  Outputs                                         │      ││
│  │  │  - checkpoints/                                  │      ││
│  │  │  - W&B logs                                      │      ││
│  │  └──────────────────────────────────────────────────┘      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 前提条件

- GPU（VRAM 24GB以上推奨、OpenVLA-7B + シミュレータ使用時）
- SFTチェックポイント or HuggingFace Hub上のベースモデル
- W&Bアカウント（ログ用、オプション）

## Dockerイメージの準備

### ビルド

```bash
cd crane_x7_vla
docker build -f docker/Dockerfile.remote-vla-rl -t crane_x7_remote_vla_rl .
```

### Docker Hubにプッシュ

```bash
docker tag crane_x7_remote_vla_rl <username>/crane_x7_remote_vla_rl:latest
docker push <username>/crane_x7_remote_vla_rl:latest
```

## 実行方法

### SSHサーバー起動（Tailscale経由）

```bash
docker run --gpus all -d \
  -e TS_AUTHKEY="tskey-auth-xxxxx" \
  -e TS_HOSTNAME="crane-x7-vla-rl" \
  -e SSH_PUBLIC_KEY="$(cat ~/.ssh/id_rsa.pub)" \
  -e WANDB_API_KEY="your-wandb-api-key" \
  -e HF_TOKEN="hf_xxxxx" \
  -v $(pwd)/outputs:/workspace/vla-rl/outputs:rw \
  <username>/crane_x7_remote_vla_rl:latest
```

### SSH接続

```bash
# Tailscaleホスト名で接続
ssh -X vla@crane-x7-vla-rl

# X11フォワーディング確認
echo $DISPLAY  # :10.0 などが表示されればOK
xeyes          # テスト用GUIアプリ
```

### SSHサーバー起動（ポートフォワード経由）

```bash
docker run --gpus all -d \
  -p 2222:22 \
  -e SSH_PUBLIC_KEY="$(cat ~/.ssh/id_rsa.pub)" \
  -e WANDB_API_KEY="your-wandb-api-key" \
  -v $(pwd)/outputs:/workspace/vla-rl/outputs:rw \
  <username>/crane_x7_remote_vla_rl:latest

# 接続
ssh -X -p 2222 vla@<host-ip>
```

### コンテナ内でトレーニング実行

SSH接続後、コンテナ内でトレーニングを実行：

```bash
# 基本的なトレーニング
python -m crane_x7_vla_rl.training.cli train \
  --pretrained openvla/openvla-7b \
  --simulator maniskill \
  --env-id PickPlace-CRANE-X7 \
  --backend gpu \
  --num-parallel-envs 4 \
  --num-updates 1000 \
  --use-wandb

# SFTチェックポイントからトレーニング
python -m crane_x7_vla_rl.training.cli train \
  --sft-checkpoint /workspace/sft_checkpoint \
  --simulator maniskill \
  --backend gpu \
  --num-parallel-envs 4 \
  --use-wandb

# 設定ファイルを使用
python -m crane_x7_vla_rl.training.cli train \
  --config /workspace/config.yaml

# チェックポイントから再開
python -m crane_x7_vla_rl.training.cli train \
  --resume /workspace/vla-rl/outputs/crane_x7_vla_rl/checkpoint_500

# 評価のみ
python -m crane_x7_vla_rl.training.cli evaluate \
  --checkpoint /workspace/vla-rl/outputs/crane_x7_vla_rl/checkpoint_best \
  --num-episodes 20
```

### 直接コマンド実行（SSHなし）

```bash
docker run --gpus all \
  -e WANDB_API_KEY="your-wandb-api-key" \
  -e HF_TOKEN="hf_xxxxx" \
  -v $(pwd)/outputs:/workspace/vla-rl/outputs:rw \
  <username>/crane_x7_remote_vla_rl:latest \
  python -m crane_x7_vla_rl.training.cli train \
    --pretrained openvla/openvla-7b \
    --simulator maniskill \
    --backend gpu \
    --num-parallel-envs 4 \
    --use-wandb
```

## 環境変数リファレンス

### Tailscale設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `TS_AUTHKEY` | (なし) | Tailscale認証キー。未設定時はTailscaleを使用しない |
| `TS_HOSTNAME` | `crane-x7-vla-rl` | Tailscaleネットワーク上のホスト名 |
| `TS_STATE_DIR` | `/var/lib/tailscale` | Tailscale状態ディレクトリ |
| `TS_USERSPACE` | `true` | ユーザースペースモードで実行（コンテナ内推奨） |

### SSH設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `SSH_PUBLIC_KEY` | (なし) | SSH公開鍵（必須）。`vla`ユーザーの認証に使用 |
| `SSH_PORT` | `22` | SSHサーバーのポート |

### VLA-RL設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VLA_MODEL_PATH` | (なし) | モデルパス（ローカル or HF Hub ID） |
| `HF_TOKEN` | (なし) | HuggingFace Hub認証トークン |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFaceキャッシュディレクトリ |
| `WANDB_API_KEY` | (なし) | W&B APIキー（ログ用） |
| `WANDB_DIR` | `/workspace/vla-rl/outputs` | W&Bログディレクトリ |

### シミュレータ設定

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `LIFT_SIMULATOR` | `maniskill` | シミュレータ（`maniskill`/`genesis`/`isaacsim`） |
| `LIFT_BACKEND` | `gpu` | シミュレーションバックエンド（`cpu`/`gpu`） |

### GPU/レンダリング

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | 使用するGPU |
| `NVIDIA_DRIVER_CAPABILITIES` | `graphics,utility,compute` | NVIDIAドライバ機能 |
| `PYOPENGL_PLATFORM` | `egl` | OpenGLプラットフォーム（ヘッドレス用） |

### PyTorch最適化

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `CUDA_LAUNCH_BLOCKING` | `0` | CUDAカーネル同期（デバッグ用は`1`） |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | CUDAメモリアロケータ設定 |

## GPUインスタンス選択の目安

VLA-RLはVLAモデルとシミュレータを同時に動かすため、推論のみより多くのVRAMが必要。

| 構成 | 最小VRAM | 推奨VRAM | 推奨インスタンス |
|------|---------|---------|----------------|
| OpenVLA-7B + ManiSkill (4並列) | 24GB | 40GB+ | A100-40GB, A6000 |
| OpenVLA-7B + ManiSkill (2並列) | 20GB | 24GB | RTX 4090, A5000 |
| MiniVLA + ManiSkill (4並列) | 16GB | 24GB | RTX 4090, A5000 |

### メモリ使用量の調整

```bash
# 並列環境数を減らす
--num-parallel-envs 2

# シミュレーションバックエンドをCPUに
--backend cpu
```

## トラブルシューティング

### シミュレータがGPUを認識しない

```bash
# EGL設定確認
python3 -c "import os; os.environ['PYOPENGL_PLATFORM']='egl'; import OpenGL.EGL"

# ManiSkill GPU確認
python3 -c "import mani_skill; print(mani_skill.utils.common.get_gpu_info())"
```

原因と対策:
- EGLドライバ未インストール → `libegl1-mesa-dev`をインストール
- NVIDIA EGL未設定 → `PYOPENGL_PLATFORM=egl`を設定

### OOM (Out of Memory)

原因と対策:
- 並列環境数が多すぎる → `--num-parallel-envs`を減らす
- ミニバッチサイズが大きい → 設定ファイルで`minibatch_size`を減らす
- シミュレータがGPUメモリを使用 → `--backend cpu`でCPUシミュレーション

### W&Bログインエラー

```bash
# コンテナ内でログイン確認
python3 -c "import wandb; wandb.login()"
```

原因と対策:
- `WANDB_API_KEY`未設定
- ネットワーク制限 → プロキシ設定確認

### チェックポイント保存先

Vast.ai/Runpodではインスタンス終了時にローカルデータが消える。

対策:
- ネットワークストレージにマウント
- 定期的にチェックポイントをダウンロード
- W&B Artifactsでチェックポイントを保存

```bash
# Network Volume使用（Runpod）
docker run --gpus all \
  -v /runpod-volume/outputs:/workspace/vla-rl/outputs:rw \
  ...
```

## Vast.ai/Runpod固有の設定

### Vast.ai

```bash
# テンプレート設定
Image: <username>/crane_x7_vla_rl:latest
Docker Options: --gpus all
On-start Script:
  cd /workspace && python -m crane_x7_vla_rl.training.cli train \
    --pretrained openvla/openvla-7b \
    --use-wandb
```

### Runpod

```bash
# Pod設定
Container Image: <username>/crane_x7_vla_rl:latest
GPU: RTX 4090 or A100
Volume: /runpod-volume → /workspace/vla-rl/outputs

# 起動コマンド
python -m crane_x7_vla_rl.training.cli train \
  --pretrained openvla/openvla-7b \
  --output-dir /workspace/vla-rl/outputs \
  --use-wandb
```

---

## 関連ドキュメント

- [vla.md](vla.md) - VLAファインチューニング（SFT）
- [vla-rl.md](vla-rl.md) - VLA-RL詳細（アルゴリズム、設定）
- [sim.md](sim.md) - Liftシミュレータ
- [ros2.md](ros2.md) - ROS 2環境詳細
