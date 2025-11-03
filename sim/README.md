# CRANE-X7 ManiSkill Simulator

このディレクトリには、ManiSkillベースのCRANE-X7ロボットシミュレータが含まれています。

## 概要

ManiSkill（SAPIEN物理エンジンベース）を使用したCRANE-X7ロボットのシミュレーション環境です。Vision-Language-Actionモデルのトレーニングやテストに使用できます。

## ディレクトリ構造

```
sim/
├── src/
│   ├── crane_x7/           # CRANE-X7ロボット定義
│   │   ├── crane_x7.py     # ManiSkill BaseAgentクラス
│   │   ├── crane_x7.xml    # MJCFロボットモデル定義
│   │   └── meshes/         # 3Dメッシュデータ
│   │       ├── collision/  # 衝突検出用メッシュ
│   │       └── visual/     # 視覚表示用メッシュ
│   ├── environments/       # タスク環境定義
│   │   ├── environment.py  # カスタム環境ベースクラス
│   │   └── dreamer_pickplace.py  # ピックアンドプレースタスク
│   └── scripts/            # ユーティリティスクリプト
│       ├── joint_test.py   # 関節動作テスト
│       ├── mjcf_test.py    # MJCFモデルテスト
│       ├── train.py        # トレーニングスクリプト
│       ├── record_policy.py  # ポリシー記録
│       └── test.py         # テストスクリプト
└── README.md
```

## 必要な環境

- Python 3.8+
- ManiSkill 3.0+
- SAPIEN
- PyTorch
- Gymnasium

## インストール

```bash
# ManiSkillのインストール
pip install mani-skill

# その他の依存パッケージ
pip install sapien torch gymnasium numpy
```

## 使い方

### ロボットの登録

CRANE-X7ロボットはManiSkillに自動登録されます：

```python
from sim.src.crane_x7 import CraneX7
# ロボットは自動的に"CRANE-X7"として登録されます
```

### 環境の使用

```python
import gymnasium as gym
from sim.src.environments import environment

# 環境の作成
env = gym.make("YourCustomEnv-v0")

# シミュレーションループ
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### スクリプトの実行

#### 関節テスト
```bash
python sim/src/scripts/joint_test.py
```

#### MJCFモデルテスト
```bash
python sim/src/scripts/mjcf_test.py
```

#### トレーニング
```bash
python sim/src/scripts/train.py
```

## ロボット仕様

### CRANE-X7関節構成

- **アーム関節**: 7自由度
  - crane_x7_shoulder_fixed_part_pan_joint
  - crane_x7_shoulder_revolute_part_tilt_joint
  - crane_x7_upper_arm_revolute_part_twist_joint
  - crane_x7_upper_arm_revolute_part_rotate_joint
  - crane_x7_lower_arm_fixed_part_joint
  - crane_x7_lower_arm_revolute_part_joint
  - crane_x7_wrist_joint

- **グリッパー関節**: 2自由度
  - crane_x7_gripper_finger_a_joint
  - crane_x7_gripper_finger_b_joint

### コントローラ設定

- **PDジョイント位置制御**: 絶対位置指定
- **PDジョイント差分位置制御**: 相対位置指定（デルタ）
- **グリッパー制御**: ミミック制御（2つのフィンガーが連動）

### センサー

- **ハンドカメラ**: グリッパーベースリンクに取り付けられたRGBカメラ
  - 解像度: 640x480
  - 視野角: 69度
  - 距離範囲: 0.01m - 10.0m

## タスク環境

### PickPlaceタスク（dreamer_pickplace.py）

Dreamerスタイルのワールドモデルトレーニング用ピックアンドプレースタスク：

- **観測**: ハンドカメラRGB画像（64x64にダウンサンプル）
- **行動空間**: 8次元（アーム7関節 + グリッパー1関節）
- **報酬**: タスク達成に基づく報酬設計

## 参考リソース

- [ManiSkill公式ドキュメント](https://maniskill.readthedocs.io/)
- [SAPIEN物理エンジン](https://sapien.ucsd.edu/)
- [CRANE-X7 ROS 2パッケージ](https://github.com/rt-net/crane_x7_ros)

## ライセンス

このシミュレータコードはMITライセンスの下で提供されています。ただし、CRANE-X7のメッシュとモデルデータは、RT Corporationの非商用ライセンスに従います。
