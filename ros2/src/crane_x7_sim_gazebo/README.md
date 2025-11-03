# crane_x7_sim_gazebo

CRANE-X7ロボットアーム用のカスタムGazeboシミュレーション環境

## 概要

このパッケージは、CRANE-X7ロボットアームを使ったpick and placeタスクなどのためのGazeboシミュレーション環境を提供します。

## 環境

### Pick and Place環境

[worlds/pick_and_place.sdf](worlds/pick_and_place.sdf)には以下が含まれます：

- **2つのテーブル**
  - ピック用テーブル（ロボット前方）
  - プレース用テーブル（側面）

- **4つの操作オブジェクト**
  - 赤い立方体（5cm）
  - 青い立方体（5cm）
  - 緑の立方体（5cm）
  - 黄色い円柱（直径5cm、高さ5cm）

- **追加照明**でより見やすい環境

## 使用方法

### ビルド

```bash
cd /workspace/ros2
colcon build --packages-select crane_x7_sim_gazebo --symlink-install
source install/setup.bash
```

### 起動

#### 基本起動

```bash
ros2 launch crane_x7_sim_gazebo pick_and_place.launch.py
```

#### RealSense D435カメラ付きで起動

```bash
ros2 launch crane_x7_sim_gazebo pick_and_place.launch.py use_d435:=true
```

### Dockerを使用した起動

```bash
# コンテナに入る
./ros2/scripts/run.sh sim

# コンテナ内でビルド
cd /workspace/ros2
colcon build --packages-select crane_x7_sim_gazebo --symlink-install
source install/setup.bash

# 起動
ros2 launch crane_x7_sim_gazebo pick_and_place.launch.py
```

## パッケージ構造

```
crane_x7_sim_gazebo/
├── CMakeLists.txt          # ビルド設定
├── package.xml             # パッケージメタデータ
├── README.md               # このファイル
├── launch/                 # 起動ファイル
│   └── pick_and_place.launch.py
├── worlds/                 # Gazeboワールドファイル
│   └── pick_and_place.sdf
└── gui/                    # GUI設定
    └── gui.config
```

## 依存関係

- ROS 2 Humble
- Ignition Gazebo
- crane_x7_description
- crane_x7_control
- crane_x7_moveit_config
- ros_gz_bridge
- controller_manager

## ライセンス

MIT License - Copyright 2025 nop
