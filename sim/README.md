# CRANE-X7 Simulator (lift)

統一シミュレータ抽象化レイヤー。複数のシミュレータバックエンド（ManiSkill, Genesis, Isaac Sim）を共通インターフェースで利用可能。

## 構成

```
sim/
├── setup.py                  # Pythonパッケージ設定
└── src/
    ├── lift/                 # 抽象化レイヤー
    │   ├── interface.py      # Simulator抽象クラス
    │   ├── types.py          # Observation, StepResult, SimulatorConfig
    │   └── factory.py        # create_simulator()
    ├── robot/                # 共有ロボットアセット
    │   ├── crane_x7.py       # ロボット設定
    │   └── assets/           # MJCF、メッシュ
    ├── maniskill/            # ManiSkill実装
    │   ├── adapter.py        # ManiSkillSimulator
    │   ├── agent.py          # CraneX7エージェント
    │   └── environments/     # タスク環境
    ├── genesis/              # Genesis実装（スケルトン）
    └── isaacsim/             # Isaac Sim実装（スケルトン）
```

## インストール

```bash
cd sim
pip install -e ".[maniskill]"  # ManiSkill使用時
```

## 使用方法

### Python API

```python
from lift import create_simulator, SimulatorConfig

config = SimulatorConfig(
    env_id="PickPlace-CRANE-X7",
    backend="gpu",
    render_mode="rgb_array",
)

simulator = create_simulator("maniskill", config)

obs, info = simulator.reset()
while True:
    action = ...  # アクション取得
    result = simulator.step(action)
    if result.terminated or result.truncated:
        break

simulator.close()
```

### ROS 2

```bash
# crane_x7_liftパッケージ使用
ros2 launch crane_x7_lift sim.launch.py simulator:=maniskill backend:=gpu
```

## 対応シミュレータ

| シミュレータ | 状態 | 説明 |
|-------------|------|------|
| `maniskill` | 実装済み | ManiSkill 3.0ベース |
| `genesis` | スケルトン | 未実装 |
| `isaacsim` | スケルトン | 未実装 |

## lift インターフェース

### SimulatorConfig

```python
@dataclass
class SimulatorConfig:
    env_id: str                           # 環境ID
    backend: str = "cpu"                  # "cpu" | "gpu"
    render_mode: str = "rgb_array"        # "rgb_array" | "human" | "none"
    control_mode: str = "pd_joint_pos"    # 制御モード
    sim_rate: float = 30.0                # Hz
    max_episode_steps: int = 200
    robot_init_qpos_noise: float = 0.02
```

### Simulator

```python
class Simulator(ABC):
    arm_joint_names: list[str]
    gripper_joint_names: list[str]

    def reset(seed: int | None) -> tuple[Observation, dict]
    def step(action: np.ndarray) -> StepResult
    def get_observation() -> Observation
    def get_qpos() -> np.ndarray
    def get_qvel() -> np.ndarray
    def close()
```

### Observation

```python
@dataclass
class Observation:
    rgb_image: np.ndarray | None   # (H, W, 3) uint8
    depth_image: np.ndarray | None # (H, W) float32
    qpos: np.ndarray | None        # 関節位置
    qvel: np.ndarray | None        # 関節速度
    extra: dict                    # 追加データ
```

## シミュレータの追加

新しいシミュレータを追加するには:

1. `sim/src/<simulator>/` ディレクトリを作成
2. `adapter.py` に `Simulator` を継承したクラスを実装
3. `@register_simulator("<name>")` デコレータで登録

```python
from lift import Simulator, register_simulator

@register_simulator("my_simulator")
class MySimulator(Simulator):
    def __init__(self, config):
        super().__init__(config)
        # 初期化処理

    # 抽象メソッドを実装...
```
