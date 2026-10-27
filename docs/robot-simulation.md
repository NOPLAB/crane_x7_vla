# Downstream CRANE-X7 simulation

`vla/src/crane_x7_vla_robot/` owns the CRANE-X7 constants, MJCF, meshes,
ManiSkill agent, pick-place task and visual-domain registrations. The `vla`
wheel includes this package and its model resources. `usim` and its separately
installed engines contain no CRANE-X7 defaults.

From `vla/`, install `uv sync --extra sim-maniskill` (or
`pip install -e '.[sim-maniskill]'` with the core/engine distributions available).
For an application rollout:

```python
from crane_x7_vla_robot.simulation import create_simulator, simulator_config

config = simulator_config(
    backend="gpu", camera_uid="scene_camera",
    actuator_profile="servo_envelope", gripper_force_limit=2.0,
)
simulator = create_simulator("maniskill", config)
try:
    observation, info = simulator.reset(seed=123)
finally:
    simulator.close()
```

Profile names and the gripper override are application settings, not usim fields.
The factory expands them to nine stiffness, damping and force-limit values
aligned with explicit `joint_names`, and supplies initial positions. The
registered ManiSkill agent owns its downstream MJCF path; `robot_uid` selects
it without a legacy `robot_path` override. The mimic controller takes eight action
values (seven arm joints and one coupled gripper).

For the new model/agent API, file models are explicitly injected:

```python
from usim.model import ModelSource
from usim.specs import AgentSpec, ControlSpec
from crane_x7_vla_robot.crane_x7 import get_mjcf_path
from crane_x7_vla_robot.simulation import simulator_config

config = simulator_config("JointControl")
agent = AgentSpec(
    name="crane",
    model=ModelSource.from_path(get_mjcf_path()),
    control=ControlSpec(
        joint_names=config.joint_names,
        position_joints=config.joint_names,
        initial_qpos=config.initial_qpos,
        joint_stiffness=config.joint_stiffness,
        joint_damping=config.joint_damping,
        joint_force_limits=config.joint_force_limits,
    ),
)
```

Use the engine's supported model/session execution path for that specification.
For native registered ManiSkill tasks, explicitly import
`crane_x7_vla_robot.maniskill` and select the registered `CRANE-X7` agent.

## ROS 2

Install the `vla` distribution into the same Python environment as ROS 2 and
the selected usim engine. Build/source the ROS workspace, then use the existing
`crane_x7_bringup` launch files. In Docker, rebuild the VLA/ROS images so the
installed wheel and colcon entry points include the new package and executable.
The downstream `crane_x7_sim_node` subclasses the shared ROS bridge and uses the
application factory for task registration and controller setup; no registration depends on an import
in the launch process. Registration resolves the MJCF from the installed model
package, while `usim_config.yaml` supplies the joint order, rest state and
numeric controller settings.

The node consumes `/vla/predicted_action`, publishes
`/camera/color/image_raw`, `/joint_states` and `/usim/episode_done`, and preserves
the generic bridge's reset/pause services and task-info output.

## Engine boundaries

ManiSkill owns the complete existing pick-place rollout behavior downstream.
Genesis and Isaac file-based control require `env_id="JointControl"` and nine
explicit action targets; expand the coupled gripper target in the application.
`crane_x7_vla_robot.genesis_pick_place.PickPlace` preserves the former Genesis
task geometry, reset and metrics, but is not registered with the generic
adapter. A Genesis pick-place application must attach this task before scene
build and call its reset/reward/termination methods during its own rollout.
The VLA wrapper rejects unsupported pick-place engine combinations rather
than returning joint-control observations as successful task rollouts.

Local CPU contract/source checks do not establish GPU physics, ROS, container
or hardware execution.
