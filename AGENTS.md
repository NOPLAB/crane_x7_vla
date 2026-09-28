# Repository guidance

## Project layout

- `ros2/`: CRANE-X7 control, Gazebo simulation, teleoperation, logging, and inference nodes.
- `vla/`: VLA training code, configurations, and the `crane_x7_vla_rl` PPO package.
- `lerobot/`: LeRobot integration.
- `docs/`: setup and workflow documentation.

## Working in this repository

- Check the worktree before editing and preserve unrelated changes.
- Update `README.md` and the relevant page under `docs/` when a workflow or directory changes.
- Keep ROS 2 documentation explicit about node inputs and outputs and cover native and Docker workflows where applicable.
- Distinguish source inspection from successful container, ROS 2, simulator, and hardware validation.
- `sim/`, `lifter/`, and `vla-rl/` were removed. The PPO source lives at `vla/src/crane_x7_vla_rl/`. Do not present commands that depend on deleted directories as runnable. The Lift ROS 2 package and VLA-RL code still depend on `lift`; the legacy Docker files still reference deleted paths.
