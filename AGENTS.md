# Repository guidance

## Project layout

- `ros2/`: CRANE-X7 control, Gazebo simulation, teleoperation, logging, and inference nodes.
- `vla/`: VLA training code and configurations.
- `vla-rl/`: PPO training code.
- `lerobot/`: LeRobot integration.
- `docs/`: setup and workflow documentation.

## Working in this repository

- Check the worktree before editing and preserve unrelated changes.
- Update `README.md` and the relevant page under `docs/` when a workflow or directory changes.
- Keep ROS 2 documentation explicit about node inputs and outputs and cover native and Docker workflows where applicable.
- Distinguish source inspection from successful container, ROS 2, simulator, and hardware validation.
- `sim/` and `lifter/` were removed. Do not present commands that depend on them as runnable. The remaining Lift ROS 2 package, VLA-RL code, and related Docker files still contain references to `sim/` or the `lift` Python package.
