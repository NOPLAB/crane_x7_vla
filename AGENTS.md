# Repository guidance

## Project layout

- `ros2/`: CRANE-X7 control, Gazebo simulation, teleoperation, logging, and inference nodes.
- `vla/`: VLA training code, configurations, the `crane_x7_vla_rl` PPO package, and Lift simulator packages and robot assets under `src/`.
- `lerobot/`: LeRobot integration.
- `docs/`: setup and workflow documentation.

## Working in this repository

- Check the worktree before editing and preserve unrelated changes.
- Update `README.md` and the relevant page under `docs/` when a workflow or directory changes.
- Keep ROS 2 documentation explicit about node inputs and outputs and cover native and Docker workflows where applicable.
- Distinguish source inspection from successful container, ROS 2, simulator, and hardware validation.
- The old top-level `sim/`, `lifter/`, and `vla-rl/` directories were removed. Lift and its adapters were restored under `vla/src/`. ManiSkill and Genesis require their optional dependencies; Isaac Sim remains a placeholder. Verify simulator execution separately from import and packaging checks.
