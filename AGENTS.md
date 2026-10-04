# Repository guidance

## Project layout

- `ros2/`: CRANE-X7 control, Gazebo simulation, teleoperation, logging, and inference nodes.
- `vla/`: VLA training code, configurations, and the `crane_x7_vla_rl` policy package. usim backends and robot assets are provided by the external `usim` package.
- `lerobot/`: LeRobot integration.
- `docs/`: setup and workflow documentation.

## Working in this repository

- Check the worktree before editing and preserve unrelated changes.
- Update `README.md` and the relevant page under `docs/` when a workflow or directory changes.
- Keep ROS 2 documentation explicit about node inputs and outputs and cover native and Docker workflows where applicable.
- Distinguish source inspection from successful container, ROS 2, simulator, and hardware validation.
- Install the robot-neutral `usim` core and the separately packaged engine (`usim-maniskill`, `usim-genesis`, `usim-isaacsim`, or `usim-gazebo`) rather than adding simulator source directories to `PYTHONPATH`. Core imports use `usim`, `usim.types`, `usim.interface`, and `usim.factory`; robot assets use `usim.robots`. Simulator execution belongs to `usim.bridges.simulation_ros2`, exposed by the external `usim_sim` ROS package. CRANE launch files and configuration belong to `crane_x7_bringup`. No legacy package names or compatibility shims are retained. Verify simulator execution separately from import and packaging checks.
