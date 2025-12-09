# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Pick and place environment for Genesis with CRANE-X7."""

from typing import Any, Optional

import numpy as np

from lift_genesis.environments.base import GenesisEnvironment


class PickPlace(GenesisEnvironment):
    """Pick and place task environment for Genesis.

    The robot must pick up a cube from the table and lift it to a target height.
    This environment mirrors the ManiSkill PickPlace implementation.
    """

    # Task parameters (matching ManiSkill)
    goal_radius: float = 0.1
    cube_half_size: float = 0.02
    cube_spawn_center: np.ndarray = np.array([0.15, 0.02])
    cube_spawn_jitter: np.ndarray = np.array([0.01, 0.01])
    lift_height_offset: float = 0.12
    grasp_distance_threshold: float = 0.05

    def __init__(self, scene: Any, robot: Any, robot_init_qpos_noise: float = 0.02):
        """Initialize the PickPlace environment.

        Args:
            scene: Genesis scene instance.
            robot: Genesis robot entity.
            robot_init_qpos_noise: Standard deviation for initial qpos noise.
        """
        super().__init__(scene, robot, robot_init_qpos_noise)
        self.lift_success_height = self.cube_half_size + self.lift_height_offset
        self.cube = None
        self.plane = None

    def setup_scene(self) -> None:
        """Add table plane and cube to the scene."""
        import genesis as gs

        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Add cube
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(self.cube_half_size * 2, self.cube_half_size * 2, self.cube_half_size * 2),
                pos=(self.cube_spawn_center[0], self.cube_spawn_center[1], self.cube_half_size),
            )
        )

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset the cube position with random jitter.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Info dictionary with cube position.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomize cube position
        jitter = self._rng.uniform(-self.cube_spawn_jitter, self.cube_spawn_jitter)
        xy = self.cube_spawn_center + jitter
        z = self.cube_half_size

        # Set cube pose
        self.cube.set_pos(np.array([xy[0], xy[1], z]))
        self.cube.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))  # Identity quaternion (w, x, y, z)

        return {"cube_pos": np.array([xy[0], xy[1], z])}

    def compute_reward(self) -> float:
        """Compute dense reward (matching ManiSkill implementation).

        Returns:
            Scalar reward value.
        """
        metrics = self._compute_metrics()

        # Reaching reward: encourage gripper to approach cube
        reaching_reward = 1.0 - np.tanh(5.0 * metrics["distance"])

        # Lift reward: encourage lifting the cube
        lift_progress = np.clip(
            (metrics["cube_height"] - self.cube_half_size)
            / max(self.lift_success_height - self.cube_half_size, 1e-6),
            0.0,
            1.0,
        )
        lift_reward = lift_progress

        # Grasp bonus: additional reward when very close
        grasp_bonus = np.exp(-10.0 * metrics["distance"])

        reward = reaching_reward + lift_reward + grasp_bonus

        # Success bonus
        if metrics["success"]:
            reward = 5.0

        return float(reward)

    def is_success(self) -> bool:
        """Check if the cube has been lifted to the target height.

        Returns:
            True if task is successfully completed.
        """
        metrics = self._compute_metrics()
        return bool(metrics["success"])

    def is_terminated(self) -> bool:
        """Check if the episode should terminate.

        Returns:
            True if successful (task complete).
        """
        return self.is_success()

    def get_info(self) -> dict[str, Any]:
        """Get task metrics.

        Returns:
            Dictionary with task-specific metrics.
        """
        metrics = self._compute_metrics()
        return {
            "success": metrics["success"],
            "height_reached": metrics["height_reached"],
            "is_close": metrics["is_close"],
            "gripper_to_cube_dist": metrics["distance"],
            "cube_height": metrics["cube_height"],
        }

    def _compute_metrics(self) -> dict[str, Any]:
        """Compute task metrics for reward and success evaluation.

        Returns:
            Dictionary containing:
                - distance: gripper to cube distance
                - cube_height: current cube height
                - height_reached: whether cube is at target height
                - is_close: whether gripper is close to cube
                - success: whether task is complete
        """
        # Get cube position
        cube_pos = self.cube.get_pos()
        if hasattr(cube_pos, "cpu"):
            cube_pos = cube_pos.cpu().numpy()
        cube_pos = np.asarray(cube_pos).flatten()

        # Get gripper link position
        gripper_link = self.robot.get_link("crane_x7_gripper_base_link")
        gripper_pos = gripper_link.get_pos()
        if hasattr(gripper_pos, "cpu"):
            gripper_pos = gripper_pos.cpu().numpy()
        gripper_pos = np.asarray(gripper_pos).flatten()

        # Compute metrics
        distance = float(np.linalg.norm(cube_pos - gripper_pos))
        height = float(cube_pos[2])
        height_reached = height >= self.lift_success_height
        is_close = distance <= self.grasp_distance_threshold
        success = height_reached and is_close

        return {
            "distance": distance,
            "cube_height": height,
            "height_reached": height_reached,
            "is_close": is_close,
            "success": success,
        }
