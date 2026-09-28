# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Pick and place environment for Genesis with CRANE-X7."""

from typing import Any, Optional

import numpy as np

from lift_genesis.environments.base import GenesisEnvironment


class PickPlace(GenesisEnvironment):
    """Pick and place task environment for Genesis with batch support.

    The robot must pick up a cube from the table and lift it to a target height.
    This environment mirrors the ManiSkill PickPlace implementation.
    Supports batch parallelization via Genesis n_envs.
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
        self._n_envs = 1  # Will be set after scene build

    def setup_scene(self) -> None:
        """Add table plane and cube to the scene."""
        import genesis as gs

        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Add cube
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(
                    self.cube_half_size * 2,
                    self.cube_half_size * 2,
                    self.cube_half_size * 2,
                ),
                pos=(
                    self.cube_spawn_center[0],
                    self.cube_spawn_center[1],
                    self.cube_half_size,
                ),
            )
        )

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Reset the cube position with random jitter (batched).

        Args:
            seed: Optional random seed for reproducibility.
            env_ids: Optional array of environment indices to reset.
                    If None, resets all environments.

        Returns:
            Info dictionary with cube positions.
        """
        # Get n_envs from scene if not set
        if hasattr(self.scene, "n_envs"):
            self._n_envs = self.scene.n_envs

        if env_ids is None:
            env_ids = np.arange(self._n_envs)
        env_ids = np.atleast_1d(env_ids)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomize cube position for each env
        n_reset = len(env_ids)
        jitters = self._rng.uniform(
            -self.cube_spawn_jitter, self.cube_spawn_jitter, size=(n_reset, 2)
        )
        xy = self.cube_spawn_center + jitters
        z = np.full(n_reset, self.cube_half_size)
        positions = np.column_stack([xy, z])

        # Set cube pose for specified envs
        self.cube.set_pos(positions, envs_idx=env_ids)
        identity_quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_reset, 1))
        self.cube.set_quat(identity_quat, envs_idx=env_ids)

        return {"cube_pos": positions}

    def compute_reward(self) -> np.ndarray:
        """Compute dense reward (matching ManiSkill implementation, batched).

        Returns:
            Reward array of shape (n_envs,).
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
        reward = np.where(metrics["success"], 5.0, reward)

        return reward.astype(np.float32)

    def is_success(self) -> np.ndarray:
        """Check if the cube has been lifted to the target height (batched).

        Returns:
            Boolean array of shape (n_envs,).
        """
        metrics = self._compute_metrics()
        return metrics["success"]

    def is_terminated(self) -> np.ndarray:
        """Check if the episode should terminate (batched).

        Returns:
            Boolean array of shape (n_envs,).
        """
        return self.is_success()

    def get_info(self) -> list[dict[str, Any]]:
        """Get task metrics (batched).

        Returns:
            List of info dictionaries, one per environment.
        """
        metrics = self._compute_metrics()
        infos = []
        for i in range(self._n_envs):
            infos.append(
                {
                    "success": bool(metrics["success"][i]),
                    "height_reached": bool(metrics["height_reached"][i]),
                    "is_close": bool(metrics["is_close"][i]),
                    "gripper_to_cube_dist": float(metrics["distance"][i]),
                    "cube_height": float(metrics["cube_height"][i]),
                }
            )
        return infos

    def _compute_metrics(self) -> dict[str, np.ndarray]:
        """Compute task metrics for reward and success evaluation (batched).

        Returns:
            Dictionary containing arrays of shape (n_envs,):
                - distance: gripper to cube distance
                - cube_height: current cube height
                - height_reached: whether cube is at target height
                - is_close: whether gripper is close to cube
                - success: whether task is complete
        """
        # Get cube position (n_envs, 3)
        cube_pos = self.cube.get_pos()
        if hasattr(cube_pos, "cpu"):
            cube_pos = cube_pos.cpu().numpy()
        cube_pos = np.asarray(cube_pos).reshape(self._n_envs, 3)

        # Get gripper link position (n_envs, 3)
        gripper_link = self.robot.get_link("crane_x7_gripper_base_link")
        gripper_pos = gripper_link.get_pos()
        if hasattr(gripper_pos, "cpu"):
            gripper_pos = gripper_pos.cpu().numpy()
        gripper_pos = np.asarray(gripper_pos).reshape(self._n_envs, 3)

        # Compute metrics (vectorized)
        distance = np.linalg.norm(cube_pos - gripper_pos, axis=1)
        height = cube_pos[:, 2]
        height_reached = height >= self.lift_success_height
        is_close = distance <= self.grasp_distance_threshold
        success = height_reached & is_close

        return {
            "distance": distance,
            "cube_height": height,
            "height_reached": height_reached,
            "is_close": is_close,
            "success": success,
        }
