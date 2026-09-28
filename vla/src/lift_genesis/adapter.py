# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Genesis simulator adapter implementing lift interface."""

from typing import Any, Optional, Type

import numpy as np

from lift.factory import register_simulator
from lift.interface import Simulator
from lift.types import Observation, SimulatorConfig, StepResult
from robot.crane_x7 import CraneX7Config, get_mjcf_path

# Environment registry
ENV_REGISTRY: dict[str, Type] = {}


def _register_environments() -> None:
    """Register available environments."""
    global ENV_REGISTRY
    from lift_genesis.environments.pick_place import PickPlace

    ENV_REGISTRY = {
        "PickPlace-CRANE-X7": PickPlace,
    }


@register_simulator("genesis")
class GenesisSimulator(Simulator):
    """Genesis simulator implementation with batch parallelization support."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)

        # Batch parallelization from config
        self._n_envs = config.n_envs

        # Internal state
        self._gs = None  # Genesis module
        self._scene = None
        self._robot = None
        self._camera = None
        self._environment = None
        self._dofs_idx: Optional[np.ndarray] = None
        self._current_obs = None
        self._episode_steps = np.zeros(self._n_envs, dtype=np.int32)
        self._max_episode_steps = config.max_episode_steps

        # Initialize Genesis and environment
        self._init_genesis()
        self._init_scene()
        self._add_robot()
        self._setup_camera()
        self._init_environment()

        # Build scene with batch parallelization
        print(f"[Genesis] Building scene with n_envs={self._n_envs}")
        self._scene.build(n_envs=self._n_envs)
        print("[Genesis] Scene built successfully")

        # Configure robot after build (requires built scene)
        self._build_dof_mapping()
        self._setup_control_gains()

        # Initial reset to populate observation
        self.reset()
        self._is_running = True

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def arm_joint_names(self) -> list[str]:
        return CraneX7Config.ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return CraneX7Config.GRIPPER_JOINT_NAMES

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids: Optional[np.ndarray] = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            env_ids: Optional array of environment indices to reset.
                    If None, resets all environments.

        Returns:
            Tuple of (initial observation, info dict).
        """
        if env_ids is None:
            env_ids = np.arange(self._n_envs)
        env_ids = np.atleast_1d(env_ids)

        self._episode_steps[env_ids] = 0

        # Set robot to rest position with noise for specified envs
        rest_qpos = np.tile(CraneX7Config.REST_QPOS, (len(env_ids), 1))
        if self.config.robot_init_qpos_noise > 0 and seed is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, self.config.robot_init_qpos_noise, rest_qpos.shape)
            rest_qpos = rest_qpos + noise

        # Set position for specified envs
        self._robot.set_dofs_position(rest_qpos, self._dofs_idx, envs_idx=env_ids)
        self._robot.set_dofs_velocity(
            np.zeros((len(env_ids), len(self._dofs_idx))),
            self._dofs_idx,
            envs_idx=env_ids,
        )

        # Reset environment (task-specific objects)
        info = {}
        if self._environment is not None:
            env_info = self._environment.reset(seed=seed, env_ids=env_ids)
            info.update(env_info)

        # Step once to settle physics
        self._scene.step()

        # Get observation
        self._current_obs = self._convert_observation()

        return self._current_obs, info

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one simulation step for all environments.

        Args:
            action: Joint position targets.
                   Shape (n_envs, 8) for 7 arm + 1 gripper,
                   or (n_envs, 9) for full DOF.
                   For single env, (8,) or (9,) is also accepted.

        Returns:
            StepResult containing batched observation, reward, terminated, truncated, info.
        """
        action = np.atleast_2d(action)
        if action.shape[0] != self._n_envs:
            raise ValueError(
                f"Action batch size {action.shape[0]} != n_envs {self._n_envs}"
            )

        # Apply action with mimic control
        self._apply_action(action)

        # Step simulation (steps all environments)
        self._scene.step()
        self._episode_steps += 1

        # Get observation
        self._current_obs = self._convert_observation()

        # Compute reward and check termination (batched)
        rewards = np.zeros(self._n_envs, dtype=np.float32)
        terminateds = np.zeros(self._n_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self._n_envs)]

        if self._environment is not None:
            rewards = self._environment.compute_reward()
            terminateds = self._environment.is_terminated()
            infos = self._environment.get_info()

        # Check truncation (max episode steps)
        truncateds = self._episode_steps >= self._max_episode_steps

        return StepResult(
            observation=self._current_obs,
            reward=rewards,
            terminated=terminateds,
            truncated=truncateds,
            info=infos,
        )

    def get_observation(self) -> Observation:
        """Get current observation without stepping.

        Returns:
            Current observation.
        """
        return self._current_obs

    def get_qpos(self) -> np.ndarray:
        """Get current joint positions for all environments.

        Returns:
            Joint positions array of shape (n_envs, 9).
        """
        qpos = self._robot.get_dofs_position(self._dofs_idx)
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        return np.asarray(qpos).reshape(self._n_envs, -1)

    def get_qvel(self) -> np.ndarray:
        """Get current joint velocities for all environments.

        Returns:
            Joint velocities array of shape (n_envs, 9).
        """
        qvel = self._robot.get_dofs_velocity(self._dofs_idx)
        if hasattr(qvel, "cpu"):
            qvel = qvel.cpu().numpy()
        return np.asarray(qvel).reshape(self._n_envs, -1)

    def close(self) -> None:
        """Release resources."""
        if self._scene is not None:
            # Genesis doesn't have explicit scene.close(), but we clear references
            self._scene = None
            self._robot = None
            self._camera = None
            self._environment = None
        self._is_running = False

    def _init_genesis(self) -> None:
        """Initialize Genesis backend."""
        import genesis as gs

        self._gs = gs

        # Select backend based on config
        backend_name = self.config.backend
        print(f"[Genesis] Initializing with backend: {backend_name}")

        if backend_name == "gpu":
            gs.init(backend=gs.cuda)
            print("[Genesis] Backend set to CUDA (gs.cuda)")
        else:
            gs.init(backend=gs.cpu)
            print("[Genesis] Backend set to CPU (gs.cpu)")

    def _init_scene(self) -> None:
        """Create Genesis scene."""
        gs = self._gs

        # Calculate timestep from sim_rate
        dt = 1.0 / self.config.sim_rate

        # Create scene with appropriate viewer settings
        show_viewer = self.config.render_mode == "human"

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=show_viewer,
        )

    def _add_robot(self) -> None:
        """Add robot entity to scene (before build)."""
        gs = self._gs

        # Load CRANE-X7 from MJCF
        mjcf_path = get_mjcf_path()
        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(
                file=mjcf_path,
                pos=(0.0, 0.0, 0.0),
            )
        )

    def _build_dof_mapping(self) -> None:
        """Build mapping from joint names to DOF indices."""
        dofs_idx = []
        for name in CraneX7Config.ALL_JOINT_NAMES:
            joint = self._robot.get_joint(name)
            if joint is not None:
                dofs_idx.append(joint.dof_idx_local)
            else:
                raise ValueError(f"Joint '{name}' not found in robot model")
        self._dofs_idx = np.array(dofs_idx)

    def _setup_control_gains(self) -> None:
        """Configure PD control gains for joints."""
        # Arm gains
        arm_kp = np.full(CraneX7Config.NUM_ARM_JOINTS, CraneX7Config.ARM_STIFFNESS)
        arm_kv = np.full(CraneX7Config.NUM_ARM_JOINTS, CraneX7Config.ARM_DAMPING)

        # Gripper gains
        gripper_kp = np.full(
            CraneX7Config.NUM_GRIPPER_JOINTS, CraneX7Config.GRIPPER_STIFFNESS
        )
        gripper_kv = np.full(
            CraneX7Config.NUM_GRIPPER_JOINTS, CraneX7Config.GRIPPER_DAMPING
        )

        # Concatenate
        kp = np.concatenate([arm_kp, gripper_kp])
        kv = np.concatenate([arm_kv, gripper_kv])

        # Set gains
        self._robot.set_dofs_kp(kp=kp, dofs_idx_local=self._dofs_idx)
        self._robot.set_dofs_kv(kv=kv, dofs_idx_local=self._dofs_idx)

    def _setup_camera(self) -> None:
        """Setup camera for rendering."""
        if self.config.render_mode == "none":
            self._camera = None
            return

        # Camera mounted on gripper base looking forward
        # Position relative to gripper base link
        self._camera = self._scene.add_camera(
            res=(640, 480),
            pos=(0.3, 0.0, 0.4),  # Offset from origin
            lookat=(0.15, 0.0, 0.1),  # Looking at workspace
            fov=69,  # Similar to RealSense D435
            GUI=False,
        )

    def _init_environment(self) -> None:
        """Initialize task environment."""
        _register_environments()

        env_cls = ENV_REGISTRY.get(self.config.env_id)
        if env_cls is None:
            available = list(ENV_REGISTRY.keys())
            raise ValueError(
                f"Unknown environment: '{self.config.env_id}'. "
                f"Available: {available}"
            )

        self._environment = env_cls(
            scene=self._scene,
            robot=self._robot,
            robot_init_qpos_noise=self.config.robot_init_qpos_noise,
        )
        self._environment.setup_scene()

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action with software mimic control for gripper (batched).

        Args:
            action: Shape (n_envs, 8) for 7 arm + 1 gripper value,
                   or (n_envs, 9) for full DOF.
                   When 8 DOF, gripper value is copied to both finger joints.
        """
        action = np.atleast_2d(action)
        n_envs, action_dim = action.shape

        if action_dim == 8:
            # Expand to 9 DOF: copy gripper value for finger_b (mimic)
            full_action = np.zeros((n_envs, 9))
            full_action[:, :7] = action[:, :7]  # Arm joints
            full_action[:, 7] = action[:, 7]  # finger_a
            full_action[:, 8] = action[:, 7]  # finger_b (mimic)
        elif action_dim == 9:
            full_action = action
        else:
            raise ValueError(
                f"Action must have 8 or 9 dimensions per env, got {action_dim}"
            )

        # Apply PD position control (batched)
        self._robot.control_dofs_position(full_action, self._dofs_idx)

    def _convert_observation(self) -> Observation:
        """Convert Genesis state to lift Observation (batched).

        Returns:
            Unified Observation object with batched data.
            - rgb_image: (n_envs, H, W, 3) if available
            - depth_image: (n_envs, H, W) if available
            - qpos: (n_envs, 9)
            - qvel: (n_envs, 9)
        """
        rgb_images = None
        depth_images = None

        # Render camera if available
        if self._camera is not None and self.config.render_mode != "none":
            render_output = self._camera.render(depth=True)
            if isinstance(render_output, tuple):
                # Genesis may return (rgb, depth, segmentation, ...) - take first two
                rgb = render_output[0]
                depth = render_output[1] if len(render_output) > 1 else None
            else:
                rgb = render_output
                depth = None

            if rgb is not None:
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8)

                # Handle batch dimension for multiple environments
                # Genesis with n_envs > 1 should return (n_envs, H, W, 3)
                # but if it returns (H, W, 3), we need to tile it
                if rgb.ndim == 3:
                    # Single image - tile for all envs
                    rgb = np.tile(rgb[np.newaxis, ...], (self._n_envs, 1, 1, 1))
                elif rgb.ndim == 4 and rgb.shape[0] != self._n_envs:
                    print(
                        f"[Genesis] Warning: RGB shape {rgb.shape} doesn't match n_envs={self._n_envs}"
                    )
                rgb_images = rgb

            if depth is not None:
                if hasattr(depth, "cpu"):
                    depth = depth.cpu().numpy()
                depth = np.asarray(depth, dtype=np.float32)
                # Handle batch dimension
                if depth.ndim == 2:
                    depth = np.tile(depth[np.newaxis, ...], (self._n_envs, 1, 1))
                depth_images = depth

        # Get joint states (already batched)
        qpos = self.get_qpos()
        qvel = self.get_qvel()

        # Build extra info
        extra: dict[str, Any] = {}
        if self._environment is not None:
            extra = self._environment.get_info()

        return Observation(
            rgb_image=rgb_images,
            depth_image=depth_images,
            qpos=qpos,
            qvel=qvel,
            extra=extra,
        )
