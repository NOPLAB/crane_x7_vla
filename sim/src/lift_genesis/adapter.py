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
    """Genesis simulator implementation."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)

        # Internal state
        self._gs = None  # Genesis module
        self._scene = None
        self._robot = None
        self._camera = None
        self._environment = None
        self._dofs_idx: Optional[np.ndarray] = None
        self._current_obs = None
        self._episode_step = 0
        self._max_episode_steps = config.max_episode_steps

        # Initialize Genesis and environment
        self._init_genesis()
        self._init_scene()
        self._add_robot()
        self._setup_camera()
        self._init_environment()

        # Build scene
        self._scene.build()

        # Configure robot after build (requires built scene)
        self._build_dof_mapping()
        self._setup_control_gains()

        # Initial reset to populate observation
        self.reset()
        self._is_running = True

    @property
    def arm_joint_names(self) -> list[str]:
        return CraneX7Config.ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return CraneX7Config.GRIPPER_JOINT_NAMES

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (initial observation, info dict).
        """
        self._episode_step = 0

        # Set robot to rest position with noise
        rest_qpos = CraneX7Config.REST_QPOS.copy()
        if self.config.robot_init_qpos_noise > 0 and seed is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, self.config.robot_init_qpos_noise, rest_qpos.shape)
            rest_qpos = rest_qpos + noise

        self._robot.set_dofs_position(rest_qpos, self._dofs_idx)
        self._robot.set_dofs_velocity(np.zeros(len(self._dofs_idx)), self._dofs_idx)

        # Reset environment (task-specific objects)
        info = {}
        if self._environment is not None:
            env_info = self._environment.reset(seed=seed)
            info.update(env_info)

        # Step once to settle physics
        self._scene.step()

        # Get observation
        self._current_obs = self._convert_observation()

        return self._current_obs, info

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one simulation step.

        Args:
            action: Joint position targets. Shape (8,) for 7 arm + 1 gripper,
                   or (9,) for full DOF.

        Returns:
            StepResult containing observation, reward, terminated, truncated, info.
        """
        # Apply action with mimic control
        self._apply_action(action)

        # Step simulation
        self._scene.step()
        self._episode_step += 1

        # Get observation
        self._current_obs = self._convert_observation()

        # Compute reward and check termination
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if self._environment is not None:
            reward = self._environment.compute_reward()
            terminated = self._environment.is_terminated()
            info = self._environment.get_info()

        # Check truncation (max episode steps)
        truncated = self._episode_step >= self._max_episode_steps

        return StepResult(
            observation=self._current_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_observation(self) -> Observation:
        """Get current observation without stepping.

        Returns:
            Current observation.
        """
        return self._current_obs

    def get_qpos(self) -> np.ndarray:
        """Get current joint positions.

        Returns:
            Joint positions array of shape (9,).
        """
        qpos = self._robot.get_dofs_position(self._dofs_idx)
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        return np.asarray(qpos).flatten()

    def get_qvel(self) -> np.ndarray:
        """Get current joint velocities.

        Returns:
            Joint velocities array of shape (9,).
        """
        qvel = self._robot.get_dofs_velocity(self._dofs_idx)
        if hasattr(qvel, "cpu"):
            qvel = qvel.cpu().numpy()
        return np.asarray(qvel).flatten()

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
        if self.config.backend == "gpu":
            gs.init(backend=gs.cuda)
        else:
            gs.init(backend=gs.cpu)

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
        gripper_kp = np.full(CraneX7Config.NUM_GRIPPER_JOINTS, CraneX7Config.GRIPPER_STIFFNESS)
        gripper_kv = np.full(CraneX7Config.NUM_GRIPPER_JOINTS, CraneX7Config.GRIPPER_DAMPING)

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
        """Apply action with software mimic control for gripper.

        Args:
            action: Shape (8,) for 7 arm + 1 gripper value, or (9,) for full DOF.
                   When 8 DOF, gripper value is copied to both finger joints.
        """
        action = np.asarray(action).flatten()

        if len(action) == 8:
            # Expand to 9 DOF: copy gripper value for finger_b (mimic)
            full_action = np.zeros(9)
            full_action[:7] = action[:7]  # Arm joints
            full_action[7] = action[7]  # finger_a
            full_action[8] = action[7]  # finger_b (mimic)
        elif len(action) == 9:
            full_action = action
        else:
            raise ValueError(
                f"Action must have 8 or 9 dimensions, got {len(action)}"
            )

        # Apply PD position control
        self._robot.control_dofs_position(full_action, self._dofs_idx)

    def _convert_observation(self) -> Observation:
        """Convert Genesis state to lift Observation.

        Returns:
            Unified Observation object.
        """
        rgb_image = None
        depth_image = None

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
                rgb_image = rgb

            if depth is not None:
                if hasattr(depth, "cpu"):
                    depth = depth.cpu().numpy()
                depth_image = np.asarray(depth, dtype=np.float32)

        # Get joint states
        qpos = self.get_qpos()
        qvel = self.get_qvel()

        # Build extra info
        extra: dict[str, Any] = {}
        if self._environment is not None:
            extra = self._environment.get_info()

        return Observation(
            rgb_image=rgb_image,
            depth_image=depth_image,
            qpos=qpos,
            qvel=qvel,
            extra=extra,
        )
