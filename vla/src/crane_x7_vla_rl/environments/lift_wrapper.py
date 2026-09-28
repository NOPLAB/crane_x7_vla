# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Wrapper for lift simulator integration with VLA-RL."""

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from lift import Simulator, SimulatorConfig, create_simulator
from lift.types import Observation, StepResult

logger = logging.getLogger(__name__)

# Mapping from simulator name to module name
_SIMULATOR_MODULES = {
    "maniskill": "lift_maniskill",
    "genesis": "lift_genesis",
    "isaacsim": "lift_isaacsim",
}


def _ensure_simulator_registered(simulator_name: str) -> None:
    """Import the simulator module to ensure it's registered.

    Args:
        simulator_name: Name of the simulator (maniskill, genesis, isaacsim).

    Raises:
        ImportError: If the simulator module cannot be imported.
    """
    if simulator_name not in _SIMULATOR_MODULES:
        logger.warning(f"Unknown simulator: {simulator_name}")
        return

    module_name = _SIMULATOR_MODULES[simulator_name]
    try:
        importlib.import_module(module_name)
        logger.debug(f"Successfully imported {module_name}")
    except ImportError as e:
        raise ImportError(
            f"Could not import simulator module '{module_name}' for simulator "
            f"'{simulator_name}'. Please ensure the required dependencies are "
            f"installed. Error: {e}"
        ) from e


@dataclass
class VLARLObservation:
    """Observation format for VLA-RL training."""

    image: np.ndarray
    """RGB image from camera (H, W, 3) uint8."""

    state: np.ndarray
    """Robot joint positions."""

    extra: dict[str, Any]
    """Additional information (task metrics, etc.)."""


class LiftRolloutEnvironment:
    """Adapter between lift Simulator and VLA-RL rollout interface.

    This class wraps the lift simulator to provide a consistent interface
    for VLA-RL training, converting observations and handling episode logic.
    """

    def __init__(
        self,
        simulator: Simulator,
        use_binary_reward: bool = True,
        dense_reward_weight: float = 0.0,
    ):
        """Initialize the lift rollout environment.

        Args:
            simulator: lift Simulator instance.
            use_binary_reward: Whether to use binary (0/1) success reward.
            dense_reward_weight: Weight for dense reward (added to binary).
        """
        self.simulator = simulator
        self.use_binary_reward = use_binary_reward
        self.dense_reward_weight = dense_reward_weight
        self._step_count = 0
        self._episode_reward = 0.0

    @classmethod
    def from_config(
        cls,
        env_id: str = "PickPlace-CRANE-X7",
        simulator_name: str = "maniskill",
        backend: str = "cpu",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 200,
        **kwargs,
    ) -> "LiftRolloutEnvironment":
        """Create environment from configuration.

        Args:
            env_id: Environment identifier.
            simulator_name: Simulator backend (maniskill, genesis).
            backend: Compute backend (cpu, gpu).
            render_mode: Render mode (rgb_array, human, none).
            max_episode_steps: Maximum steps per episode.
            **kwargs: Additional arguments for LiftRolloutEnvironment.

        Returns:
            Configured LiftRolloutEnvironment instance.
        """
        # Import the simulator module to ensure it's registered
        _ensure_simulator_registered(simulator_name)

        config = SimulatorConfig(
            env_id=env_id,
            backend=backend,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        simulator = create_simulator(simulator_name, config)
        return cls(simulator, **kwargs)

    def reset(self, seed: int | None = None) -> tuple[VLARLObservation, dict[str, Any]]:
        """Reset environment and return initial observation.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observation, info dict).
        """
        obs, info = self.simulator.reset(seed=seed)
        self._step_count = 0
        self._episode_reward = 0.0
        return self._convert_observation(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[VLARLObservation, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action array to execute (8 DOF for CRANE-X7).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        result: StepResult = self.simulator.step(action)
        self._step_count += 1

        # Compute reward
        reward = self._compute_reward(result)
        self._episode_reward += reward

        # Convert observation
        obs = self._convert_observation(result.observation)

        # Add episode info
        info = result.info.copy()
        info["step_count"] = self._step_count
        info["episode_reward"] = self._episode_reward

        return obs, reward, result.terminated, result.truncated, info

    def _convert_observation(self, obs: Observation) -> VLARLObservation:
        """Convert lift Observation to VLARLObservation.

        Args:
            obs: lift Observation object.

        Returns:
            VLARLObservation for VLA model input.
        """
        # Get RGB image (required for VLA)
        if obs.rgb_image is not None:
            image = obs.rgb_image
        else:
            # Fallback: create dummy image if no camera
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Get robot state
        state = obs.qpos if obs.qpos is not None else np.zeros(9)

        return VLARLObservation(
            image=image,
            state=state,
            extra=obs.extra or {},
        )

    def _compute_reward(self, result: StepResult) -> float:
        """Compute reward from step result.

        Args:
            result: StepResult from simulator.

        Returns:
            Computed reward value.
        """
        reward = 0.0

        if self.use_binary_reward:
            # Binary reward: 1.0 on success, 0.0 otherwise
            success = result.info.get("success", False)
            if isinstance(success, np.ndarray):
                success = success.item() if success.size == 1 else success[0]
            reward = 1.0 if success else 0.0

        if self.dense_reward_weight > 0:
            # Add weighted dense reward from simulator
            dense_reward = result.reward
            if isinstance(dense_reward, np.ndarray):
                dense_reward = (
                    dense_reward.item() if dense_reward.size == 1 else dense_reward[0]
                )
            reward += self.dense_reward_weight * dense_reward

        return reward

    def get_observation(self) -> VLARLObservation:
        """Get current observation without stepping.

        Returns:
            Current observation.
        """
        obs = self.simulator.get_observation()
        return self._convert_observation(obs)

    def close(self) -> None:
        """Release environment resources."""
        self.simulator.close()

    @property
    def action_dim(self) -> int:
        """Get action dimension (8 for CRANE-X7: 7 arm + 1 gripper)."""
        return 8

    @property
    def state_dim(self) -> int:
        """Get state dimension (9 for CRANE-X7: 7 arm + 2 gripper fingers)."""
        return 9

    @property
    def max_episode_steps(self) -> int:
        """Get maximum episode steps."""
        return self.simulator.config.max_episode_steps

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self.simulator.is_running
