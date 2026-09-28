# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Parallel environment management using Genesis batch parallelization."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from crane_x7_vla_rl.config.rollout_config import RolloutConfig

# Import lift modules
from lift import SimulatorConfig, create_simulator
from lift.types import Observation


def _ensure_simulator_registered(simulator_name: str) -> None:
    """Import the simulator module to ensure it's registered."""
    import importlib

    module_map = {
        "maniskill": "lift_maniskill",
        "genesis": "lift_genesis",
        "isaacsim": "lift_isaacsim",
    }
    if simulator_name in module_map:
        importlib.import_module(module_map[simulator_name])


@dataclass
class BatchObservation:
    """Batched observations from parallel environments."""

    images: np.ndarray
    """Stacked RGB images (N, H, W, 3) uint8."""

    states: np.ndarray
    """Stacked robot states (N, state_dim)."""

    extras: list[dict[str, Any]]
    """List of extra info dicts."""


@dataclass
class BatchStepResult:
    """Batched step results from parallel environments."""

    observations: BatchObservation
    """Batched observations."""

    rewards: np.ndarray
    """Rewards (N,)."""

    terminateds: np.ndarray
    """Terminated flags (N,)."""

    truncateds: np.ndarray
    """Truncated flags (N,)."""

    infos: list[dict[str, Any]]
    """List of info dicts."""


class ParallelLiftEnvironments:
    """Manage parallel environments using Genesis batch parallelization.

    This class uses Genesis's native n_envs support for efficient parallel
    simulation in a single scene, avoiding the overhead of multiple instances.
    """

    def __init__(
        self,
        num_envs: int,
        config: RolloutConfig,
    ):
        """Initialize parallel environments with batch parallelization.

        Args:
            num_envs: Number of parallel environments.
            config: Rollout configuration.
        """
        self.num_envs = num_envs
        self.config = config

        # Ensure simulator module is registered
        _ensure_simulator_registered(config.simulator)

        # Create single batched simulator
        sim_config = SimulatorConfig(
            env_id=config.env_id,
            backend=config.backend,
            render_mode=config.render_mode,
            max_episode_steps=config.max_steps,
            robot_init_qpos_noise=0.02,
            n_envs=num_envs,  # Use batch parallelization
        )
        self._simulator = create_simulator(config.simulator, sim_config)

        # Reward settings
        self._use_binary_reward = config.use_binary_reward
        self._dense_reward_weight = config.dense_reward_weight

        # Track active episodes
        self._active_mask = np.ones(num_envs, dtype=bool)
        self._episode_steps = np.zeros(num_envs, dtype=np.int32)
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)

    def reset_all(self, seed: int | None = None) -> BatchObservation:
        """Reset all environments.

        Args:
            seed: Optional random seed.

        Returns:
            Batched observations.
        """
        obs, info = self._simulator.reset(seed=seed)
        self._active_mask = np.ones(self.num_envs, dtype=bool)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        return self._convert_batch_observation(obs, info)

    def reset_done(
        self,
        done_mask: np.ndarray,
        seed: int | None = None,
    ) -> None:
        """Reset environments that are done.

        Args:
            done_mask: Boolean mask of environments to reset (N,).
            seed: Optional random seed.
        """
        env_ids = np.where(done_mask)[0]
        if len(env_ids) > 0:
            self._simulator.reset(seed=seed, env_ids=env_ids)
            self._episode_steps[env_ids] = 0
            self._episode_rewards[env_ids] = 0.0

    def step_all(self, actions: np.ndarray) -> BatchStepResult:
        """Step all environments.

        Args:
            actions: Actions for all environments (N, action_dim).

        Returns:
            Batched step results.
        """
        result = self._simulator.step(actions)
        self._episode_steps += 1

        # Compute rewards
        rewards = self._compute_rewards(result.reward, result.info)
        self._episode_rewards += rewards

        # Convert observations
        batch_obs = self._convert_batch_observation(result.observation, result.info)

        # Get infos list
        if isinstance(result.info, list):
            infos = result.info
        else:
            infos = [result.info] * self.num_envs

        # Add episode info to each
        for i, info in enumerate(infos):
            info["step_count"] = int(self._episode_steps[i])
            info["episode_reward"] = float(self._episode_rewards[i])

        return BatchStepResult(
            observations=batch_obs,
            rewards=rewards,
            terminateds=np.atleast_1d(result.terminated),
            truncateds=np.atleast_1d(result.truncated),
            infos=infos,
        )

    def _convert_batch_observation(
        self,
        obs: Observation,
        info: Any,
    ) -> BatchObservation:
        """Convert lift Observation to BatchObservation.

        Args:
            obs: lift Observation object (with batched data).
            info: Info dict or list of dicts.

        Returns:
            BatchObservation for VLA model input.
        """
        # Get RGB images (N, H, W, 3)
        if obs.rgb_image is not None:
            images = obs.rgb_image
            # Ensure batch dimension
            if images.ndim == 3:
                images = images[np.newaxis, ...]
        else:
            # Fallback: create dummy images
            images = np.zeros((self.num_envs, 224, 224, 3), dtype=np.uint8)

        # Get robot states (N, 9)
        if obs.qpos is not None:
            states = obs.qpos
            # Ensure batch dimension
            if states.ndim == 1:
                states = states[np.newaxis, ...]
        else:
            states = np.zeros((self.num_envs, 9), dtype=np.float32)

        # Get extras
        if isinstance(info, list):
            extras = info
        elif isinstance(info, dict):
            extras = [info] * self.num_envs
        else:
            extras = [{}] * self.num_envs

        return BatchObservation(
            images=images,
            states=states,
            extras=extras,
        )

    def _compute_rewards(
        self,
        raw_rewards: np.ndarray,
        infos: Any,
    ) -> np.ndarray:
        """Compute rewards from step results.

        Args:
            raw_rewards: Raw rewards from simulator (N,).
            infos: Info dict or list of dicts.

        Returns:
            Computed rewards (N,).
        """
        raw_rewards = np.atleast_1d(raw_rewards).astype(np.float32)

        if self._use_binary_reward:
            # Binary reward: 1.0 on success, 0.0 otherwise
            if isinstance(infos, list):
                successes = np.array(
                    [info.get("success", False) for info in infos], dtype=bool
                )
            else:
                success = infos.get("success", False)
                if isinstance(success, np.ndarray):
                    successes = success
                else:
                    successes = np.full(self.num_envs, success, dtype=bool)

            rewards = successes.astype(np.float32)
        else:
            rewards = np.zeros(self.num_envs, dtype=np.float32)

        if self._dense_reward_weight > 0:
            rewards += self._dense_reward_weight * raw_rewards

        return rewards

    def close(self) -> None:
        """Release resources."""
        self._simulator.close()

    def __enter__(self) -> "ParallelLiftEnvironments":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
