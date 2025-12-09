# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Parallel environment management for efficient rollout."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from crane_x7_vlarl.config.rollout_config import RolloutConfig
from crane_x7_vlarl.environments.lift_wrapper import LiftRolloutEnvironment, VLARLObservation


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
    """Manage multiple lift simulator instances for parallel rollout.

    This class enables efficient parallel data collection by running
    multiple environments concurrently using a thread pool.
    """

    def __init__(
        self,
        num_envs: int,
        config: RolloutConfig,
        max_workers: int | None = None,
    ):
        """Initialize parallel environments.

        Args:
            num_envs: Number of parallel environments.
            config: Rollout configuration.
            max_workers: Maximum worker threads (defaults to num_envs).
        """
        self.num_envs = num_envs
        self.config = config
        self.max_workers = max_workers or num_envs

        # Create environments
        self.envs: list[LiftRolloutEnvironment] = []
        self._create_envs()

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Track active episodes
        self._active_mask = np.ones(num_envs, dtype=bool)

    def _create_envs(self) -> None:
        """Create all parallel environment instances."""
        for i in range(self.num_envs):
            env = LiftRolloutEnvironment.from_config(
                env_id=self.config.env_id,
                simulator_name=self.config.simulator,
                backend=self.config.backend,
                render_mode=self.config.render_mode,
                max_episode_steps=self.config.max_steps,
                use_binary_reward=self.config.use_binary_reward,
                dense_reward_weight=self.config.dense_reward_weight,
            )
            self.envs.append(env)

    def reset_all(self, seed: int | None = None) -> BatchObservation:
        """Reset all environments in parallel.

        Args:
            seed: Base random seed (each env gets seed + i).

        Returns:
            Batched initial observations.
        """
        observations: list[VLARLObservation] = []

        def reset_env(i: int, env: LiftRolloutEnvironment) -> VLARLObservation:
            env_seed = seed + i if seed is not None else None
            obs, _ = env.reset(seed=env_seed)
            return obs

        # Submit all reset tasks
        futures = {
            self.executor.submit(reset_env, i, env): i
            for i, env in enumerate(self.envs)
        }

        # Collect results in order
        results = [None] * self.num_envs
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

        observations = results
        self._active_mask = np.ones(self.num_envs, dtype=bool)

        return self._batch_observations(observations)

    def reset_done(self, dones: np.ndarray, seed: int | None = None) -> None:
        """Reset only environments that are done.

        Args:
            dones: Boolean array indicating which envs to reset.
            seed: Base random seed.
        """
        def reset_env(i: int, env: LiftRolloutEnvironment) -> VLARLObservation:
            env_seed = seed + i if seed is not None else None
            obs, _ = env.reset(seed=env_seed)
            return obs

        futures = {}
        for i, (env, done) in enumerate(zip(self.envs, dones)):
            if done:
                futures[self.executor.submit(reset_env, i, env)] = i

        # Wait for all resets to complete
        for future in as_completed(futures):
            future.result()

    def step_all(self, actions: np.ndarray) -> BatchStepResult:
        """Execute actions in all environments in parallel.

        Args:
            actions: Batched actions (N, action_dim).

        Returns:
            Batched step results.
        """
        def step_env(i: int, env: LiftRolloutEnvironment, action: np.ndarray):
            return env.step(action)

        # Submit all step tasks
        futures = {
            self.executor.submit(step_env, i, env, actions[i]): i
            for i, env in enumerate(self.envs)
        }

        # Collect results in order
        results = [None] * self.num_envs
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

        # Unpack results
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for obs, reward, terminated, truncated, info in results:
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return BatchStepResult(
            observations=self._batch_observations(observations),
            rewards=np.array(rewards, dtype=np.float32),
            terminateds=np.array(terminateds, dtype=bool),
            truncateds=np.array(truncateds, dtype=bool),
            infos=infos,
        )

    def step_active(
        self,
        actions: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> BatchStepResult:
        """Step only active (non-done) environments.

        Args:
            actions: Batched actions (N, action_dim).
            active_mask: Boolean mask for active envs (defaults to internal mask).

        Returns:
            Batched step results (inactive envs have zero rewards).
        """
        if active_mask is None:
            active_mask = self._active_mask

        def step_env(i: int, env: LiftRolloutEnvironment, action: np.ndarray):
            return env.step(action)

        # Submit tasks only for active environments
        futures = {}
        for i, (env, active) in enumerate(zip(self.envs, active_mask)):
            if active:
                futures[self.executor.submit(step_env, i, env, actions[i])] = i

        # Initialize results with defaults
        observations = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        # Collect results
        for future in as_completed(futures):
            idx = futures[future]
            obs, reward, terminated, truncated, info = future.result()
            observations[idx] = obs
            rewards[idx] = reward
            terminateds[idx] = terminated
            truncateds[idx] = truncated
            infos[idx] = info

            # Update active mask
            if terminated or truncated:
                self._active_mask[idx] = False

        # Fill in observations for inactive envs
        for i, obs in enumerate(observations):
            if obs is None:
                observations[i] = self.envs[i].get_observation()

        return BatchStepResult(
            observations=self._batch_observations(observations),
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            infos=infos,
        )

    def _batch_observations(self, observations: list[VLARLObservation]) -> BatchObservation:
        """Stack observations into batched format.

        Args:
            observations: List of individual observations.

        Returns:
            Batched observation.
        """
        images = np.stack([obs.image for obs in observations], axis=0)
        states = np.stack([obs.state for obs in observations], axis=0)
        extras = [obs.extra for obs in observations]

        return BatchObservation(images=images, states=states, extras=extras)

    def get_observations(self) -> BatchObservation:
        """Get current observations from all environments.

        Returns:
            Batched current observations.
        """
        observations = [env.get_observation() for env in self.envs]
        return self._batch_observations(observations)

    def close(self) -> None:
        """Release all environment resources."""
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)

    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return self.envs[0].action_dim if self.envs else 8

    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        return self.envs[0].state_dim if self.envs else 9

    @property
    def active_count(self) -> int:
        """Get number of currently active environments."""
        return int(self._active_mask.sum())

    def __len__(self) -> int:
        """Get number of environments."""
        return self.num_envs

    def __enter__(self) -> "ParallelLiftEnvironments":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
