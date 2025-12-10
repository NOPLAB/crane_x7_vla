# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Trajectory buffer for storing and managing rollout data."""

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    """A single episode trajectory.

    Stores all data needed for PPO training from one episode.
    """

    observations: list[dict[str, np.ndarray]] = field(default_factory=list)
    """Observations at each timestep (list of {image, state} dicts)."""

    actions: list[np.ndarray] = field(default_factory=list)
    """Actions taken at each timestep."""

    rewards: list[float] = field(default_factory=list)
    """Rewards received at each timestep."""

    log_probs: list[float] = field(default_factory=list)
    """Log probabilities of actions under behavior policy."""

    values: list[float] = field(default_factory=list)
    """Value estimates at each timestep."""

    dones: list[bool] = field(default_factory=list)
    """Done flags at each timestep."""

    infos: list[dict[str, Any]] = field(default_factory=list)
    """Info dicts at each timestep."""

    def append(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Append a single timestep to the trajectory.

        Args:
            observation: Observation dict with image and state.
            action: Action array.
            reward: Reward value.
            log_prob: Log probability of action.
            value: Value estimate.
            done: Whether episode ended.
            info: Optional info dict.
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.infos.append(info or {})

    def __len__(self) -> int:
        """Get trajectory length."""
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """Get total episode reward."""
        return sum(self.rewards)

    @property
    def success(self) -> bool:
        """Check if episode was successful."""
        for info in self.infos:
            if info.get("success", False):
                return True
        return False

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert trajectory to numpy arrays.

        Returns:
            Dict with arrays for each trajectory component.
        """
        # Stack observations
        images = np.stack([obs["image"] for obs in self.observations])
        states = np.stack([obs["state"] for obs in self.observations])

        return {
            "images": images,
            "states": states,
            "actions": np.stack(self.actions),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "dones": np.array(self.dones, dtype=bool),
        }


class TrajectoryBuffer:
    """Buffer for storing multiple trajectories.

    Used for collecting rollout data before PPO updates.
    """

    def __init__(self, max_trajectories: int | None = None):
        """Initialize trajectory buffer.

        Args:
            max_trajectories: Maximum trajectories to store (None for unlimited).
        """
        self.trajectories: list[Trajectory] = []
        self.max_trajectories = max_trajectories

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer.

        Args:
            trajectory: Completed trajectory to add.
        """
        self.trajectories.append(trajectory)

        # Remove oldest if over limit
        if self.max_trajectories is not None:
            while len(self.trajectories) > self.max_trajectories:
                self.trajectories.pop(0)

    def clear(self) -> None:
        """Clear all trajectories from buffer."""
        self.trajectories.clear()

    def sample(self, n: int) -> list[Trajectory]:
        """Sample n trajectories from buffer.

        Args:
            n: Number of trajectories to sample.

        Returns:
            List of sampled trajectories.
        """
        return random.sample(self.trajectories, min(n, len(self.trajectories)))

    def get_all(self) -> list[Trajectory]:
        """Get all trajectories in buffer.

        Returns:
            List of all trajectories.
        """
        return list(self.trajectories)

    def to_batch(self) -> dict[str, np.ndarray]:
        """Convert all trajectories to a single batch.

        Concatenates all trajectories into arrays suitable for PPO training.

        Returns:
            Dict with concatenated arrays.
        """
        if not self.trajectories:
            return {}

        # Convert each trajectory
        all_arrays = [traj.to_arrays() for traj in self.trajectories]

        # Concatenate
        batch = {}
        for key in all_arrays[0].keys():
            batch[key] = np.concatenate([arr[key] for arr in all_arrays], axis=0)

        return batch

    def __len__(self) -> int:
        """Get number of trajectories."""
        return len(self.trajectories)

    @property
    def total_steps(self) -> int:
        """Get total number of timesteps across all trajectories."""
        return sum(len(traj) for traj in self.trajectories)

    @property
    def success_rate(self) -> float:
        """Get success rate across all trajectories."""
        if not self.trajectories:
            return 0.0
        successes = sum(traj.success for traj in self.trajectories)
        return successes / len(self.trajectories)

    @property
    def mean_reward(self) -> float:
        """Get mean total reward across trajectories."""
        if not self.trajectories:
            return 0.0
        return np.mean([traj.total_reward for traj in self.trajectories])

    @property
    def mean_length(self) -> float:
        """Get mean trajectory length."""
        if not self.trajectories:
            return 0.0
        return np.mean([len(traj) for traj in self.trajectories])

    def get_statistics(self) -> dict[str, float]:
        """Get buffer statistics.

        Returns:
            Dict with buffer statistics.
        """
        return {
            "num_trajectories": len(self.trajectories),
            "total_steps": self.total_steps,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_length": self.mean_length,
        }
