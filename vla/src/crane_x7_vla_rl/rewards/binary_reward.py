# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Binary reward function for task success."""

from typing import Any

import numpy as np

from crane_x7_vla_rl.rewards.base import RewardFunction


class BinaryRewardFunction(RewardFunction):
    """Binary reward: 1.0 on task success, 0.0 otherwise.

    This follows SimpleVLA-RL's approach of using outcome-based rewards
    without complex reward shaping.
    """

    def __init__(
        self,
        success_key: str = "success",
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
    ):
        """Initialize binary reward function.

        Args:
            success_key: Key in info dict indicating success.
            success_reward: Reward value on success.
            failure_reward: Reward value on failure.
        """
        self.success_key = success_key
        self.success_reward = success_reward
        self.failure_reward = failure_reward

    def compute(self, info: dict[str, Any]) -> float:
        """Compute binary reward from step info.

        Args:
            info: Information dictionary from environment step.

        Returns:
            success_reward if success, failure_reward otherwise.
        """
        success = info.get(self.success_key, False)

        # Handle tensor/array values
        if isinstance(success, np.ndarray):
            success = success.item() if success.size == 1 else success[0]
        elif hasattr(success, "item"):
            success = success.item()

        return self.success_reward if success else self.failure_reward


class SparseRewardFunction(RewardFunction):
    """Sparse reward: only given at episode end based on success.

    Unlike BinaryRewardFunction which gives reward at success step,
    this gives reward only at the final step of an episode.
    """

    def __init__(
        self,
        success_key: str = "success",
        terminated_key: str = "terminated",
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
    ):
        """Initialize sparse reward function.

        Args:
            success_key: Key in info dict indicating success.
            terminated_key: Key in info dict indicating episode end.
            success_reward: Reward value on success.
            failure_reward: Reward value on failure.
        """
        self.success_key = success_key
        self.terminated_key = terminated_key
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self._episode_success = False

    def compute(self, info: dict[str, Any]) -> float:
        """Compute sparse reward from step info.

        Args:
            info: Information dictionary from environment step.

        Returns:
            Reward only at episode end, 0.0 otherwise.
        """
        success = info.get(self.success_key, False)
        if isinstance(success, np.ndarray):
            success = success.item() if success.size == 1 else success[0]

        # Track if success happened during episode
        if success:
            self._episode_success = True

        # Check if episode ended
        terminated = info.get(self.terminated_key, False)
        truncated = info.get("truncated", False)
        done = terminated or truncated

        if done:
            reward = (
                self.success_reward if self._episode_success else self.failure_reward
            )
            return reward

        return 0.0

    def reset(self) -> None:
        """Reset episode success tracking."""
        self._episode_success = False


class DenseRewardWrapper(RewardFunction):
    """Wrapper that adds dense reward to a base reward function.

    This allows combining binary/sparse rewards with dense rewards
    from the simulator for faster learning.
    """

    def __init__(
        self,
        base_reward: RewardFunction,
        dense_weight: float = 0.1,
        dense_key: str = "dense_reward",
    ):
        """Initialize dense reward wrapper.

        Args:
            base_reward: Base reward function (e.g., BinaryRewardFunction).
            dense_weight: Weight for dense reward.
            dense_key: Key in info dict for dense reward.
        """
        self.base_reward = base_reward
        self.dense_weight = dense_weight
        self.dense_key = dense_key

    def compute(self, info: dict[str, Any]) -> float:
        """Compute combined reward.

        Args:
            info: Information dictionary from environment step.

        Returns:
            base_reward + dense_weight * dense_reward.
        """
        base = self.base_reward.compute(info)

        dense = info.get(self.dense_key, 0.0)
        if isinstance(dense, np.ndarray):
            dense = dense.item() if dense.size == 1 else dense[0]

        return base + self.dense_weight * float(dense)

    def reset(self) -> None:
        """Reset base reward function."""
        self.base_reward.reset()
