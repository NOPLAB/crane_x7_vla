# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Base reward function interface."""

from abc import ABC, abstractmethod
from typing import Any


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(self, info: dict[str, Any]) -> float:
        """Compute reward from step info.

        Args:
            info: Information dictionary from environment step.

        Returns:
            Computed reward value.
        """
        pass

    def reset(self) -> None:
        """Reset any internal state (called at episode start)."""
        pass

    def __call__(self, info: dict[str, Any]) -> float:
        """Compute reward (callable interface).

        Args:
            info: Information dictionary from environment step.

        Returns:
            Computed reward value.
        """
        return self.compute(info)
