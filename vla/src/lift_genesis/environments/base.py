# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Base class for Genesis task environments."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class GenesisEnvironment(ABC):
    """Abstract base class for Genesis task environments.

    This class defines the interface that all Genesis task environments
    must implement for use with the GenesisSimulator adapter.
    """

    def __init__(self, scene: Any, robot: Any, robot_init_qpos_noise: float = 0.02):
        """Initialize the environment.

        Args:
            scene: Genesis scene instance.
            robot: Genesis robot entity.
            robot_init_qpos_noise: Standard deviation for initial qpos noise.
        """
        self.scene = scene
        self.robot = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._rng = np.random.default_rng()

    @abstractmethod
    def setup_scene(self) -> None:
        """Add task-specific objects to the scene.

        Called once after the robot is loaded but before scene.build().
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset the task state for a new episode.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Info dictionary with task-specific reset information.
        """
        pass

    @abstractmethod
    def compute_reward(self) -> float:
        """Compute the current reward.

        Returns:
            Scalar reward value.
        """
        pass

    @abstractmethod
    def is_success(self) -> bool:
        """Check if the task has been successfully completed.

        Returns:
            True if the task is complete, False otherwise.
        """
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        """Check if the episode should terminate.

        Returns:
            True if the episode should end, False otherwise.
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """Get additional task-specific information.

        Returns:
            Dictionary with task metrics and state information.
        """
        return {}
