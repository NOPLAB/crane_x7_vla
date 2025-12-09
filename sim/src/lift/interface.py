# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Abstract interface for simulators."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from lift.types import Observation, SimulatorConfig, StepResult


class Simulator(ABC):
    """Abstract base class for all simulators.

    This interface provides a unified API for different simulation backends
    (ManiSkill, Genesis, Isaac Sim, etc.).
    """

    def __init__(self, config: SimulatorConfig):
        """Initialize the simulator with configuration.

        Args:
            config: Simulator configuration.
        """
        self.config = config
        self._is_running = False

    @property
    @abstractmethod
    def arm_joint_names(self) -> list[str]:
        """Return list of arm joint names."""
        pass

    @property
    @abstractmethod
    def gripper_joint_names(self) -> list[str]:
        """Return list of gripper joint names."""
        pass

    @property
    def all_joint_names(self) -> list[str]:
        """Return list of all joint names (arm + gripper)."""
        return self.arm_joint_names + self.gripper_joint_names

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and return initial observation.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observation, info dict).
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Execute one simulation step.

        Args:
            action: Action array to execute.

        Returns:
            StepResult containing observation, reward, and done flags.
        """
        pass

    @abstractmethod
    def get_observation(self) -> Observation:
        """Get current observation without stepping.

        Returns:
            Current observation.
        """
        pass

    @abstractmethod
    def get_qpos(self) -> np.ndarray:
        """Get current joint positions.

        Returns:
            Array of joint positions.
        """
        pass

    @abstractmethod
    def get_qvel(self) -> np.ndarray:
        """Get current joint velocities.

        Returns:
            Array of joint velocities.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""
        pass

    @property
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        """Set simulation running state."""
        self._is_running = value
