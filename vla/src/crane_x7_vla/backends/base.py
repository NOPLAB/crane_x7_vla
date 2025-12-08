# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Abstract base class for VLA backends.

This module defines the interface that all VLA backends (OpenVLA, OpenPI, etc.)
must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from crane_x7_vla.data_types import EvaluationResult, Observation, TrainingResult


class VLABackend(ABC):
    """
    Abstract base class for Vision-Language-Action model backends.

    All VLA implementations (OpenVLA, OpenPI, etc.) must inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: Any):
        """
        Initialize the VLA backend.

        Args:
            config: Configuration object (backend-specific)
        """
        self.config = config
        self.model = None
        self.processor = None

    @abstractmethod
    def train(self) -> TrainingResult:
        """
        Execute the training loop.

        Returns:
            TrainingResult containing training metrics and checkpoint paths
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None
    ) -> EvaluationResult:
        """
        Evaluate the model on test data.

        Args:
            checkpoint_path: Path to model checkpoint (optional)
            test_data_path: Path to test dataset (optional)

        Returns:
            EvaluationResult containing evaluation metrics
        """
        pass

    @abstractmethod
    def infer(
        self,
        observation: Union[Observation, Dict[str, np.ndarray]],
        language_instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform inference on a single observation.

        Args:
            observation: Observation object or dictionary containing:
                - 'state': Robot state (joint positions, etc.)
                - 'image': RGB image(s)
                - 'depth': Depth image(s) (optional)
            language_instruction: Task instruction (optional)

        Returns:
            Predicted action as numpy array
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        Get the action dimension of the model.

        Returns:
            Action dimension
        """
        pass

    @property
    @abstractmethod
    def action_horizon(self) -> int:
        """
        Get the action horizon (number of future actions predicted).

        Returns:
            Action horizon (1 for single-step, >1 for action chunking)
        """
        pass

    @property
    @abstractmethod
    def expected_image_size(self) -> tuple:
        """
        Get the expected image size for the model.

        Returns:
            Tuple of (height, width)
        """
        pass

    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"{self.__class__.__name__}(action_dim={self.action_dim}, action_horizon={self.action_horizon})"
