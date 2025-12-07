# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Type definitions for crane_x7_vla."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np


class TrainingResult(TypedDict):
    """Result of a training run."""

    final_step: int
    """Final training step number"""

    final_epoch: int
    """Final epoch number"""

    checkpoint_dir: str
    """Path to checkpoint directory"""

    best_checkpoint: str | None
    """Path to best checkpoint (if available)"""


class EvaluationResult(TypedDict):
    """Result of model evaluation."""

    loss: float
    """Average loss on evaluation set"""

    action_accuracy: float
    """Action prediction accuracy"""

    l1_loss: float
    """L1 loss on continuous actions"""

    num_samples: int
    """Number of samples evaluated"""


class OverfitMetrics(TypedDict):
    """Metrics for overfitting detection."""

    overfit_loss: float
    """Loss on held-out steps"""

    overfit_action_accuracy: float
    """Action accuracy on held-out steps"""

    overfit_l1_loss: float
    """L1 loss on held-out steps"""


@dataclass
class Observation:
    """
    Standardized observation data structure.

    Used for inference across all backends.
    """

    state: np.ndarray
    """Robot state (joint positions, etc.) - shape: (state_dim,)"""

    image: np.ndarray | dict[str, np.ndarray]
    """RGB image(s) - shape: (H, W, 3) or dict of camera_name -> image"""

    depth: np.ndarray | dict[str, np.ndarray] | None = None
    """Depth image(s) (optional) - shape: (H, W) or dict of camera_name -> depth"""

    timestamp: float | None = None
    """Timestamp of observation"""


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    path: Path
    """Path to checkpoint directory"""

    backend: Literal["openvla", "openpi", "openpi-pytorch"]
    """Backend type that created this checkpoint"""

    step: int
    """Training step when checkpoint was saved"""

    is_valid: bool
    """Whether checkpoint passed validation"""

    has_optimizer_state: bool
    """Whether checkpoint contains optimizer state"""

    has_config: bool
    """Whether checkpoint contains config file"""


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    num_episodes: int
    """Number of episodes in dataset"""

    num_transitions: int
    """Total number of transitions"""

    action_dim: int
    """Action dimension"""

    state_dim: int
    """State dimension"""

    has_images: bool
    """Whether dataset contains images"""

    has_depth: bool
    """Whether dataset contains depth images"""

    image_size: tuple[int, int] | None
    """Image size (H, W) if available"""
