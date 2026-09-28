# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Common types for the lift abstraction layer."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SimulatorConfig:
    """Configuration for simulator initialization."""

    env_id: str
    backend: str = "cpu"
    render_mode: str = "rgb_array"
    control_mode: str = "pd_joint_pos"
    sim_rate: float = 30.0
    max_episode_steps: int = 200
    robot_init_qpos_noise: float = 0.02
    n_envs: int = 1  # Number of parallel environments for batch parallelization


@dataclass
class Observation:
    """Unified observation structure across simulators."""

    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = None
    qvel: Optional[np.ndarray] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a simulation step."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
