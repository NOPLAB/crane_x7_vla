# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Genesis simulator adapter (not yet implemented)."""

from typing import Any, Optional

import numpy as np

from lift.interface import Simulator
from lift.types import Observation, SimulatorConfig, StepResult
from lift.factory import register_simulator
from robot.crane_x7 import CraneX7Config


@register_simulator("genesis")
class GenesisSimulator(Simulator):
    """Genesis simulator implementation (not yet implemented)."""

    def __init__(self, config: SimulatorConfig):
        raise NotImplementedError(
            "Genesis simulator is not yet implemented. "
            "Please use 'maniskill' simulator for now."
        )

    @property
    def arm_joint_names(self) -> list[str]:
        return CraneX7Config.ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> list[str]:
        return CraneX7Config.GRIPPER_JOINT_NAMES

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError

    def get_observation(self) -> Observation:
        raise NotImplementedError

    def get_qpos(self) -> np.ndarray:
        raise NotImplementedError

    def get_qvel(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass
