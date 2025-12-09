# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""CRANE-X7 robot configuration shared across simulators."""

import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class CraneX7Config:
    """CRANE-X7 robot configuration constants."""

    ARM_JOINT_NAMES: ClassVar[list[str]] = [
        "crane_x7_shoulder_fixed_part_pan_joint",
        "crane_x7_shoulder_revolute_part_tilt_joint",
        "crane_x7_upper_arm_revolute_part_twist_joint",
        "crane_x7_upper_arm_revolute_part_rotate_joint",
        "crane_x7_lower_arm_fixed_part_joint",
        "crane_x7_lower_arm_revolute_part_joint",
        "crane_x7_wrist_joint",
    ]

    GRIPPER_JOINT_NAMES: ClassVar[list[str]] = [
        "crane_x7_gripper_finger_a_joint",
        "crane_x7_gripper_finger_b_joint",
    ]

    ALL_JOINT_NAMES: ClassVar[list[str]] = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

    NUM_ARM_JOINTS: ClassVar[int] = 7
    NUM_GRIPPER_JOINTS: ClassVar[int] = 2
    NUM_TOTAL_JOINTS: ClassVar[int] = 9

    REST_QPOS: ClassVar[np.ndarray] = np.array(
        [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, -np.pi / 4, np.pi / 2, 0.0, 0.0]
    )

    ARM_STIFFNESS: ClassVar[float] = 1e3
    ARM_DAMPING: ClassVar[float] = 1e2
    ARM_FORCE_LIMIT: ClassVar[float] = 10000

    GRIPPER_STIFFNESS: ClassVar[float] = 1e3
    GRIPPER_DAMPING: ClassVar[float] = 1e2
    GRIPPER_FORCE_LIMIT: ClassVar[float] = 10000


def get_assets_dir() -> str:
    """Get path to the shared assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_mjcf_path() -> str:
    """Get path to the CRANE-X7 MJCF model."""
    return os.path.join(get_assets_dir(), "crane_x7.xml")
