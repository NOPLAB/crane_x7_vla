# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Robot-specific configuration for VLA training."""

from dataclasses import dataclass, field
from typing import Literal


# CRANE-X7 default joint names
CRANE_X7_JOINT_NAMES = [
    "crane_x7_shoulder_fixed_part_pan_joint",
    "crane_x7_shoulder_revolute_part_tilt_joint",
    "crane_x7_upper_arm_revolute_part_twist_joint",
    "crane_x7_upper_arm_revolute_part_rotate_joint",
    "crane_x7_lower_arm_fixed_part_joint",
    "crane_x7_lower_arm_revolute_part_joint",
    "crane_x7_wrist_joint",
    "crane_x7_gripper_finger_a_joint",
]


@dataclass
class RobotConfig:
    """
    Robot-specific configuration.

    This configuration encapsulates robot-specific parameters that were
    previously hardcoded throughout the codebase.
    """

    robot_type: Literal["crane_x7", "custom"] = "crane_x7"
    """Type of robot (crane_x7 or custom)"""

    native_action_dim: int = 8
    """Native action dimension of the robot (number of joints)"""

    native_state_dim: int = 8
    """Native state dimension of the robot (number of joint states)"""

    joint_names: list[str] = field(default_factory=lambda: CRANE_X7_JOINT_NAMES.copy())
    """Names of robot joints"""

    gripper_joint_index: int = 7
    """Index of gripper joint in action/state vectors"""

    # Action limits (for normalization)
    action_low: list[float] | None = None
    """Lower bounds for each action dimension (optional)"""

    action_high: list[float] | None = None
    """Upper bounds for each action dimension (optional)"""

    def __post_init__(self):
        """Validate configuration."""
        if len(self.joint_names) != self.native_action_dim:
            raise ValueError(
                f"Number of joint names ({len(self.joint_names)}) must match "
                f"native_action_dim ({self.native_action_dim})"
            )

        if self.gripper_joint_index >= self.native_action_dim:
            raise ValueError(
                f"gripper_joint_index ({self.gripper_joint_index}) must be less than "
                f"native_action_dim ({self.native_action_dim})"
            )

    @classmethod
    def crane_x7(cls) -> "RobotConfig":
        """Create default CRANE-X7 configuration."""
        return cls(
            robot_type="crane_x7",
            native_action_dim=8,
            native_state_dim=8,
            joint_names=CRANE_X7_JOINT_NAMES.copy(),
            gripper_joint_index=7,
        )

    @classmethod
    def custom(
        cls,
        native_action_dim: int,
        native_state_dim: int,
        joint_names: list[str],
        gripper_joint_index: int = -1,
    ) -> "RobotConfig":
        """
        Create custom robot configuration.

        Args:
            native_action_dim: Number of action dimensions
            native_state_dim: Number of state dimensions
            joint_names: Names of robot joints
            gripper_joint_index: Index of gripper joint (-1 if no gripper)

        Returns:
            Custom RobotConfig instance
        """
        return cls(
            robot_type="custom",
            native_action_dim=native_action_dim,
            native_state_dim=native_state_dim,
            joint_names=joint_names,
            gripper_joint_index=gripper_joint_index,
        )
