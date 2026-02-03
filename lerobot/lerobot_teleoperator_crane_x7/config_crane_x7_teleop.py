# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Configuration for CRANE-X7 Teleoperator (Leader arm)."""

from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("crane_x7_teleop")
@dataclass
class CraneX7TeleopConfig(TeleoperatorConfig):
    """Configuration for CRANE-X7 teleoperator (leader arm).

    The leader arm is used for teleoperation - the user manually moves this arm
    and its position is read to control the follower robot.

    Attributes:
        port: USB port for Dynamixel communication (e.g., "/dev/ttyUSB1")
        baudrate: Communication baudrate (CRANE-X7 uses 3Mbps)
        use_degrees: If True, use degrees for joint positions; otherwise normalized values
        gripper_open_pos: Gripper position when open (0-100, used for spring-back trigger)
    """

    # Required: Dynamixel port (typically different from follower robot)
    port: str = "/dev/ttyUSB1"

    # Communication settings
    baudrate: int = 3_000_000

    # Joint position mode
    use_degrees: bool = True

    # Sets the gripper motor to this position with torque enabled.
    # This makes it possible to squeeze the gripper and have it spring back to open.
    gripper_open_pos: float = 50.0
