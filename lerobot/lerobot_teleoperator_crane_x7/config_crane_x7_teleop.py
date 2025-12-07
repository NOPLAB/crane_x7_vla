# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Configuration for CRANE-X7 Teleoperator (Leader arm)."""

from dataclasses import dataclass

from lerobot.common.teleoperators.teleoperator import TeleoperatorConfig


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
    """

    # Required: Dynamixel port (typically different from follower robot)
    port: str = "/dev/ttyUSB1"

    # Communication settings
    baudrate: int = 3_000_000

    # Joint position mode
    use_degrees: bool = True
