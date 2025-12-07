# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Configuration for CRANE-X7 Robot."""

from dataclasses import dataclass, field
from typing import Optional

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.configs import ColorMode
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.config import RobotConfig

# Try to import RealSense config, fallback if not available
try:
    from lerobot.common.cameras.realsense.configuration_realsense import (
        RealSenseCameraConfig,
    )

    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


# CRANE-X7 joint limits (in degrees)
# Reference: ros2/src/crane_x7_teleop/config/teleop_config.yaml
JOINT_LIMITS_DEG = {
    "joint1": (-157.0, 157.0),
    "joint2": (-90.0, 90.0),
    "joint3": (-157.0, 157.0),
    "joint4": (-159.0, 0.001),
    "joint5": (-157.0, 157.0),
    "joint6": (-90.0, 90.0),
    "joint7": (-167.0, 167.0),
    "gripper": (-5.0, 90.0),
}


def default_cameras() -> dict[str, CameraConfig]:
    """Default camera configuration with RealSense D435."""
    if HAS_REALSENSE:
        return {
            "cam_wrist": RealSenseCameraConfig(
                serial_number_or_name="",  # Auto-detect
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
        }
    else:
        # Fallback to OpenCV camera
        return {
            "cam_wrist": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
            )
        }


@RobotConfig.register_subclass("crane_x7")
@dataclass
class CraneX7RobotConfig(RobotConfig):
    """Configuration for CRANE-X7 robot.

    Attributes:
        port: USB port for Dynamixel communication (e.g., "/dev/ttyUSB0")
        baudrate: Communication baudrate (CRANE-X7 uses 3Mbps)
        use_degrees: If True, use degrees for joint positions; otherwise normalized values
        enforce_joint_limits: If True, clip actions to joint limits
        max_relative_target: Maximum relative movement per step (degrees), for safety
        torque_off_on_disconnect: If True, disable torque when disconnecting
        cameras: Camera configurations
    """

    # Required: Dynamixel port
    port: str = "/dev/ttyUSB0"

    # Communication settings
    baudrate: int = 3_000_000

    # Joint position mode
    use_degrees: bool = True

    # Safety settings
    enforce_joint_limits: bool = True
    max_relative_target: Optional[float] = 5.0  # degrees per step

    # Disconnect behavior
    torque_off_on_disconnect: bool = True

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(default_factory=default_cameras)
