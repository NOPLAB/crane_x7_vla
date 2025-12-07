# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""CRANE-X7 Robot implementation for LeRobot."""

from typing import Any

import numpy as np

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.common.robots.robot import Robot

from .config_crane_x7 import JOINT_LIMITS_DEG, CraneX7RobotConfig

# CRANE-X7 motor configuration
# Reference: ros2/src/crane_x7_ros/crane_x7_control/config/manipulator_config.yaml
CRANE_X7_MOTORS = {
    "joint1": {"id": 2, "model": "xm430-w350"},  # Shoulder pan
    "joint2": {"id": 3, "model": "xm540-w270"},  # Shoulder tilt
    "joint3": {"id": 4, "model": "xm430-w350"},  # Upper arm twist
    "joint4": {"id": 5, "model": "xm430-w350"},  # Upper arm rotate
    "joint5": {"id": 6, "model": "xm430-w350"},  # Lower arm
    "joint6": {"id": 7, "model": "xm430-w350"},  # Lower arm rotate
    "joint7": {"id": 8, "model": "xm430-w350"},  # Wrist
    "gripper": {"id": 9, "model": "xm430-w350"},  # Gripper
}


class CraneX7Robot(Robot):
    """CRANE-X7 Robot implementation for LeRobot.

    This class provides the interface between LeRobot and the CRANE-X7 robot arm,
    using Dynamixel motors for control and optional RealSense/USB cameras for vision.
    """

    config_class = CraneX7RobotConfig
    name = "crane_x7"

    def __init__(self, config: CraneX7RobotConfig):
        super().__init__(config)
        self.config = config

        # Select normalization mode
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Initialize motor bus
        motors = {}
        for joint_name, motor_info in CRANE_X7_MOTORS.items():
            # Use RANGE_0_100 for gripper for intuitive control (0=closed, 100=open)
            mode = MotorNormMode.RANGE_0_100 if joint_name == "gripper" else norm_mode
            motors[joint_name] = Motor(motor_info["id"], motor_info["model"], mode)

        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors=motors,
            calibration=self.calibration,
        )

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    # -------------------------------------------------------------------------
    # Feature definitions
    # -------------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor position feature definitions."""
        return {f"{name}.pos": float for name in CRANE_X7_MOTORS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera image feature definitions."""
        features = {}
        for cam_name, cam in self.cameras.items():
            # RGB image
            features[cam_name] = (cam.height, cam.width, 3)
            # Depth image (if available)
            if hasattr(cam.config, "use_depth") and cam.config.use_depth:
                features[f"{cam_name}_depth"] = (cam.height, cam.width)
        return features

    @property
    def observation_features(self) -> dict:
        """Define observation space (motor positions + camera images)."""
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        """Define action space (motor position commands)."""
        return self._motors_ft

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        motors_connected = self.bus.is_connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return motors_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot hardware.

        Args:
            calibrate: If True and not calibrated, run calibration procedure
        """
        # Connect motor bus
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Configure motors
        self.configure()

    def disconnect(self) -> None:
        """Disconnect from robot hardware."""
        # Disable torque before disconnecting (safety)
        if self.config.torque_off_on_disconnect and self.bus.is_connected:
            try:
                self.bus.disable_torque()
            except Exception:
                pass  # Ignore errors during disconnect

        # Disconnect motor bus
        self.bus.disconnect()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Run calibration procedure.

        This procedure:
        1. Records homing offsets at the center position
        2. Records range of motion for each joint
        """
        from lerobot.common.motors.dynamixel import MotorCalibration

        # Disable torque for manual positioning
        self.bus.disable_torque()

        # Set position control mode
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Step 1: Record homing offsets
        input(
            "\n[Calibration] Move the arm to the CENTER of its range of motion.\n"
            "Press ENTER when ready..."
        )
        homing_offsets = self.bus.set_half_turn_homings()

        # Step 2: Record range of motion
        print(
            "\n[Calibration] Move each joint through its FULL range of motion.\n"
            "Recording positions... Press ENTER when done."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        # Save calibration data
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"\n[Calibration] Saved to: {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors for operation."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                # Set position control mode
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set PID gains (tuned for CRANE-X7)
                self.bus.write("P_Coefficient", motor, 800)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 0)

    # -------------------------------------------------------------------------
    # Observation and Action
    # -------------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from robot.

        Returns:
            Dictionary containing motor positions and camera images
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = {}

        # Read motor positions
        positions = self.bus.sync_read("Present_Position")
        for motor, val in positions.items():
            obs_dict[f"{motor}.pos"] = val

        # Read camera images
        for cam_key, cam in self.cameras.items():
            # RGB image
            obs_dict[cam_key] = cam.async_read()

            # Depth image (if available)
            if hasattr(cam, "read_depth") and hasattr(cam.config, "use_depth"):
                if cam.config.use_depth:
                    obs_dict[f"{cam_key}_depth"] = cam.read_depth()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to robot.

        Args:
            action: Dictionary of motor position commands

        Returns:
            Dictionary of actually sent commands (may be clipped for safety)
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Extract goal positions
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        # Apply joint limits if enabled
        if self.config.enforce_joint_limits:
            for motor, target in goal_pos.items():
                if motor in JOINT_LIMITS_DEG:
                    min_val, max_val = JOINT_LIMITS_DEG[motor]
                    goal_pos[motor] = np.clip(target, min_val, max_val)

        # Apply relative movement limit if enabled
        if self.config.max_relative_target is not None:
            current = self.bus.sync_read("Present_Position")
            for motor, target in goal_pos.items():
                delta = abs(target - current[motor])
                if delta > self.config.max_relative_target:
                    # Clip to max relative movement
                    direction = 1 if target > current[motor] else -1
                    goal_pos[motor] = (
                        current[motor] + direction * self.config.max_relative_target
                    )

        # Send goal positions
        self.bus.sync_write("Goal_Position", goal_pos)

        # Return what was actually sent
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def __repr__(self) -> str:
        return f"CraneX7Robot(port={self.config.port!r}, connected={self.is_connected})"
