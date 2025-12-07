# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""CRANE-X7 Teleoperator (Leader arm) implementation for LeRobot."""

from typing import Any

from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.common.teleoperators.teleoperator import Teleoperator

from .config_crane_x7_teleop import CraneX7TeleopConfig

# Same motor configuration as the robot
CRANE_X7_MOTORS = {
    "joint1": {"id": 2, "model": "xm430-w350"},
    "joint2": {"id": 3, "model": "xm540-w270"},
    "joint3": {"id": 4, "model": "xm430-w350"},
    "joint4": {"id": 5, "model": "xm430-w350"},
    "joint5": {"id": 6, "model": "xm430-w350"},
    "joint6": {"id": 7, "model": "xm430-w350"},
    "joint7": {"id": 8, "model": "xm430-w350"},
    "gripper": {"id": 9, "model": "xm430-w350"},
}


class CraneX7Teleop(Teleoperator):
    """CRANE-X7 Teleoperator (Leader arm) implementation.

    This teleoperator reads joint positions from a leader CRANE-X7 arm
    with torque disabled, allowing the user to manually position it.
    The positions are then used as action commands for the follower robot.
    """

    config_class = CraneX7TeleopConfig
    name = "crane_x7_teleop"

    def __init__(self, config: CraneX7TeleopConfig):
        super().__init__(config)
        self.config = config

        # Select normalization mode
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Initialize motor bus
        motors = {}
        for joint_name, motor_info in CRANE_X7_MOTORS.items():
            mode = MotorNormMode.RANGE_0_100 if joint_name == "gripper" else norm_mode
            motors[joint_name] = Motor(motor_info["id"], motor_info["model"], mode)

        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors=motors,
            calibration=self.calibration,
        )

    # -------------------------------------------------------------------------
    # Feature definitions
    # -------------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        """Define action space (motor position commands)."""
        return {f"{name}.pos": float for name in CRANE_X7_MOTORS}

    @property
    def feedback_features(self) -> dict:
        """Define feedback space (empty - no force feedback on CRANE-X7)."""
        return {}

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to leader arm hardware.

        Args:
            calibrate: If True and not calibrated, run calibration procedure
        """
        self.bus.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()

    def disconnect(self) -> None:
        """Disconnect from leader arm hardware."""
        # Ensure torque is off before disconnecting
        if self.bus.is_connected:
            try:
                self.bus.disable_torque()
            except Exception:
                pass

        self.bus.disconnect()

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """Run calibration procedure for leader arm.

        Same procedure as the robot calibration.
        """
        from lerobot.common.motors.dynamixel import MotorCalibration

        # Disable torque for manual positioning
        self.bus.disable_torque()

        # Set position control mode
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Step 1: Record homing offsets
        input(
            "\n[Leader Calibration] Move the leader arm to the CENTER of its range.\n"
            "Press ENTER when ready..."
        )
        homing_offsets = self.bus.set_half_turn_homings()

        # Step 2: Record range of motion
        print(
            "\n[Leader Calibration] Move each joint through its FULL range.\n"
            "Recording... Press ENTER when done."
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
        print(f"\n[Leader Calibration] Saved to: {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure leader arm for teleoperation (torque OFF)."""
        # Leader arm operates with torque disabled for manual positioning
        self.bus.disable_torque()

        with self.bus.torque_disabled():
            for motor in self.bus.motors:
                # Set position control mode (for reading)
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Low PID gains (not used since torque is off, but set for safety)
                self.bus.write("P_Coefficient", motor, 5)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 0)

    # -------------------------------------------------------------------------
    # Action and Feedback
    # -------------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        """Get current position of leader arm as action command.

        Returns:
            Dictionary of motor positions to be sent to follower robot
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        positions = self.bus.sync_read("Present_Position")
        return {f"{motor}.pos": val for motor, val in positions.items()}

    def send_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        """Send feedback to teleoperator (not implemented for CRANE-X7).

        CRANE-X7 does not have force feedback capability.

        Args:
            feedback: Feedback data (ignored)

        Returns:
            The same feedback data (no-op)
        """
        # No force feedback on CRANE-X7
        return feedback

    def __repr__(self) -> str:
        return f"CraneX7Teleop(port={self.config.port!r}, connected={self.is_connected})"
