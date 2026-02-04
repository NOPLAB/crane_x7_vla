# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""CRANE-X7 Teleoperator (Leader arm) implementation for LeRobot."""

import logging

import draccus

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_crane_x7_teleop import CraneX7TeleopConfig

logger = logging.getLogger(__name__)

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
    def action_features(self) -> dict[str, type]:
        """Define action space (motor position commands)."""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Define feedback space (empty - no force feedback on CRANE-X7)."""
        return {}

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to leader arm hardware.

        Args:
            calibrate: If True and not calibrated, run calibration procedure
        """
        # Connect motor bus with custom baudrate
        # LeRobot defaults to 1Mbps, but CRANE-X7 uses 3Mbps
        # Set baudrate BEFORE openPort() since it's used during port initialization
        self.bus.port_handler.baudrate = self.config.baudrate
        self.bus.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from leader arm hardware."""
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated.

        CRANE-X7 uses hardware-configured calibration values, so we only check
        if a calibration file exists and the bus has calibration data.
        """
        # Don't compare with motor registers - just check if we have calibration data
        return bool(self.calibration) and bool(self.bus.calibration)

    def calibrate(self) -> None:
        """Run calibration procedure for leader arm.

        CRANE-X7 preserves hardware-configured motor settings (Homing_Offset,
        Min/Max Position Limit). This method reads the current motor settings
        and saves them as calibration data without modifying motor registers.

        If follower robot calibration exists, uses its range values for consistency.
        """
        self.bus.disable_torque()

        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                # Just cache the calibration data without writing to motors
                self.bus.calibration = self.calibration
                return

        logger.info(f"\nRunning calibration of {self}")
        logger.info("Reading current motor settings (hardware values will be preserved)")

        # Read current motor settings from hardware
        hw_calibration = self.bus.read_calibration()

        # Try to load follower robot's calibration for range values
        robot_calibration = self._load_robot_calibration()

        if robot_calibration:
            # Use follower's range-of-motion values for consistency
            logger.info("Using range-of-motion values from follower robot calibration.")
            self.calibration = {}
            for motor, m in self.bus.motors.items():
                self.calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=hw_calibration[motor].drive_mode,
                    homing_offset=hw_calibration[motor].homing_offset,
                    range_min=robot_calibration[motor].range_min,
                    range_max=robot_calibration[motor].range_max,
                )
        else:
            # Use hardware settings directly
            logger.info("Using hardware calibration settings.")
            self.calibration = hw_calibration

        # Cache calibration in bus (without writing to motors)
        self.bus.calibration = self.calibration

        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def _load_robot_calibration(self) -> dict[str, MotorCalibration] | None:
        """Load calibration data from the follower robot (CraneX7Robot).

        Returns:
            Calibration dictionary if found, None otherwise.
        """
        # Compute follower robot's calibration file path
        # Robot uses: HF_LEROBOT_CALIBRATION / "robots" / "crane_x7" / f"{id}.json"
        robot_calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "crane_x7"
        robot_calibration_fpath = robot_calibration_dir / f"{self.id}.json"

        if not robot_calibration_fpath.is_file():
            logger.info(f"Follower robot calibration not found at: {robot_calibration_fpath}")
            return None

        logger.info(f"Loading follower robot calibration from: {robot_calibration_fpath}")
        with open(robot_calibration_fpath) as f, draccus.config_type("json"):
            return draccus.load(dict[str, MotorCalibration], f)

    def configure(self) -> None:
        """Configure leader arm for teleoperation (torque OFF)."""
        # Leader arm operates with torque disabled for manual positioning
        self.bus.disable_torque()
        self.bus.configure_motors()

        for motor in self.bus.motors:
            # Set position control mode (for reading)
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # -------------------------------------------------------------------------
    # Action and Feedback
    # -------------------------------------------------------------------------

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        """Get current position of leader arm as action command.

        Returns:
            Dictionary of motor positions to be sent to follower robot
        """
        action = self.bus.sync_read("Present_Position")
        return {f"{motor}.pos": val for motor, val in action.items()}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send feedback to teleoperator (not implemented for CRANE-X7).

        CRANE-X7 does not have force feedback capability.
        """
        # No force feedback on CRANE-X7
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"CraneX7Teleop(port={self.config.port!r}, connected={self.is_connected})"
