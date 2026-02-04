#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Calibration script for CRANE-X7 robot and teleoperator.

For teleoperation, both leader and follower must be calibrated at the same
physical position to ensure position matching. Use --type=both to calibrate
them sequentially while maintaining the same pose.
"""

import argparse
import sys


def calibrate_robot(port: str):
    """Calibrate the follower robot."""
    from lerobot_robot_crane_x7 import CraneX7Robot, CraneX7RobotConfig

    config = CraneX7RobotConfig(
        port=port,
        use_degrees=True,
        cameras={},  # No cameras needed for calibration
    )
    robot = CraneX7Robot(config)

    print(f"\nConnecting to robot on {port}...")
    robot.bus.port_handler.baudrate = config.baudrate
    robot.bus.connect()

    # Run calibration
    robot.calibrate()

    robot.bus.disconnect()
    print("Robot calibration complete.")
    return robot


def calibrate_teleop(port: str):
    """Calibrate the leader teleoperator."""
    from lerobot_teleoperator_crane_x7 import CraneX7Teleop, CraneX7TeleopConfig

    config = CraneX7TeleopConfig(
        port=port,
        use_degrees=True,
    )
    teleop = CraneX7Teleop(config)

    print(f"\nConnecting to teleoperator on {port}...")
    teleop.bus.port_handler.baudrate = config.baudrate
    teleop.bus.connect()

    # Run calibration
    teleop.calibrate()

    teleop.bus.disconnect()
    print("Teleoperator calibration complete.")
    return teleop


def calibrate_both(robot_port: str, teleop_port: str):
    """Calibrate both robot and teleoperator at the same position.

    Only the follower (robot) performs the range-of-motion calibration.
    The leader (teleoperator) shares the same calibration data from the follower,
    using its own homing offset but the follower's range of motion values.
    """
    from lerobot.motors import MotorCalibration
    from lerobot.motors.dynamixel import OperatingMode

    from lerobot_robot_crane_x7 import CraneX7Robot, CraneX7RobotConfig
    from lerobot_teleoperator_crane_x7 import CraneX7Teleop, CraneX7TeleopConfig

    # Create configs
    robot_config = CraneX7RobotConfig(
        port=robot_port,
        use_degrees=True,
        cameras={},
    )
    teleop_config = CraneX7TeleopConfig(
        port=teleop_port,
        use_degrees=True,
    )

    # Create devices
    robot = CraneX7Robot(robot_config)
    teleop = CraneX7Teleop(teleop_config)

    # Connect both (without calibration)
    print(f"\nConnecting to robot on {robot_port}...")
    robot.bus.port_handler.baudrate = robot_config.baudrate
    robot.bus.connect()

    print(f"Connecting to teleoperator on {teleop_port}...")
    teleop.bus.port_handler.baudrate = teleop_config.baudrate
    teleop.bus.connect()

    # Disable torque on both for manual positioning
    robot.bus.disable_torque()
    teleop.bus.disable_torque()

    # Set position control mode on both
    for motor in robot.bus.motors:
        robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
    for motor in teleop.bus.motors:
        teleop.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    print("\n" + "=" * 60)
    print("CRANE-X7 Dual Calibration")
    print("=" * 60)
    print("\nBoth arms will be calibrated at the SAME position.")
    print("Only the FOLLOWER performs range-of-motion calibration.")
    print("The LEADER will share the same calibration data.")

    # Step 1: Move both to center position
    input(
        "\n[Step 1] Move BOTH arms to the SAME center position.\n"
        "         Align them visually or use a reference pose.\n"
        "         Press ENTER when ready..."
    )

    # Record homing offsets for both
    print("\nRecording homing offsets for both arms...")
    robot_homing = robot.bus.set_half_turn_homings()
    teleop_homing = teleop.bus.set_half_turn_homings()

    # Step 2: Record range of motion for robot (follower) only
    print(
        "\n[Step 2] Move the ROBOT (follower) through its full range of motion.\n"
        "         The leader will use the same range values.\n"
        "         Recording positions. Press ENTER to stop..."
    )
    robot_mins, robot_maxes = robot.bus.record_ranges_of_motion()

    # Save robot calibration
    robot.calibration = {}
    for motor, m in robot.bus.motors.items():
        robot.calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=robot_homing[motor],
            range_min=robot_mins[motor],
            range_max=robot_maxes[motor],
        )
    robot.bus.write_calibration(robot.calibration)
    robot._save_calibration()
    print(f"\nRobot calibration saved to: {robot.calibration_fpath}")

    # Save teleop calibration (using follower's range values)
    teleop.calibration = {}
    for motor, m in teleop.bus.motors.items():
        teleop.calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=teleop_homing[motor],
            range_min=robot_mins[motor],  # Use follower's range
            range_max=robot_maxes[motor],  # Use follower's range
        )
    teleop.bus.write_calibration(teleop.calibration)
    teleop._save_calibration()
    print(f"Teleop calibration saved to: {teleop.calibration_fpath}")

    # Disconnect
    robot.bus.disconnect()
    teleop.bus.disconnect()

    print("\n" + "=" * 60)
    print("Dual calibration complete!")
    print("Both arms share the same range-of-motion calibration.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate CRANE-X7 robot and/or teleoperator"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["robot", "teleop", "both"],
        default="both",
        help="Device type to calibrate (default: both)",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyUSB0",
        help="USB port for robot (follower)",
    )
    parser.add_argument(
        "--teleop-port",
        type=str,
        default="/dev/ttyUSB1",
        help="USB port for teleoperator (leader)",
    )
    # Legacy support
    parser.add_argument(
        "--port",
        type=str,
        help="USB port (legacy, use --robot-port or --teleop-port instead)",
    )
    args = parser.parse_args()

    try:
        if args.type == "robot":
            port = args.port or args.robot_port
            print("=" * 60)
            print("CRANE-X7 Robot (Follower) Calibration")
            print("=" * 60)
            calibrate_robot(port)

        elif args.type == "teleop":
            port = args.port or args.teleop_port
            print("=" * 60)
            print("CRANE-X7 Teleoperator (Leader) Calibration")
            print("=" * 60)
            calibrate_teleop(port)

        else:  # both
            print("=" * 60)
            print("CRANE-X7 Dual Calibration (Robot + Teleoperator)")
            print("=" * 60)
            calibrate_both(args.robot_port, args.teleop_port)

    except KeyboardInterrupt:
        print("\n\nCalibration cancelled by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
