#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Calibration script for CRANE-X7 robot and teleoperator."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate CRANE-X7 robot or teleoperator"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="USB port for Dynamixel communication",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["robot", "teleop"],
        default="robot",
        help="Device type to calibrate",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("/workspace/lerobot/calibration"),
        help="Directory to save calibration data",
    )
    args = parser.parse_args()

    # Ensure calibration directory exists
    args.calibration_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "robot":
        print("=" * 60)
        print("CRANE-X7 Robot Calibration")
        print("=" * 60)

        from lerobot_robot_crane_x7 import CraneX7Robot, CraneX7RobotConfig

        config = CraneX7RobotConfig(
            port=args.port,
            use_degrees=True,
            cameras={},  # No cameras needed for calibration
        )
        device = CraneX7Robot(config)

    else:  # teleop
        print("=" * 60)
        print("CRANE-X7 Teleoperator (Leader Arm) Calibration")
        print("=" * 60)

        from lerobot_teleoperator_crane_x7 import CraneX7Teleop, CraneX7TeleopConfig

        config = CraneX7TeleopConfig(
            port=args.port,
            use_degrees=True,
        )
        device = CraneX7Teleop(config)

    try:
        print(f"\nConnecting to {args.type} on {args.port}...")
        device.connect(calibrate=True)
        print("\nCalibration complete!")

    except KeyboardInterrupt:
        print("\n\nCalibration cancelled by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError during calibration: {e}")
        sys.exit(1)

    finally:
        if device.is_connected:
            device.disconnect()
            print("Device disconnected.")


if __name__ == "__main__":
    main()
