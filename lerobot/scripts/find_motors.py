#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Find and list Dynamixel motors connected to the system."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Find Dynamixel motors on USB ports")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="USB port to scan",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=3_000_000,
        help="Baudrate (CRANE-X7 uses 3Mbps)",
    )
    parser.add_argument(
        "--id-range",
        type=str,
        default="1-20",
        help="Range of motor IDs to scan (e.g., '1-20')",
    )
    args = parser.parse_args()

    # Parse ID range
    start_id, end_id = map(int, args.id_range.split("-"))

    print("=" * 60)
    print("Dynamixel Motor Scanner")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate:,}")
    print(f"Scanning IDs: {start_id} to {end_id}")
    print("=" * 60)

    try:
        from lerobot.common.motors.dynamixel import DynamixelMotorsBus

        # Create a temporary bus to scan
        from lerobot.common.motors import Motor, MotorNormMode

        # We need at least one motor to create the bus
        # Use a dummy motor configuration
        motors = {f"scan_{i}": Motor(i, "xm430-w350", MotorNormMode.DEGREES) for i in range(start_id, end_id + 1)}

        bus = DynamixelMotorsBus(
            port=args.port,
            motors=motors,
        )

        print("\nConnecting...")
        bus.connect()

        print("\nScanning for motors...")
        found_motors = []

        for motor_id in range(start_id, end_id + 1):
            try:
                # Try to ping the motor
                result = bus._drivers["default"].ping(motor_id)
                if result:
                    # Try to read model number
                    model_num = bus.read("Model_Number", f"scan_{motor_id}")
                    found_motors.append((motor_id, model_num))
                    print(f"  Found: ID={motor_id}, Model={model_num}")
            except Exception:
                pass

        print("\n" + "=" * 60)
        print(f"Found {len(found_motors)} motor(s)")
        print("=" * 60)

        if found_motors:
            print("\nCRANE-X7 Expected Configuration:")
            expected = [
                (2, "joint1 - Shoulder pan (XM430)"),
                (3, "joint2 - Shoulder tilt (XM540)"),
                (4, "joint3 - Upper arm twist (XM430)"),
                (5, "joint4 - Upper arm rotate (XM430)"),
                (6, "joint5 - Lower arm (XM430)"),
                (7, "joint6 - Lower arm rotate (XM430)"),
                (8, "joint7 - Wrist (XM430)"),
                (9, "gripper - Gripper (XM430)"),
            ]
            for motor_id, desc in expected:
                status = "OK" if any(m[0] == motor_id for m in found_motors) else "MISSING"
                print(f"  ID {motor_id}: {desc} [{status}]")

        bus.disconnect()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Check USB connection")
        print("  2. Verify port exists: ls /dev/ttyUSB*")
        print("  3. Check permissions: sudo chmod 666 /dev/ttyUSB0")
        print("  4. Ensure power is on")


if __name__ == "__main__":
    main()
