#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Inference script for running trained policies on CRANE-X7."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run trained policy on CRANE-X7 robot"
    )
    parser.add_argument(
        "--robot.type",
        dest="robot_type",
        type=str,
        default="crane_x7",
        help="Robot type",
    )
    parser.add_argument(
        "--robot.port",
        dest="robot_port",
        type=str,
        default="/dev/ttyUSB0",
        help="USB port for robot",
    )
    parser.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        required=True,
        help="Path to trained policy checkpoint",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the object",
        help="Task instruction",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of control steps",
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        default=True,
        help="Use camera for observation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without sending to robot",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CRANE-X7 Policy Inference")
    print("=" * 60)
    print(f"Policy: {args.policy_path}")
    print(f"Task: {args.task}")
    print(f"Control frequency: {args.control_freq} Hz")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load policy
    print("\nLoading policy...")
    try:
        from lerobot.common.policies.factory import make_policy

        policy = make_policy(
            hydra_cfg_path=Path(args.policy_path) / "config.yaml",
            pretrained_policy_name_or_path=args.policy_path,
        )
        policy.to(device)
        policy.eval()
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Initialize robot
    print("\nInitializing robot...")
    from lerobot_robot_crane_x7 import CraneX7Robot, CraneX7RobotConfig

    if args.use_camera:
        from lerobot.common.cameras.realsense.configuration_realsense import (
            RealSenseCameraConfig,
        )

        cameras = {
            "cam_wrist": RealSenseCameraConfig(
                serial_number_or_name="",
                fps=30,
                width=640,
                height=480,
                use_depth=False,
            )
        }
    else:
        cameras = {}

    config = CraneX7RobotConfig(
        port=args.robot_port,
        use_degrees=True,
        max_relative_target=5.0,
        cameras=cameras,
    )

    robot = CraneX7Robot(config)

    try:
        robot.connect(calibrate=False)
        print("Robot connected!")

        if args.dry_run:
            print("\n[DRY RUN] Actions will be printed but not sent to robot.")

        print(f"\nStarting inference: {args.task}")
        print("Press Ctrl+C to stop\n")

        control_period = 1.0 / args.control_freq
        step = 0

        # Reset policy state
        policy.reset()

        while step < args.max_steps:
            start_time = time.time()

            # Get observation
            obs = robot.get_observation()

            # Prepare observation for policy
            obs_dict = {}

            # Motor positions
            state = np.array([
                obs["joint1.pos"],
                obs["joint2.pos"],
                obs["joint3.pos"],
                obs["joint4.pos"],
                obs["joint5.pos"],
                obs["joint6.pos"],
                obs["joint7.pos"],
                obs["gripper.pos"],
            ], dtype=np.float32)
            obs_dict["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

            # Camera image (if available)
            if "cam_wrist" in obs:
                image = obs["cam_wrist"]
                # Normalize to [0, 1] and convert to CHW format
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
                obs_dict["observation.image"] = (
                    torch.from_numpy(image).unsqueeze(0).to(device)
                )

            # Run inference
            with torch.no_grad():
                action = policy.select_action(obs_dict)

            # Extract action values
            action_np = action.squeeze(0).cpu().numpy()

            # Create action dict
            action_dict = {
                "joint1.pos": float(action_np[0]),
                "joint2.pos": float(action_np[1]),
                "joint3.pos": float(action_np[2]),
                "joint4.pos": float(action_np[3]),
                "joint5.pos": float(action_np[4]),
                "joint6.pos": float(action_np[5]),
                "joint7.pos": float(action_np[6]),
                "gripper.pos": float(action_np[7]),
            }

            # Send action
            if args.dry_run:
                if step % 10 == 0:
                    print(f"Step {step}: {action_dict}")
            else:
                robot.send_action(action_dict)

            step += 1

            # Maintain control frequency
            elapsed = time.time() - start_time
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

            # Print progress
            if step % 100 == 0:
                print(f"Step {step}/{args.max_steps}")

        print(f"\nCompleted {step} steps.")

    except KeyboardInterrupt:
        print("\n\nInference stopped by user.")

    except Exception as e:
        print(f"\nError during inference: {e}")
        raise

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
