#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Convert CRANE-X7 TFRecord data to LeRobot format for OpenPI training.

Usage:
    python -m crane_x7_vla.scripts.convert_to_lerobot \
        --data_dir /path/to/tfrecord_logs \
        --output_repo crane_x7_vla \
        --fps 10

This script converts TFRecord data collected by the crane_x7_log package
to LeRobot format, which is required for training OpenPI models.

The output dataset will be saved to $HF_LEROBOT_HOME/crane_x7_vla.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

# Try to import LeRobot (may not be available in all environments)
try:
    from lerobot.common.datasets.lerobot_dataset import (
        HF_LEROBOT_HOME,
        LeRobotDataset,
    )
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Install with: pip install lerobot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CRANE-X7 TFRecord feature description
FEATURE_DESCRIPTION = {
    "observation/state": tf.io.FixedLenFeature([8], tf.float32),
    "observation/image": tf.io.FixedLenFeature([], tf.string),
    "action": tf.io.FixedLenFeature([8], tf.float32),
    "prompt": tf.io.FixedLenFeature([], tf.string),
}

# Alternative feature names for compatibility
ALTERNATIVE_FEATURES = {
    "observation/state": tf.io.FixedLenFeature([8], tf.float32),
    "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
    "observation/image": tf.io.FixedLenFeature([], tf.string),
    "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
    "action": tf.io.FixedLenFeature([8], tf.float32),
    "prompt": tf.io.FixedLenFeature([], tf.string),
    "task/language_instruction": tf.io.FixedLenFeature([], tf.string),
}


def find_tfrecord_files(data_dir: Path) -> list[Path]:
    """Find all TFRecord files in the data directory."""
    patterns = ["**/*.tfrecord", "**/*.tfrecord.gz", "**/*.tfrecord-*"]
    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    return sorted(set(files))


def parse_tfrecord_example(example_proto) -> dict:
    """Parse a single TFRecord example with flexible key support."""
    # Try standard features first
    try:
        parsed = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    except tf.errors.InvalidArgumentError:
        # Try with alternative features
        try:
            parsed = tf.io.parse_single_example(example_proto, ALTERNATIVE_FEATURES)
        except tf.errors.InvalidArgumentError as e:
            logger.warning(f"Failed to parse example: {e}")
            return None

    # Normalize keys
    result = {}

    # State (proprioception)
    if "observation/state" in parsed:
        result["state"] = parsed["observation/state"]
    elif "observation/proprio" in parsed:
        result["state"] = parsed["observation/proprio"]

    # Image
    if "observation/image" in parsed:
        result["image_bytes"] = parsed["observation/image"]
    elif "observation/image_primary" in parsed:
        result["image_bytes"] = parsed["observation/image_primary"]

    # Action
    if "action" in parsed:
        result["action"] = parsed["action"]

    # Prompt/Task
    if "prompt" in parsed:
        result["prompt"] = parsed["prompt"]
    elif "task/language_instruction" in parsed:
        result["prompt"] = parsed["task/language_instruction"]

    return result


def decode_image(image_bytes: tf.Tensor) -> np.ndarray:
    """Decode JPEG image bytes to numpy array."""
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    return image.numpy()


def iterate_tfrecord_episodes(
    tfrecord_files: list[Path],
) -> Iterator[tuple[list[dict], str]]:
    """
    Iterate over episodes in TFRecord files.

    TFRecord files from crane_x7_log are organized by episode.
    Each file or group of files contains one episode.

    Yields:
        Tuple of (list of steps, task name)
    """
    # Group files by episode (assuming directory structure)
    episode_dirs = set()
    for f in tfrecord_files:
        episode_dirs.add(f.parent)

    for episode_dir in sorted(episode_dirs):
        episode_files = [f for f in tfrecord_files if f.parent == episode_dir]

        if not episode_files:
            continue

        # Read all steps in the episode
        steps = []
        task_name = "manipulate objects"  # Default task

        dataset = tf.data.TFRecordDataset([str(f) for f in episode_files])

        for example_proto in dataset:
            parsed = parse_tfrecord_example(example_proto)
            if parsed is None:
                continue

            # Decode image
            if "image_bytes" in parsed:
                image = decode_image(parsed["image_bytes"])
            else:
                image = np.zeros((224, 224, 3), dtype=np.uint8)

            # Get task name from first step
            if "prompt" in parsed:
                prompt = parsed["prompt"].numpy()
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")
                if prompt:
                    task_name = prompt

            steps.append({
                "state": parsed.get("state", tf.zeros([8], dtype=tf.float32)).numpy(),
                "action": parsed.get("action", tf.zeros([8], dtype=tf.float32)).numpy(),
                "image": image,
            })

        if steps:
            yield steps, task_name


def convert_to_lerobot(
    data_dir: Path,
    output_repo: str = "crane_x7_vla",
    fps: int = 10,
    overwrite: bool = False,
    push_to_hub: bool = False,
):
    """
    Convert CRANE-X7 TFRecord data to LeRobot format.

    Args:
        data_dir: Directory containing TFRecord files
        output_repo: Name for the LeRobot dataset
        fps: Frames per second of the data
        overwrite: Whether to overwrite existing dataset
        push_to_hub: Whether to push to Hugging Face Hub
    """
    if not LEROBOT_AVAILABLE:
        raise ImportError(
            "LeRobot is required for this script. Install with: pip install lerobot"
        )

    # Find TFRecord files
    tfrecord_files = find_tfrecord_files(data_dir)
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {data_dir}")

    logger.info(f"Found {len(tfrecord_files)} TFRecord files")

    # Clean up existing dataset if overwriting
    output_path = HF_LEROBOT_HOME / output_repo
    if output_path.exists():
        if overwrite:
            logger.info(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
        else:
            raise ValueError(
                f"Dataset already exists at {output_path}. Use --overwrite to replace."
            )

    # Create LeRobot dataset
    # Define features according to OpenPI expectations
    dataset = LeRobotDataset.create(
        repo_id=output_repo,
        robot_type="crane_x7",
        fps=fps,
        features={
            # Primary camera image
            "observation.images.primary": {
                "dtype": "image",
                "shape": (480, 640, 3),  # Original CRANE-X7 resolution
                "names": ["height", "width", "channel"],
            },
            # Robot state (8 DOF: 7 joints + gripper)
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            # Action (8 DOF: 7 joints + gripper)
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # Convert episodes
    episode_count = 0
    step_count = 0

    for steps, task_name in iterate_tfrecord_episodes(tfrecord_files):
        if not steps:
            continue

        # Add each step to the dataset
        for step in steps:
            dataset.add_frame({
                "observation.images.primary": step["image"],
                "observation.state": step["state"],
                "action": step["action"],
                "task": task_name,
            })
            step_count += 1

        # Save episode
        dataset.save_episode()
        episode_count += 1

        if episode_count % 10 == 0:
            logger.info(f"Converted {episode_count} episodes, {step_count} steps")

    logger.info(f"Conversion complete: {episode_count} episodes, {step_count} steps")

    # Push to hub if requested
    if push_to_hub:
        logger.info("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["crane_x7", "openpi", "manipulation"],
            private=False,
            push_videos=True,
            license="mit",
        )
        logger.info("Push complete!")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert CRANE-X7 TFRecord data to LeRobot format"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing TFRecord files",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        default="crane_x7_vla",
        help="Name for the LeRobot dataset (default: crane_x7_vla)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second of the data (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to Hugging Face Hub after conversion",
    )

    args = parser.parse_args()

    convert_to_lerobot(
        data_dir=args.data_dir,
        output_repo=args.output_repo,
        fps=args.fps,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
