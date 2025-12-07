#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 nop
# SPDX-License-Identifier: MIT

"""Convert existing TFRecord data to LeRobot dataset format."""

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_tfrecord_example(example_proto, feature_description):
    """Parse a single TFRecord example."""
    import tensorflow as tf

    example = tf.io.parse_single_example(example_proto, feature_description)

    result = {
        "state": example["observation/state"].numpy(),
        "action": example["action"].numpy(),
    }

    # Handle image
    if "observation/image" in example:
        image_bytes = example["observation/image"].numpy()
        image = Image.open(io.BytesIO(image_bytes))
        result["image"] = np.array(image)

    # Handle language instruction
    if "prompt" in example:
        result["prompt"] = example["prompt"].numpy().decode("utf-8")

    return result


def convert_tfrecord_to_lerobot(
    tfrecord_dir: Path,
    output_dir: Path,
    repo_id: str,
    task_description: str,
    fps: int = 30,
):
    """Convert TFRecord episodes to LeRobot dataset format."""
    import tensorflow as tf

    # Disable GPU for TensorFlow (we only need CPU for parsing)
    tf.config.set_visible_devices([], "GPU")

    print("=" * 60)
    print("TFRecord to LeRobot Conversion")
    print("=" * 60)
    print(f"Input: {tfrecord_dir}")
    print(f"Output: {output_dir}")
    print(f"Repo ID: {repo_id}")
    print(f"Task: {task_description}")
    print("=" * 60)

    # Find TFRecord files
    tfrecord_files = sorted(tfrecord_dir.glob("**/*.tfrecord"))
    if not tfrecord_files:
        print("No TFRecord files found!")
        return

    print(f"\nFound {len(tfrecord_files)} TFRecord file(s)")

    # Define feature description for parsing
    feature_description = {
        "observation/state": tf.io.FixedLenFeature([8], tf.float32),
        "action": tf.io.FixedLenFeature([8], tf.float32),
        "observation/image": tf.io.FixedLenFeature([], tf.string),
    }

    # Optional features
    optional_features = {
        "prompt": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "observation/timestamp": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    # Try to import LeRobot dataset tools
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        use_lerobot_api = True
    except ImportError:
        print("Warning: LeRobot dataset API not available, using manual conversion")
        use_lerobot_api = False

    if use_lerobot_api:
        # Use LeRobot API for conversion
        output_dir.mkdir(parents=True, exist_ok=True)

        all_episodes = []

        for tfrecord_file in tqdm(tfrecord_files, desc="Processing files"):
            raw_dataset = tf.data.TFRecordDataset(str(tfrecord_file))

            episode_frames = []
            for raw_record in raw_dataset:
                try:
                    frame = parse_tfrecord_example(
                        raw_record, {**feature_description, **optional_features}
                    )
                    episode_frames.append(frame)
                except Exception as e:
                    print(f"Warning: Failed to parse record: {e}")
                    continue

            if episode_frames:
                all_episodes.append(episode_frames)

        print(f"\nConverted {len(all_episodes)} episodes")

        # Save in a simple format that can be loaded
        output_file = output_dir / "episodes.npz"
        np.savez_compressed(
            output_file,
            episodes=[
                {
                    "states": np.array([f["state"] for f in ep]),
                    "actions": np.array([f["action"] for f in ep]),
                    "images": np.array([f.get("image", np.zeros((480, 640, 3))) for f in ep]),
                }
                for ep in all_episodes
            ],
            task=task_description,
            fps=fps,
        )
        print(f"Saved to: {output_file}")

    else:
        # Manual conversion
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, tfrecord_file in enumerate(tqdm(tfrecord_files, desc="Converting")):
            raw_dataset = tf.data.TFRecordDataset(str(tfrecord_file))

            states = []
            actions = []
            images = []

            for raw_record in raw_dataset:
                try:
                    frame = parse_tfrecord_example(raw_record, feature_description)
                    states.append(frame["state"])
                    actions.append(frame["action"])
                    if "image" in frame:
                        images.append(frame["image"])
                except Exception:
                    continue

            if states:
                episode_dir = output_dir / f"episode_{idx:04d}"
                episode_dir.mkdir(exist_ok=True)

                np.savez_compressed(
                    episode_dir / "data.npz",
                    states=np.array(states),
                    actions=np.array(actions),
                    images=np.array(images) if images else None,
                )

        print(f"\nConversion complete! Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TFRecord data to LeRobot format"
    )
    parser.add_argument(
        "--tfrecord-dir",
        type=Path,
        required=True,
        help="Directory containing TFRecord files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/crane_x7_converted",
        help="Repository ID for the dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="manipulation task",
        help="Task description",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second",
    )
    args = parser.parse_args()

    convert_tfrecord_to_lerobot(
        tfrecord_dir=args.tfrecord_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
