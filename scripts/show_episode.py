#!/usr/bin/env python3
"""
Display contents of a specified episode from TFRecord data.

Usage:
    python scripts/show_episode.py <episode_path>
    python scripts/show_episode.py data/2_yosasou/episode_0000_20251027_053324
    python scripts/show_episode.py data/2_yosasou/episode_0000_20251027_053324 --show-image
    python scripts/show_episode.py data/2_yosasou/episode_0000_20251027_053324 --save-images output/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_episode(tfrecord_path: str) -> list[dict]:
    """Load all steps from a TFRecord file."""
    steps = []
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        step = {}

        # Language instruction
        for key in ["task/language_instruction", "language_instruction", "prompt"]:
            if key in features and features[key].bytes_list.value:
                step["language_instruction"] = features[key].bytes_list.value[
                    0
                ].decode("utf-8")
                break

        # State/Proprio
        for key in ["observation/proprio", "observation/state"]:
            if key in features and features[key].float_list.value:
                step["state"] = np.array(
                    features[key].float_list.value, dtype=np.float32
                )
                break

        # Action
        if "action" in features and features["action"].float_list.value:
            step["action"] = np.array(
                features["action"].float_list.value, dtype=np.float32
            )

        # Image
        for key in ["observation/image_primary", "observation/image"]:
            if key in features and features[key].bytes_list.value:
                step["image_bytes"] = features[key].bytes_list.value[0]
                break

        # Timestamp
        for key in ["observation/timestamp", "timestamp"]:
            if key in features and features[key].float_list.value:
                step["timestamp"] = features[key].float_list.value[0]
                break

        steps.append(step)

    return steps


def print_episode_info(steps: list[dict], episode_path: str) -> None:
    """Print episode information."""
    print(f"\n{'=' * 60}")
    print(f"Episode: {episode_path}")
    print(f"{'=' * 60}")

    if not steps:
        print("No steps found in this episode.")
        return

    # Language instruction
    lang_instr = steps[0].get("language_instruction", "N/A")
    print(f"\nLanguage Instruction: {lang_instr}")
    print(f"Number of Steps: {len(steps)}")

    # State statistics
    if "state" in steps[0]:
        states = np.array([s["state"] for s in steps if "state" in s])
        print(f"\nState Shape: {states.shape}")
        print(f"State Mean: {np.mean(states, axis=0).round(4)}")
        print(f"State Std:  {np.std(states, axis=0).round(4)}")
        print(f"State Min:  {np.min(states, axis=0).round(4)}")
        print(f"State Max:  {np.max(states, axis=0).round(4)}")

    # Action statistics
    if "action" in steps[0]:
        actions = np.array([s["action"] for s in steps if "action" in s])
        print(f"\nAction Shape: {actions.shape}")
        print(f"Action Mean: {np.mean(actions, axis=0).round(4)}")
        print(f"Action Std:  {np.std(actions, axis=0).round(4)}")
        print(f"Action Min:  {np.min(actions, axis=0).round(4)}")
        print(f"Action Max:  {np.max(actions, axis=0).round(4)}")

    # Image info
    if "image_bytes" in steps[0]:
        img = tf.io.decode_jpeg(steps[0]["image_bytes"]).numpy()
        print(f"\nImage Shape: {img.shape}")

    # Timestamp info
    if "timestamp" in steps[0]:
        timestamps = [s["timestamp"] for s in steps if "timestamp" in s]
        if len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            print(f"\nDuration: {duration:.2f} seconds")


def print_step_details(steps: list[dict], step_indices: list[int] | None = None) -> None:
    """Print details for specific steps."""
    if step_indices is None:
        step_indices = [0, len(steps) // 2, len(steps) - 1]

    print(f"\n{'=' * 60}")
    print("Step Details")
    print(f"{'=' * 60}")

    for idx in step_indices:
        if 0 <= idx < len(steps):
            step = steps[idx]
            print(f"\n--- Step {idx} ---")
            if "state" in step:
                print(f"State:  {step['state'].round(4)}")
            if "action" in step:
                print(f"Action: {step['action'].round(4)}")
            if "timestamp" in step:
                print(f"Timestamp: {step['timestamp']:.4f}")


def show_image(steps: list[dict], step_idx: int = 0) -> None:
    """Display image from a step using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for --show-image. Install with: pip install matplotlib")
        return

    if step_idx >= len(steps) or "image_bytes" not in steps[step_idx]:
        print(f"No image available for step {step_idx}")
        return

    img = tf.io.decode_jpeg(steps[step_idx]["image_bytes"]).numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Step {step_idx}")
    plt.axis("off")
    plt.show()


def save_images(steps: list[dict], output_dir: str) -> None:
    """Save all images from episode to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx, step in enumerate(steps):
        if "image_bytes" in step:
            img_path = output_path / f"step_{idx:04d}.jpg"
            with open(img_path, "wb") as f:
                f.write(step["image_bytes"])
            saved_count += 1

    print(f"\nSaved {saved_count} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Display contents of a specified episode from TFRecord data."
    )
    parser.add_argument(
        "episode_path",
        help="Path to episode directory or TFRecord file",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Display the first image using matplotlib",
    )
    parser.add_argument(
        "--save-images",
        metavar="DIR",
        help="Save all images to the specified directory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Specific step indices to show details for",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Show details for all steps",
    )

    args = parser.parse_args()

    # Resolve TFRecord path
    episode_path = Path(args.episode_path)
    if episode_path.is_dir():
        tfrecord_path = episode_path / "episode_data.tfrecord"
    else:
        tfrecord_path = episode_path

    if not tfrecord_path.exists():
        print(f"Error: TFRecord not found at {tfrecord_path}", file=sys.stderr)
        sys.exit(1)

    # Load and display episode
    steps = load_episode(str(tfrecord_path))
    print_episode_info(steps, str(episode_path))

    # Step details
    if args.all_steps:
        print_step_details(steps, list(range(len(steps))))
    elif args.steps:
        print_step_details(steps, args.steps)
    else:
        print_step_details(steps)

    # Image operations
    if args.show_image:
        show_image(steps)

    if args.save_images:
        save_images(steps, args.save_images)


if __name__ == "__main__":
    main()
