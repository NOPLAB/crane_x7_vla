#!/usr/bin/env python3
"""
Update language instruction in TFRecord episode files.

Usage:
    python scripts/update_language_instruction.py <data_dir> --instruction "new instruction"
    python scripts/update_language_instruction.py data/2_yosasou --instruction "Move the yellow tape to the white plate"
"""

import argparse
import os
import shutil
from pathlib import Path

import tensorflow as tf


def update_episode_instruction(tfrecord_path: str, new_instruction: str, output_path: str) -> int:
    """Update language instruction in a TFRecord file."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    writer = tf.io.TFRecordWriter(output_path)
    step_count = 0

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Create new example with updated instruction
        new_features = {}

        for key, feature in example.features.feature.items():
            # Update language instruction keys
            if key in ["task/language_instruction", "language_instruction", "prompt"]:
                new_features[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[new_instruction.encode("utf-8")])
                )
            else:
                # Copy existing feature
                if feature.HasField("bytes_list"):
                    new_features[key] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=list(feature.bytes_list.value))
                    )
                elif feature.HasField("float_list"):
                    new_features[key] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=list(feature.float_list.value))
                    )
                elif feature.HasField("int64_list"):
                    new_features[key] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=list(feature.int64_list.value))
                    )

        new_example = tf.train.Example(features=tf.train.Features(feature=new_features))
        writer.write(new_example.SerializeToString())
        step_count += 1

    writer.close()
    return step_count


def main():
    parser = argparse.ArgumentParser(
        description="Update language instruction in TFRecord episode files."
    )
    parser.add_argument(
        "data_dir",
        help="Path to data directory containing episode_* subdirectories",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="New language instruction to set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Directory not found: {data_path}")
        return 1

    # Find all episode directories
    episode_dirs = sorted(data_path.glob("episode_*"))
    if not episode_dirs:
        print(f"Error: No episode directories found in {data_path}")
        return 1

    print(f"Found {len(episode_dirs)} episodes in {data_path}")
    print(f"New instruction: {args.instruction}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made.")
        return 0

    print()
    updated_count = 0

    for episode_dir in episode_dirs:
        tfrecord_path = episode_dir / "episode_data.tfrecord"
        if not tfrecord_path.exists():
            print(f"  Skipping {episode_dir.name}: no TFRecord found")
            continue

        # Create temporary output file
        temp_path = episode_dir / "episode_data.tfrecord.tmp"
        backup_path = episode_dir / "episode_data.tfrecord.bak"

        try:
            step_count = update_episode_instruction(
                str(tfrecord_path),
                args.instruction,
                str(temp_path)
            )

            # Backup original and replace
            shutil.copy2(tfrecord_path, backup_path)
            shutil.move(temp_path, tfrecord_path)

            print(f"  Updated {episode_dir.name} ({step_count} steps)")
            updated_count += 1

        except Exception as e:
            print(f"  Error updating {episode_dir.name}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    print(f"\nUpdated {updated_count}/{len(episode_dirs)} episodes")
    print("Backup files saved as episode_data.tfrecord.bak")

    return 0


if __name__ == "__main__":
    exit(main())
