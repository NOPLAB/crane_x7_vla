#!/usr/bin/env python3
"""Read and display TFRecord episode data."""

import tensorflow as tf
import numpy as np
import sys

def read_tfrecord(tfrecord_path: str):
    """Read and display TFRecord contents."""

    print(f"=== Reading {tfrecord_path} ===\n")

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    steps = []
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        step_data = {}
        for key, feature in example.features.feature.items():
            if feature.HasField('float_list'):
                values = list(feature.float_list.value)
                step_data[key] = values
            elif feature.HasField('bytes_list'):
                step_data[key] = f"<bytes, length={len(feature.bytes_list.value[0])}>"
            elif feature.HasField('int64_list'):
                values = list(feature.int64_list.value)
                step_data[key] = values

        steps.append(step_data)

    print(f"Total steps: {len(steps)}\n")

    # Display first 3 steps
    for i in range(min(3, len(steps))):
        print(f"--- Step {i} ---")
        step = steps[i]

        if 'observation/state' in step:
            print(f"  State (8 joints): {step['observation/state']}")
        if 'action' in step:
            print(f"  Action (next state): {step['action']}")
        if 'observation/timestamp' in step:
            print(f"  Timestamp: {step['observation/timestamp'][0]}")
        if 'observation/image' in step:
            print(f"  Image: {step['observation/image']}")
            if 'observation/image/height' in step:
                h, w, c = step['observation/image/height'][0], step['observation/image/width'][0], step['observation/image/channels'][0]
                print(f"    Shape: {h}x{w}x{c}")
        if 'observation/depth' in step:
            print(f"  Depth: {step['observation/depth']}")
            if 'observation/depth/height' in step:
                h, w = step['observation/depth/height'][0], step['observation/depth/width'][0]
                print(f"    Shape: {h}x{w}")
        print()

    # Display last step
    if len(steps) > 3:
        print(f"--- Step {len(steps)-1} (last) ---")
        step = steps[-1]
        if 'observation/state' in step:
            print(f"  State (8 joints): {step['observation/state']}")
        if 'action' in step:
            print(f"  Action (next state): {step['action']}")
        print()

    # Statistics
    if steps:
        print("=== Statistics ===")
        if 'observation/state' in steps[0]:
            all_states = np.array([s['observation/state'] for s in steps])
            print(f"State shape: {all_states.shape}")
            print(f"State range: min={all_states.min():.4f}, max={all_states.max():.4f}")
        if 'action' in steps[0]:
            all_actions = np.array([s['action'] for s in steps])
            print(f"Action shape: {all_actions.shape}")
            print(f"Action range: min={all_actions.min():.4f}, max={all_actions.max():.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: read_tfrecord.py <file.tfrecord>')
        sys.exit(1)

    read_tfrecord(sys.argv[1])
