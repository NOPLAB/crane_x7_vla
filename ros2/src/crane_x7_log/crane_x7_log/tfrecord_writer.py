#!/usr/bin/env python3
# Copyright 2025
# Licensed under the MIT License

"""TFRecord writer for robot data logging."""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Any
import os


class TFRecordWriter:
    """Writes episode data to TFRecord format."""

    def __init__(self, output_path: str):
        """
        Initialize TFRecord writer.

        Args:
            output_path: Path to output TFRecord file
        """
        self.output_path = output_path
        self.writer = tf.io.TFRecordWriter(output_path)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def encode_image(self, image: np.ndarray) -> bytes:
        """Encode image to JPEG bytes."""
        import cv2
        _, encoded = cv2.imencode('.jpg', image)
        return encoded.tobytes()

    def create_step_example(self, step_data: Dict[str, Any]) -> tf.train.Example:
        """
        Create a TFRecord Example for one step.

        Args:
            step_data: Dictionary containing:
                - observation/state: np.ndarray of joint positions
                - observation/image: np.ndarray RGB image (optional)
                - observation/depth: np.ndarray depth image (optional)
                - action: np.ndarray of actions
                - timestamp: float

        Returns:
            tf.train.Example
        """
        feature = {}

        # State (joint positions)
        if 'observation' in step_data and 'state' in step_data['observation']:
            state = step_data['observation']['state'].astype(np.float32).flatten()
            feature['observation/state'] = self._float_feature(state.tolist())

        # Action
        if 'action' in step_data:
            action = step_data['action'].astype(np.float32).flatten()
            feature['action'] = self._float_feature(action.tolist())

        # RGB Image
        if 'observation' in step_data and 'image' in step_data['observation']:
            image = step_data['observation']['image']
            encoded_image = self.encode_image(image)
            feature['observation/image'] = self._bytes_feature(encoded_image)
            feature['observation/image/height'] = self._int64_feature([image.shape[0]])
            feature['observation/image/width'] = self._int64_feature([image.shape[1]])
            feature['observation/image/channels'] = self._int64_feature([image.shape[2]])

        # Depth Image
        if 'observation' in step_data and 'depth' in step_data['observation']:
            depth = step_data['observation']['depth']
            depth_bytes = depth.astype(np.float32).tobytes()
            feature['observation/depth'] = self._bytes_feature(depth_bytes)
            feature['observation/depth/height'] = self._int64_feature([depth.shape[0]])
            feature['observation/depth/width'] = self._int64_feature([depth.shape[1]])

        # Timestamp
        if 'observation' in step_data and 'timestamp' in step_data['observation']:
            timestamp = float(step_data['observation']['timestamp'])
            feature['observation/timestamp'] = self._float_feature([timestamp])

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_episode(self, episode_data: List[Dict[str, Any]]):
        """
        Write an episode (sequence of steps) to TFRecord.

        Args:
            episode_data: List of step dictionaries
        """
        for step_data in episode_data:
            example = self.create_step_example(step_data)
            self.writer.write(example.SerializeToString())

    def close(self):
        """Close the TFRecord writer."""
        self.writer.close()


def convert_npz_to_tfrecord(npz_path: str, tfrecord_path: str):
    """
    Convert saved NPZ episode data to TFRecord format.

    Args:
        npz_path: Path to input NPZ file
        tfrecord_path: Path to output TFRecord file
    """
    # Load NPZ data
    data = np.load(npz_path)

    states = data['states']
    actions = data['actions']
    timestamps = data['timestamps']

    # Optional data
    images = data['images'] if 'images' in data else None
    depths = data['depths'] if 'depths' in data else None

    # Create episode data structure
    episode_data = []
    for i in range(len(states)):
        step_data = {
            'observation': {
                'state': states[i],
                'timestamp': timestamps[i]
            },
            'action': actions[i]
        }

        if images is not None:
            step_data['observation']['image'] = images[i]

        if depths is not None:
            step_data['observation']['depth'] = depths[i]

        episode_data.append(step_data)

    # Write to TFRecord
    writer = TFRecordWriter(tfrecord_path)
    writer.write_episode(episode_data)
    writer.close()

    print(f'Converted {npz_path} to {tfrecord_path}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('Usage: tfrecord_writer.py <input.npz> <output.tfrecord>')
        sys.exit(1)

    npz_path = sys.argv[1]
    tfrecord_path = sys.argv[2]

    convert_npz_to_tfrecord(npz_path, tfrecord_path)
