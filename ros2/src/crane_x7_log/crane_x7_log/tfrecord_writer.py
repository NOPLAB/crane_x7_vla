#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""TFRecord writer for robot data logging in RLDS format."""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Any, Optional
import os


class TFRecordWriter:
    """Writes episode data to RLDS-compatible TFRecord format."""

    # Supported image keys (RLDS naming convention)
    IMAGE_KEYS = ['image_primary', 'image_secondary', 'image_wrist']
    DEPTH_KEYS = ['depth_primary', 'depth_secondary', 'depth_wrist']

    def __init__(
        self,
        output_path: str,
        dataset_name: str = "crane_x7",
        language_instruction: Optional[str] = None
    ):
        """
        Initialize TFRecord writer.

        Args:
            output_path: Path to output TFRecord file
            dataset_name: Dataset identifier for RLDS
            language_instruction: Language instruction for the episode
        """
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.language_instruction = language_instruction or "manipulate the object"
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

    def create_step_example(
        self,
        step_data: Dict[str, Any],
        timestep: int
    ) -> tf.train.Example:
        """
        Create an RLDS-formatted TFRecord Example for one step.

        Args:
            step_data: Dictionary containing observation and action data
            timestep: Integer timestep index

        Returns:
            tf.train.Example in RLDS format
        """
        feature = {}

        # Proprioceptive state (renamed from 'state' to 'proprio' for RLDS)
        if 'observation' in step_data and 'state' in step_data['observation']:
            proprio = step_data['observation']['state'].astype(np.float32).flatten()
            feature['observation/proprio'] = self._float_feature(proprio.tolist())

        # Timestep (changed from timestamp float to timestep int)
        feature['observation/timestep'] = self._int64_feature([timestep])

        # Action
        if 'action' in step_data:
            action = step_data['action'].astype(np.float32).flatten()
            feature['action'] = self._float_feature(action.tolist())

        # Multiple camera RGB images (RLDS naming convention)
        if 'observation' in step_data:
            obs = step_data['observation']

            # Handle multiple image keys (image_primary, image_secondary, image_wrist)
            for image_key in self.IMAGE_KEYS:
                if image_key in obs:
                    image = obs[image_key]
                    encoded_image = self.encode_image(image)
                    feature[f'observation/{image_key}'] = self._bytes_feature(
                        encoded_image
                    )

            # Handle multiple depth keys (depth_primary, depth_secondary, depth_wrist)
            for depth_key in self.DEPTH_KEYS:
                if depth_key in obs:
                    depth = obs[depth_key]
                    depth_bytes = depth.astype(np.float32).tobytes()
                    feature[f'observation/{depth_key}'] = self._bytes_feature(
                        depth_bytes
                    )

        # Dataset name (required for RLDS)
        feature['dataset_name'] = self._bytes_feature(
            self.dataset_name.encode('utf-8')
        )

        # Language instruction (required for VLA models)
        feature['task/language_instruction'] = self._bytes_feature(
            self.language_instruction.encode('utf-8')
        )

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_episode(self, episode_data: List[Dict[str, Any]]):
        """
        Write an episode (sequence of steps) to RLDS-formatted TFRecord.

        Args:
            episode_data: List of step dictionaries
        """
        for timestep, step_data in enumerate(episode_data):
            example = self.create_step_example(step_data, timestep)
            self.writer.write(example.SerializeToString())

    def close(self):
        """Close the TFRecord writer."""
        self.writer.close()


def convert_npz_to_tfrecord(
    npz_path: str,
    tfrecord_path: str,
    dataset_name: str = "crane_x7",
    language_instruction: str = "manipulate the object"
):
    """
    Convert saved NPZ episode data to RLDS-formatted TFRecord.

    Args:
        npz_path: Path to input NPZ file
        tfrecord_path: Path to output TFRecord file
        dataset_name: Dataset identifier
        language_instruction: Task description
    """
    # Load NPZ data
    data = np.load(npz_path)

    states = data['states']
    actions = data['actions']
    timestamps = data['timestamps']

    # Find image keys (images_primary, images_secondary, images_wrist)
    image_keys = [k for k in data.files if k.startswith('images_')]
    depth_keys = [k for k in data.files if k.startswith('depths_')]

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

        # Add images from all cameras
        for image_key in image_keys:
            # Convert images_primary -> image_primary
            obs_key = image_key.replace('images_', 'image_')
            step_data['observation'][obs_key] = data[image_key][i]

        # Add depths from all cameras
        for depth_key in depth_keys:
            # Convert depths_primary -> depth_primary
            obs_key = depth_key.replace('depths_', 'depth_')
            step_data['observation'][obs_key] = data[depth_key][i]

        episode_data.append(step_data)

    # Write to TFRecord in RLDS format
    writer = TFRecordWriter(
        tfrecord_path,
        dataset_name=dataset_name,
        language_instruction=language_instruction
    )
    writer.write_episode(episode_data)
    writer.close()

    print(f'Converted {npz_path} to RLDS-formatted {tfrecord_path}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: tfrecord_writer.py <input.npz> <output.tfrecord> [dataset_name] [language_instruction]')
        print('\nExample:')
        print('  python tfrecord_writer.py episode.npz episode.tfrecord')
        print('  python tfrecord_writer.py episode.npz episode.tfrecord crane_x7 "pick up the red cube"')
        sys.exit(1)

    npz_path = sys.argv[1]
    tfrecord_path = sys.argv[2]
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "crane_x7"
    language_instruction = sys.argv[4] if len(sys.argv) > 4 else "manipulate the object"

    convert_npz_to_tfrecord(npz_path, tfrecord_path, dataset_name, language_instruction)
