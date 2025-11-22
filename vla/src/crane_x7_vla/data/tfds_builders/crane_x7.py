# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
TFDS builder for CRANE-X7 dataset.

This module provides a TensorFlow Datasets builder for loading CRANE-X7
manipulation data in a format compatible with OpenVLA.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


_DESCRIPTION = """
CRANE-X7 robotic manipulation dataset.

This dataset contains demonstrations of pick-and-place tasks performed
with the CRANE-X7 robotic arm. Each episode includes:
- Joint state observations (7 arm joints + 1 gripper)
- RGB camera images
- Optional depth images
- Language instructions
"""

_CITATION = """
@misc{crane_x7_vla_2025,
  author = {nop},
  title = {CRANE-X7 VLA Dataset},
  year = {2025},
}
"""


class Crane_x7(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for crane_x7 dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(None, None, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                        ),
                        'depth': tfds.features.Tensor(
                            shape=(None, None),
                            dtype=np.float32,
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                    ),
                    'language_instruction': tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
            homepage='https://github.com/nop/crane_x7_vla',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # The data directory should be passed via data_dir parameter
        # Try manual_dir first, then fall back to data_dir parent
        if dl_manager.manual_dir:
            data_path = Path(dl_manager.manual_dir)
        else:
            # When called with tfds.builder('crane_x7', data_dir='/path/to/data'),
            # the actual episode data is in the data_dir itself, not in a subdirectory
            # dl_manager._data_dir points to /path/to/data/crane_x7, so we use parent
            data_path = Path(dl_manager._data_dir).parent

        print(f"Loading CRANE-X7 dataset from: {data_path}")

        return {
            'train': self._generate_examples(data_path),
        }

    def _generate_examples(self, data_path: Path) -> Iterator[tuple[str, Dict[str, Any]]]:
        """Yields examples from the dataset."""
        # Find all episode directories
        episode_dirs = sorted(data_path.glob('episode_*'))

        for episode_idx, episode_dir in enumerate(episode_dirs):
            tfrecord_path = episode_dir / 'episode_data.tfrecord'

            if not tfrecord_path.exists():
                continue

            # Read the TFRecord file
            steps = []
            for serialized_example in tf.data.TFRecordDataset(str(tfrecord_path)):
                example = tf.train.Example()
                example.ParseFromString(serialized_example.numpy())

                features = example.features.feature

                # Extract observation/state (joint positions)
                state = np.array(
                    features['observation/state'].float_list.value,
                    dtype=np.float32
                )

                # Extract action (next joint positions)
                action = np.array(
                    features['action'].float_list.value,
                    dtype=np.float32
                )

                # Extract image (JPEG encoded)
                image_bytes = features['observation/image'].bytes_list.value[0]

                # Decode image
                image = tf.io.decode_jpeg(image_bytes).numpy()

                # Extract depth if present
                depth = None
                if 'observation/depth' in features:
                    depth_bytes = features['observation/depth'].bytes_list.value[0]
                    depth = np.frombuffer(depth_bytes, dtype=np.float32)
                    # Reshape depth to image dimensions
                    # Assuming depth has same spatial dimensions as image
                    depth = depth.reshape(image.shape[:2])

                # Extract language instruction if present
                language_instruction = "manipulate objects"  # Default
                if 'language_instruction' in features:
                    language_instruction = features['language_instruction'].bytes_list.value[0].decode('utf-8')
                elif 'prompt' in features:
                    language_instruction = features['prompt'].bytes_list.value[0].decode('utf-8')

                step = {
                    'observation': {
                        'image': image,
                        'depth': depth if depth is not None else np.zeros_like(image[:, :, 0], dtype=np.float32),
                        'state': state,
                    },
                    'action': action,
                    'language_instruction': language_instruction,
                }

                steps.append(step)

            if steps:
                yield f'episode_{episode_idx:06d}', {
                    'steps': steps,
                }
