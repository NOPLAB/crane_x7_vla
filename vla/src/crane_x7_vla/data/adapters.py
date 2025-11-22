# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Data adapter for loading CRANE-X7 TFRecord data.

Provides a unified interface for loading data and converting to different formats.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Tuple
import logging

logger = logging.getLogger(__name__)


class CraneX7DataAdapter:
    """
    Adapter for loading CRANE-X7 TFRecord data.

    Handles loading from TFRecord files and provides data in various formats
    for different VLA backends.
    """

    # TFRecord feature description (CRANE-X7 format)
    FEATURE_DESCRIPTION = {
        'observation/state': tf.io.FixedLenFeature([8], tf.float32),
        'observation/image': tf.io.FixedLenFeature([], tf.string),
        'observation/depth': tf.io.FixedLenFeature([], tf.string),
        'observation/timestamp': tf.io.FixedLenFeature([1], tf.float32),
        'action': tf.io.FixedLenFeature([8], tf.float32),
        'prompt': tf.io.FixedLenFeature([], tf.string),
        'task': tf.io.FixedLenFeature([], tf.string),
    }

    # Alternative key names (for compatibility)
    KEY_ALIASES = {
        'observation/proprio': 'observation/state',
        'observation/image_primary': 'observation/image',
        'observation/depth_primary': 'observation/depth',
        'task/language_instruction': 'prompt',
        'dataset_name': 'task',
    }

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        shuffle: bool = True,
        buffer_size: int = 1000,
        include_depth: bool = False
    ):
        """
        Initialize data adapter.

        Args:
            data_root: Root directory containing TFRecord files
            split: Data split ('train' or 'val')
            shuffle: Whether to shuffle the dataset
            buffer_size: Buffer size for shuffling
            include_depth: Whether to include depth images
        """
        self.data_root = Path(data_root)
        self.split = split
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.include_depth = include_depth

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()
        logger.info(f"Found {len(self.tfrecord_files)} TFRecord files in {self.data_root}")

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files in the data directory."""
        pattern = f"**/*{self.split}*.tfrecord*"
        files = list(self.data_root.glob(pattern))

        if not files:
            # Try without split filter
            files = list(self.data_root.glob("**/*.tfrecord*"))

        return sorted(files)

    def _parse_example(self, example_proto):
        """Parse a single TFRecord example."""
        # Create a flexible feature description
        feature_description = {}
        for key, feature in self.FEATURE_DESCRIPTION.items():
            feature_description[key] = feature

        # Also add aliases
        for alias_key, real_key in self.KEY_ALIASES.items():
            if real_key in self.FEATURE_DESCRIPTION:
                feature_description[alias_key] = self.FEATURE_DESCRIPTION[real_key]

        # Parse with flexible keys (some might be missing)
        try:
            parsed = tf.io.parse_single_example(example_proto, feature_description)
        except tf.errors.InvalidArgumentError:
            # Try with minimal required features only
            minimal_description = {
                'observation/state': tf.io.FixedLenFeature([8], tf.float32),
                'observation/image': tf.io.FixedLenFeature([], tf.string),
                'action': tf.io.FixedLenFeature([8], tf.float32),
            }
            # Try aliases
            for alias, real in self.KEY_ALIASES.items():
                if real in minimal_description:
                    minimal_description[alias] = minimal_description[real]

            parsed = tf.io.parse_single_example(example_proto, minimal_description)

        # Normalize keys (convert aliases to standard names)
        normalized = {}
        for key, value in parsed.items():
            if key in self.KEY_ALIASES:
                normalized[self.KEY_ALIASES[key]] = value
            else:
                normalized[key] = value

        return normalized

    def _decode_image(self, image_bytes):
        """Decode JPEG-encoded image."""
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        return image

    def _decode_depth(self, depth_bytes):
        """Decode depth image from bytes."""
        # Depth is stored as float32 bytes
        depth = tf.io.decode_raw(depth_bytes, tf.float32)
        # Note: We don't know the original shape here
        # This is a limitation - ideally shape should be stored in metadata
        return depth

    def get_tf_dataset(
        self,
        batch_size: int = 32,
        drop_remainder: bool = False
    ) -> tf.data.Dataset:
        """
        Get TensorFlow dataset.

        Args:
            batch_size: Batch size
            drop_remainder: Whether to drop the last incomplete batch

        Returns:
            tf.data.Dataset
        """
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(
            [str(f) for f in self.tfrecord_files],
            num_parallel_reads=tf.data.AUTOTUNE
        )

        # Parse examples
        dataset = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Decode images
        def decode_images(example):
            example['observation/image'] = self._decode_image(example['observation/image'])
            if self.include_depth and 'observation/depth' in example:
                example['observation/depth'] = self._decode_depth(example['observation/depth'])
            return example

        dataset = dataset.map(decode_images, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        # Batch
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def iterate_episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Iterate over episodes (unbatched).

        Yields:
            Dictionary containing episode data
        """
        dataset = tf.data.TFRecordDataset([str(f) for f in self.tfrecord_files])
        dataset = dataset.map(self._parse_example)

        for example in dataset:
            # Convert to numpy
            episode = {}
            for key, value in example.items():
                if isinstance(value, tf.Tensor):
                    episode[key] = value.numpy()
                else:
                    episode[key] = value

            # Decode image
            if 'observation/image' in episode:
                episode['observation/image'] = self._decode_image(
                    episode['observation/image']
                ).numpy()

            if self.include_depth and 'observation/depth' in episode:
                episode['observation/depth'] = self._decode_depth(
                    episode['observation/depth']
                ).numpy()

            yield episode

    def compute_statistics(
        self,
        keys: List[str] = ['observation/state', 'action']
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute statistics over the dataset for normalization.

        Args:
            keys: Keys to compute statistics for

        Returns:
            Dictionary mapping keys to statistics dictionaries
        """
        logger.info("Computing dataset statistics...")

        # Collect all data
        data = {key: [] for key in keys}

        for episode in self.iterate_episodes():
            for key in keys:
                if key in episode:
                    data[key].append(episode[key])

        # Compute statistics
        stats = {}
        for key in keys:
            if data[key]:
                arr = np.array(data[key])
                stats[key] = {
                    'mean': np.mean(arr, axis=0),
                    'std': np.std(arr, axis=0),
                    'min': np.min(arr, axis=0),
                    'max': np.max(arr, axis=0),
                    'q_low': np.quantile(arr, 0.01, axis=0),
                    'q_high': np.quantile(arr, 0.99, axis=0),
                }

        logger.info(f"Computed statistics for keys: {list(stats.keys())}")
        return stats

    def __len__(self) -> int:
        """Get approximate dataset size (number of steps across all files)."""
        # This is approximate - iterating once to count
        count = sum(1 for _ in self.iterate_episodes())
        return count
