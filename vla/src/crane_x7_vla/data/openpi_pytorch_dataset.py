# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Dataset for OpenPI PyTorch Training.

This dataset loader is designed for training Pi0 models with action chunks.
It loads CRANE-X7 robot data (8 DOF) and pads to Pi0's expected format (32 DOF).
Each sample includes an action chunk of future actions for flow matching training.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from crane_x7_vla.transforms.action_transforms import (
    ActionChunker,
    ActionNormalizer,
    ActionPadder,
)


class CraneX7ActionChunkDataset(IterableDataset):
    """
    PyTorch IterableDataset for CRANE-X7 robot data with action chunks.

    This dataset:
    - Loads TFRecord data in episode format
    - Pads 8 DOF actions/states to 32 DOF for Pi0 compatibility
    - Returns action chunks (50 future actions) for flow matching training
    - Supports image augmentation and normalization
    """

    # TFRecord feature description for CRANE-X7 format
    FEATURE_DESCRIPTION = {
        "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
        "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
        "observation/timestep": tf.io.FixedLenFeature([1], tf.int64),
        "action": tf.io.FixedLenFeature([8], tf.float32),
        "task/language_instruction": tf.io.FixedLenFeature([], tf.string),
        "dataset_name": tf.io.FixedLenFeature([], tf.string),
    }

    def __init__(
        self,
        data_root_dir: Path,
        action_horizon: int = 50,
        source_action_dim: int = 8,
        target_action_dim: int = 32,
        resize_resolution: Tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 10000,
        train: bool = True,
        image_aug: bool = False,
        normalize_actions: bool = True,
        normalization_mode: str = "quantile",
        default_prompt: str = "manipulate objects",
        overfit_split_ratio: float = 0.0,
        split: str = "train",
        split_seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize action chunk dataset.

        Args:
            data_root_dir: Root directory containing TFRecord episode files
            action_horizon: Number of future actions to predict (chunk size)
            source_action_dim: Source action dimension (CRANE-X7 has 8 DOF)
            target_action_dim: Target action dimension (Pi0 uses 32)
            resize_resolution: Target image resolution (height, width)
            shuffle_buffer_size: Buffer size for shuffling episodes
            train: Whether this is training set
            image_aug: Whether to apply image augmentation
            normalize_actions: Whether to normalize actions
            normalization_mode: Normalization mode ("quantile" or "zscore")
            default_prompt: Default language instruction
            overfit_split_ratio: Ratio of steps for overfitting detection
            split: Dataset split ("train" or "overfit")
            split_seed: Random seed for splitting
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.action_horizon = action_horizon
        self.source_action_dim = source_action_dim
        self.target_action_dim = target_action_dim
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.image_aug = image_aug
        self.normalize_actions = normalize_actions
        self.normalization_mode = normalization_mode
        self.default_prompt = default_prompt
        self.overfit_split_ratio = overfit_split_ratio
        self.split = split
        self.split_seed = split_seed
        self.rank = rank
        self.world_size = world_size

        # Initialize transforms
        self.action_padder = ActionPadder(source_action_dim, target_action_dim)
        self.action_chunker = ActionChunker(action_horizon, interpolation="linear")
        self.action_normalizer = ActionNormalizer(mode=normalization_mode)

        # Find all TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()
        print(f"[Rank {self.rank}/{self.world_size}] Found {len(self.tfrecord_files)} TFRecord files")

        # Load or compute statistics
        self.dataset_statistics = self._get_dataset_statistics()

        # Fit normalizer with statistics
        if self.normalize_actions and self.dataset_statistics:
            self._fit_normalizer()

        # Load episodes into memory for efficient chunking
        self.episodes = self._load_episodes()
        print(f"[Rank {self.rank}/{self.world_size}] Loaded {len(self.episodes)} episodes")

        # Compute total number of valid samples (each step that can form a full chunk)
        self.total_samples = sum(
            max(0, len(ep) - self.action_horizon)
            for ep in self.episodes
        )
        print(f"[Rank {self.rank}/{self.world_size}] Total samples: {self.total_samples}")

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files in the data directory."""
        tfrecord_files = list(self.data_root_dir.glob("episode_*/episode_data.tfrecord"))
        if not tfrecord_files:
            tfrecord_files = list(self.data_root_dir.glob("**/*.tfrecord"))
        tfrecord_files = [f for f in tfrecord_files if not f.name.endswith(".bak")]
        return sorted(tfrecord_files)

    def _get_dataset_statistics(self) -> Dict[str, Any]:
        """Load or compute dataset statistics."""
        stats_path = self.data_root_dir / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
                # Handle nested format
                if "crane_x7" in stats:
                    return stats["crane_x7"]
                return stats
        return {}

    def _fit_normalizer(self) -> None:
        """Fit action normalizer with dataset statistics."""
        if not self.dataset_statistics:
            return

        action_stats = self.dataset_statistics.get("action", {})
        if "q01" in action_stats and "q99" in action_stats:
            self.action_normalizer.stats = {
                "q_low": np.array(action_stats["q01"]),
                "q_high": np.array(action_stats["q99"]),
                "range": np.array(action_stats["q99"]) - np.array(action_stats["q01"]),
            }
            # Avoid division by zero
            self.action_normalizer.stats["range"] = np.where(
                self.action_normalizer.stats["range"] < 1e-6,
                1.0,
                self.action_normalizer.stats["range"]
            )

    def _load_episodes(self) -> List[List[Dict[str, Any]]]:
        """Load all episodes from TFRecord files."""
        episodes = []

        # Shard files for distributed training
        files_to_load = self.tfrecord_files
        if self.world_size > 1:
            files_to_load = [
                f for i, f in enumerate(self.tfrecord_files)
                if i % self.world_size == self.rank
            ]

        for tfrecord_file in files_to_load:
            episode = []
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))

            for raw_record in dataset:
                try:
                    example = tf.io.parse_single_example(
                        raw_record, self.FEATURE_DESCRIPTION
                    )
                except tf.errors.InvalidArgumentError:
                    # Try minimal features
                    minimal_desc = {
                        "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
                        "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
                        "action": tf.io.FixedLenFeature([8], tf.float32),
                    }
                    example = tf.io.parse_single_example(raw_record, minimal_desc)
                    example["task/language_instruction"] = tf.constant(
                        self.default_prompt.encode()
                    )

                step = {
                    "state": example["observation/proprio"].numpy(),
                    "action": example["action"].numpy(),
                    "image_bytes": example["observation/image_primary"].numpy(),
                    "prompt": example.get(
                        "task/language_instruction",
                        tf.constant(self.default_prompt.encode())
                    ).numpy().decode(),
                }
                episode.append(step)

            if len(episode) >= self.action_horizon:
                episodes.append(episode)

        return episodes

    def _decode_and_process_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode and process image."""
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, self.resize_resolution)
        image = tf.cast(image, tf.uint8).numpy()

        if self.image_aug and self.train:
            image = self._apply_image_augmentation(image)

        return image

    def _apply_image_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply image augmentation."""
        image = tf.cast(image, tf.float32) / 255.0

        # Random resized crop
        crop_size = tf.cast(
            tf.cast(tf.shape(image)[:2], tf.float32) * 0.9, tf.int32
        )
        image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
        image = tf.image.resize(image, self.resize_resolution)

        # Color augmentations
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.05)

        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.cast(image * 255.0, tf.uint8).numpy()

        return image

    def _should_include_step(self, episode_idx: int, step_idx: int) -> bool:
        """Determine if a step should be included based on split."""
        if self.overfit_split_ratio <= 0:
            return True

        # Deterministic hash for splitting
        hash_value = hash((episode_idx, step_idx, self.split_seed))
        mod_value = abs(hash_value) % 1000
        threshold = int(self.overfit_split_ratio * 1000)

        if self.split == "overfit":
            return mod_value < threshold
        else:
            return mod_value >= threshold

    def __iter__(self):
        """Iterate over dataset yielding action chunk samples."""
        # Create list of (episode_idx, step_idx) pairs
        samples = []
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode) - self.action_horizon):
                if self._should_include_step(ep_idx, step_idx):
                    samples.append((ep_idx, step_idx))

        # Shuffle if training
        if self.train:
            np.random.shuffle(samples)

        for ep_idx, step_idx in samples:
            episode = self.episodes[ep_idx]

            # Current observation
            current_step = episode[step_idx]

            # Decode and process image
            image = self._decode_and_process_image(current_step["image_bytes"])

            # Current state (pad to target dim)
            state = current_step["state"]
            if self.normalize_actions and self.action_normalizer.stats:
                state = self.action_normalizer.normalize(state)
            state_padded = self.action_padder.pad(state)

            # Action chunk (future actions)
            action_chunk = []
            for i in range(self.action_horizon):
                action = episode[step_idx + i]["action"]
                if self.normalize_actions and self.action_normalizer.stats:
                    action = self.action_normalizer.normalize(action)
                action_padded = self.action_padder.pad(action)
                action_chunk.append(action_padded)

            action_chunk = np.stack(action_chunk, axis=0)  # (horizon, 32)

            # Prompt
            prompt = current_step.get("prompt", self.default_prompt)

            yield {
                "observation": {
                    "state": torch.tensor(state_padded, dtype=torch.float32),
                    "image": {
                        "base_0_rgb": torch.tensor(image, dtype=torch.uint8),
                    },
                },
                "actions": torch.tensor(action_chunk, dtype=torch.float32),
                "prompt": prompt,
            }

    def __len__(self) -> int:
        return self.total_samples


def collate_action_chunk_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for action chunk dataset.

    Args:
        batch: List of samples from CraneX7ActionChunkDataset

    Returns:
        Collated batch with batched tensors
    """
    states = torch.stack([b["observation"]["state"] for b in batch])
    images = torch.stack([b["observation"]["image"]["base_0_rgb"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])
    prompts = [b["prompt"] for b in batch]

    return {
        "observation": {
            "state": states,
            "image": {
                "base_0_rgb": images,
            },
        },
        "actions": actions,
        "prompts": prompts,
    }
