# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Dataset for OpenVLA Training.

This dataset loader is specifically designed for CRANE-X7 robot data,
which uses 7-axis joint angles + gripper (8 dimensions total).
The dataset follows OpenVLA's RLDS format and uses joint angle actions directly,
similar to other datasets in the Open X-Embodiment mixture (e.g., Berkeley Cable Routing).

This implementation is aligned with the original OpenVLA finetune.py implementation.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


# Image augmentation configuration (matches OpenVLA finetune.py)
DEFAULT_IMAGE_AUG_KWARGS = {
    "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    "random_brightness": [0.2],
    "random_contrast": [0.8, 1.2],
    "random_saturation": [0.8, 1.2],
    "random_hue": [0.05],
    "augment_order": [
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
}


@dataclass
class CraneX7DatasetConfig:
    """Configuration for CRANE-X7 dataset."""

    data_root: Path
    """Root directory containing TFRecord episode files"""

    action_dim: int = 8
    """Action dimension (7 joint angles + 1 gripper)"""

    state_dim: int = 8
    """State dimension (7 joint angles + 1 gripper)"""

    normalize_actions: bool = True
    """Whether to normalize actions to [-1, 1] using BOUNDS_Q99 (matches OpenVLA)"""

    normalize_states: bool = True
    """Whether to normalize states"""

    image_size: Tuple[int, int] = (224, 224)
    """Target image size (height, width)"""

    use_language_instruction: bool = True
    """Whether to use language instructions"""

    default_instruction: str = "manipulate the object"
    """Default language instruction if not present"""

    normalization_stats_path: Optional[Path] = None
    """Path to normalization statistics JSON file (optional)"""

    image_aug: bool = True
    """Whether to apply image augmentation (matches OpenVLA finetune.py default)"""

    image_aug_kwargs: Optional[Dict[str, Any]] = None
    """Image augmentation kwargs (uses DEFAULT_IMAGE_AUG_KWARGS if None)"""


@dataclass
class CraneX7BatchTransform:
    """
    Batch transform for CRANE-X7 data.

    This implementation matches the original OpenVLA RLDSBatchTransform exactly.
    Converts RLDS batch format to OpenVLA expected format.
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class CraneX7Dataset(IterableDataset):
    """
    PyTorch IterableDataset for CRANE-X7 robot data.

    This implementation matches the original OpenVLA RLDSDataset,
    including BOUNDS_Q99 normalization and image augmentation.

    For overfitting detection, the dataset splits individual steps (not episodes)
    into train/overfit sets using a deterministic hash-based approach.
    This ensures the overfit set contains steps from the same episodes as training,
    allowing proper detection of memorization vs generalization.
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
        data_mix: str,
        batch_transform: CraneX7BatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        overfit_split_ratio: float = 0.0,
        split: str = "train",
        split_seed: int = 42,
    ) -> None:
        """
        Lightweight wrapper around TFRecord pipeline for use with PyTorch/OpenVLA Data Loaders.

        This constructor signature matches the original OpenVLA RLDSDataset.

        Args:
            data_root_dir: Root directory containing TFRecord episode files
            data_mix: Dataset name (for CRANE-X7, this is typically "crane_x7")
            batch_transform: CraneX7BatchTransform instance
            resize_resolution: Target image resolution (height, width)
            shuffle_buffer_size: Buffer size for shuffling (default: 256_000 to match OpenVLA)
            train: Whether this is training set
            image_aug: Whether to apply image augmentation (matches OpenVLA finetune.py)
            overfit_split_ratio: Ratio of steps to use for overfitting detection (0.0 to disable)
                                 Steps are split deterministically within each episode.
            split: Dataset split to use ("train" or "overfit")
            split_seed: Random seed for deterministic step-level splitting
        """
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.batch_transform = batch_transform
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.image_aug = image_aug
        self.overfit_split_ratio = overfit_split_ratio
        self.split = split
        self.split_seed = split_seed

        # Find all TFRecord files (use all files for both splits)
        self.tfrecord_files = self._find_tfrecord_files()

        print(f"Found {len(self.tfrecord_files)} TFRecord files for {split} split in {self.data_root_dir}")
        if overfit_split_ratio > 0:
            print(f"  Using step-level splitting with {overfit_split_ratio:.1%} for overfitting detection")

        # Compute or load dataset statistics (matches OpenVLA behavior)
        self.dataset_statistics = self._get_dataset_statistics()

        # Store dataset length (stats are nested: {dataset_name: {num_transitions: int}})
        stats = self.dataset_statistics.get(self.data_mix, {})
        self.dataset_length = stats.get("num_transitions", 0)

        # Create TensorFlow dataset with normalization and augmentation
        self.dataset = self._create_tf_dataset()

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files in the data directory."""
        # Look for episode directories containing tfrecord files
        tfrecord_files = list(self.data_root_dir.glob("episode_*/episode_data.tfrecord"))

        # Also try to find tfrecord files directly
        if not tfrecord_files:
            tfrecord_files = list(self.data_root_dir.glob("**/*.tfrecord"))

        # Filter out .bak files
        tfrecord_files = [f for f in tfrecord_files if not f.name.endswith(".bak")]

        return sorted(tfrecord_files)

    def _get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Compute or load dataset statistics for normalization.

        This matches the OpenVLA behavior of computing q01/q99 for BOUNDS_Q99 normalization.
        Returns statistics in the format expected by save_dataset_statistics:
        {dataset_name: {action: {...}, proprio: {...}, num_transitions: int, num_trajectories: int}}
        """
        import json

        # Try to load existing statistics
        stats_path = self.data_root_dir / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                full_stats = json.load(f)
                # Handle nested format (dataset_name -> action -> stats)
                if self.data_mix in full_stats:
                    print(f"Loaded dataset statistics from {stats_path}")
                    return full_stats  # Return full nested format for save_dataset_statistics
                elif "crane_x7" in full_stats:
                    print(f"Loaded dataset statistics from {stats_path}")
                    return full_stats  # Return full nested format
                else:
                    # Old format - wrap in dataset name
                    print(f"Loaded dataset statistics from {stats_path} (converting to nested format)")
                    return {self.data_mix: full_stats}

        # Compute statistics if not found
        print("Computing dataset statistics (this may take a moment)...")
        actions = []
        proprios = []
        num_transitions = 0

        for tfrecord_file in self.tfrecord_files:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            for raw_record in dataset:
                example = tf.io.parse_single_example(raw_record, {
                    "action": tf.io.FixedLenFeature([8], tf.float32),
                    "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
                })
                actions.append(example["action"].numpy())
                proprios.append(example["observation/proprio"].numpy())
                num_transitions += 1

        actions = np.array(actions)
        proprios = np.array(proprios)

        stats = {
            "action": {
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "max": actions.max(0).tolist(),
                "min": actions.min(0).tolist(),
                "q01": np.quantile(actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions, 0.99, axis=0).tolist(),
            },
            "proprio": {
                "mean": proprios.mean(0).tolist(),
                "std": proprios.std(0).tolist(),
                "max": proprios.max(0).tolist(),
                "min": proprios.min(0).tolist(),
                "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
                "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
            },
            "num_transitions": num_transitions,
            "num_trajectories": len(self.tfrecord_files),
        }

        # Wrap in dataset name (format expected by save_dataset_statistics)
        full_stats = {self.data_mix: stats}

        # Save statistics for future use
        with open(stats_path, "w") as f:
            json.dump(full_stats, f, indent=2)
        print(f"Saved dataset statistics to {stats_path}")

        return full_stats

    def _parse_example(self, example_proto):
        """Parse a single TFRecord example."""
        try:
            parsed = tf.io.parse_single_example(example_proto, self.FEATURE_DESCRIPTION)
        except tf.errors.InvalidArgumentError:
            # Try with minimal features if full parse fails
            minimal_description = {
                "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
                "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
                "action": tf.io.FixedLenFeature([8], tf.float32),
            }
            parsed = tf.io.parse_single_example(example_proto, minimal_description)

            # Add defaults for missing keys
            parsed["task/language_instruction"] = tf.constant(b"manipulate the object")
            parsed["dataset_name"] = tf.constant(b"crane_x7")
            parsed["observation/timestep"] = tf.constant([0], dtype=tf.int64)

        return parsed

    def _normalize_action(self, action: tf.Tensor) -> tf.Tensor:
        """
        Normalize action using BOUNDS_Q99 (matches OpenVLA).

        Maps [q01, q99] -> [-1, 1] and clips to [-1, 1].
        """
        # Get stats for this dataset (dataset_statistics is nested: {dataset_name: {action: {...}}})
        stats = self.dataset_statistics[self.data_mix]

        q01 = tf.constant(stats["action"]["q01"], dtype=tf.float32)
        q99 = tf.constant(stats["action"]["q99"], dtype=tf.float32)

        # Normalize to [-1, 1]
        normalized = 2.0 * (action - q01) / (q99 - q01 + 1e-8) - 1.0

        # Clip to [-1, 1]
        normalized = tf.clip_by_value(normalized, -1.0, 1.0)

        # Map unused dimensions (where min == max) to 0
        action_min = tf.constant(stats["action"]["min"], dtype=tf.float32)
        action_max = tf.constant(stats["action"]["max"], dtype=tf.float32)
        zeros_mask = tf.equal(action_min, action_max)
        normalized = tf.where(zeros_mask, 0.0, normalized)

        return normalized

    def _decode_and_resize_image(self, image_bytes: tf.Tensor) -> tf.Tensor:
        """Decode JPEG image and resize to target resolution."""
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, self.resize_resolution)
        image = tf.cast(image, tf.uint8)
        return image

    def _apply_image_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply image augmentation (matches OpenVLA finetune.py).

        Augmentations:
        - Random resized crop (scale=[0.9, 0.9], ratio=[1.0, 1.0])
        - Random brightness (max_delta=0.2)
        - Random contrast (lower=0.8, upper=1.2)
        - Random saturation (lower=0.8, upper=1.2)
        - Random hue (max_delta=0.05)
        """
        # Convert to float for augmentation
        image = tf.cast(image, tf.float32) / 255.0

        # Random resized crop (scale=0.9 means crop 90% of image)
        crop_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * 0.9, tf.int32)
        image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
        image = tf.image.resize(image, self.resize_resolution)

        # Color augmentations
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.05)

        # Clip values and convert back to uint8
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.cast(image * 255.0, tf.uint8)

        return image

    def _process_example(self, example):
        """Process a single example: decode image, normalize action, apply augmentation."""
        # Decode and resize image
        image = self._decode_and_resize_image(example["observation/image_primary"])

        # Apply image augmentation if enabled
        if self.image_aug:
            image = self._apply_image_augmentation(image)

        # Normalize action using BOUNDS_Q99
        action = self._normalize_action(example["action"])

        # Restructure to RLDS batch format (with window_size=1, so add batch dim)
        return {
            "observation": {
                "image_primary": tf.expand_dims(image, 0),  # [1, H, W, 3]
                "proprio": tf.expand_dims(example["observation/proprio"], 0),  # [1, 8]
            },
            "task": {
                "language_instruction": example["task/language_instruction"],
            },
            "action": tf.expand_dims(action, 0),  # [1, 8]
            "dataset_name": example["dataset_name"],
        }

    def _should_include_step(self, step_index: tf.Tensor) -> tf.Tensor:
        """
        Determine if a step should be included based on split type.

        Uses a deterministic hash-based approach to split steps within episodes.
        This ensures reproducible splits regardless of data order.

        Args:
            step_index: Global step index in the dataset

        Returns:
            Boolean tensor indicating if this step should be included
        """
        if self.overfit_split_ratio <= 0:
            return tf.constant(True)

        # Use deterministic hash for splitting
        # Hash the step index with the seed to get a pseudo-random value
        hash_value = tf.bitwise.bitwise_xor(
            tf.cast(step_index, tf.int64),
            tf.constant(self.split_seed, dtype=tf.int64)
        )
        # Use modulo to get a value in [0, 1000)
        mod_value = tf.math.abs(hash_value) % 1000
        threshold = tf.cast(self.overfit_split_ratio * 1000, tf.int64)

        if self.split == "overfit":
            # Include steps where mod_value < threshold (overfit set)
            return mod_value < threshold
        else:
            # Include steps where mod_value >= threshold (train set)
            return mod_value >= threshold

    def _create_tf_dataset(self) -> tf.data.Dataset:
        """Create TensorFlow dataset pipeline with normalization and augmentation."""
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(
            [str(f) for f in self.tfrecord_files],
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        # Add step index for deterministic splitting
        dataset = dataset.enumerate()

        # Parse examples and add step index
        def parse_with_index(index, example_proto):
            parsed = self._parse_example(example_proto)
            parsed["_step_index"] = index
            return parsed

        dataset = dataset.map(
            parse_with_index,
            num_parallel_calls=16,  # Match OpenVLA: num_parallel_calls=16
        )

        # Filter based on train/overfit split (step-level splitting)
        if self.overfit_split_ratio > 0:
            dataset = dataset.filter(
                lambda x: self._should_include_step(x["_step_index"])
            )

        # Process: decode image, normalize action, apply augmentation
        dataset = dataset.map(
            self._process_example,
            num_parallel_calls=16,
        )

        # Shuffle if training (with large buffer to match OpenVLA)
        if self.train:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # Repeat for training
        if self.train:
            dataset = dataset.repeat()

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
