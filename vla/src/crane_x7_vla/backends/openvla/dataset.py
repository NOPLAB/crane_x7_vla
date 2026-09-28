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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from crane_x7_vla.backends.common.prompting import PromptBuilder
from crane_x7_vla.backends.common.tokenizer import ActionTokenizer
from crane_x7_vla.backends.common.types import ImageTransform
from crane_x7_vla.core.data.image_augmentation import ImageAugmentationConfig, ImageAugmentor
from crane_x7_vla.core.data.image_utils import decode_jpeg, resize_image
from crane_x7_vla.core.data.tfrecord_reader import TFRecordReader, find_tfrecord_files


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

    image_size: tuple[int, int] = (224, 224)
    """Target image size (height, width)"""

    use_language_instruction: bool = True
    """Whether to use language instructions"""

    default_instruction: str = "manipulate the object"
    """Default language instruction if not present"""

    normalization_stats_path: Path | None = None
    """Path to normalization statistics JSON file (optional)"""

    image_aug: bool = True
    """Whether to apply image augmentation (matches OpenVLA finetune.py default)"""

    image_aug_kwargs: dict[str, Any] | None = None
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
    prompt_builder_fn: type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"]
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8", errors="replace")
        lang = lang.lower()

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

        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels, "dataset_name": dataset_name}


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

    # TFRecord feature description for CRANE-X7 format (tfrecord library format)
    FEATURE_DESCRIPTION: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "observation/timestep": "int",
        "action": "float",
        "task/language_instruction": "byte",
        "dataset_name": "byte",
        # Optional multi-camera support
        "observation/image_secondary": "byte",
        "observation/image_wrist": "byte",
    }

    # Minimal feature description for backward compatibility
    FEATURE_DESCRIPTION_MINIMAL: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "action": "float",
    }

    # Optional image keys for multi-camera support (RLDS naming convention)
    OPTIONAL_IMAGE_KEYS: ClassVar[list[str]] = ["observation/image_secondary", "observation/image_wrist"]

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: CraneX7BatchTransform,
        resize_resolution: tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        overfit_split_ratio: float = 0.0,
        split: str = "train",
        split_seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
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
            rank: Process rank for distributed training (0-indexed)
            world_size: Total number of processes for distributed training
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
        self.rank = rank
        self.world_size = world_size

        # Find all TFRecord files (use all files for both splits)
        self.tfrecord_files = self._find_tfrecord_files()

        print(
            f"[Rank {self.rank}/{self.world_size}] Found {len(self.tfrecord_files)} TFRecord files for {split} split in {self.data_root_dir}"
        )
        if self.world_size > 1:
            print(
                f"[Rank {self.rank}/{self.world_size}] Distributed training enabled: data will be sharded across {self.world_size} processes"
            )
        if overfit_split_ratio > 0:
            print(f"  Using step-level splitting with {overfit_split_ratio:.1%} for overfitting detection")

        # Compute or load dataset statistics (matches OpenVLA behavior)
        self.dataset_statistics = self._get_dataset_statistics()

        # Store dataset length (stats are nested: {dataset_name: {num_transitions: int}})
        stats = self.dataset_statistics.get(self.data_mix, {})
        self.dataset_length = stats.get("num_transitions", 0)

        # Create TensorFlow dataset with normalization and augmentation
        self.dataset = self._create_tf_dataset()

    def _find_tfrecord_files(self) -> list[Path]:
        """Find all TFRecord files in the data directory."""
        return find_tfrecord_files(self.data_root_dir)

    def _get_dataset_statistics(self) -> dict[str, Any]:
        """
        Compute or load dataset statistics for normalization.

        This matches the OpenVLA behavior of computing q01/q99 for BOUNDS_Q99 normalization.
        Returns statistics in the format expected by save_dataset_statistics:
        {dataset_name: {action: {...}, proprio: {...}, num_transitions: int, num_trajectories: int}}

        Raises:
            FileNotFoundError: If no TFRecord files are found in the data directory.
            ValueError: If TFRecord files exist but contain no valid data.
        """

        # Check if any TFRecord files were found
        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"\n"
                f"========================================\n"
                f"ERROR: No TFRecord files found\n"
                f"========================================\n"
                f"Data directory: {self.data_root_dir}\n"
                f"\n"
                f"Expected TFRecord file patterns:\n"
                f"  - {self.data_root_dir}/episode_*/episode_data.tfrecord\n"
                f"  - {self.data_root_dir}/**/*.tfrecord\n"
                f"\n"
                f"Please ensure:\n"
                f"  1. The --data_dir path is correct\n"
                f"  2. TFRecord files have been generated using crane_x7_log\n"
                f"  3. The directory contains episode_*/episode_data.tfrecord files\n"
                f"\n"
                f"To collect data, use the ROS 2 data logging package:\n"
                f"  cd ros2 && docker compose --profile real up\n"
                f"========================================\n"
            )

        # Try to load existing statistics
        stats_path = self.data_root_dir / "dataset_statistics.json"
        if stats_path.exists():
            with stats_path.open() as f:
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

        # Use TFRecordReader with minimal features for statistics computation
        reader = TFRecordReader(
            self.tfrecord_files,
            feature_spec=self.FEATURE_DESCRIPTION_MINIMAL,
        )
        for example in reader:
            action = example.get("action")
            proprio = example.get("observation/proprio")

            if action is not None:
                action = np.array(action, dtype=np.float32).reshape(8)
                actions.append(action)

            if proprio is not None:
                proprio = np.array(proprio, dtype=np.float32).reshape(8)
                proprios.append(proprio)

            num_transitions += 1

        # Check if any data was loaded from the TFRecord files
        if num_transitions == 0:
            raise ValueError(
                f"\n"
                f"========================================\n"
                f"ERROR: TFRecord files contain no data\n"
                f"========================================\n"
                f"Found {len(self.tfrecord_files)} TFRecord file(s) in: {self.data_root_dir}\n"
                f"But no valid transitions could be read from them.\n"
                f"\n"
                f"TFRecord files found:\n"
                + "\n".join(f"  - {f}" for f in self.tfrecord_files[:10])
                + (f"\n  ... and {len(self.tfrecord_files) - 10} more" if len(self.tfrecord_files) > 10 else "")
                + "\n\n"
                "Possible causes:\n"
                "  1. TFRecord files are corrupted or empty\n"
                "  2. TFRecord format doesn't match expected schema\n"
                "  3. Data collection was interrupted before saving\n"
                "\n"
                "Expected TFRecord features:\n"
                "  - action: float32[8]\n"
                "  - observation/proprio: float32[8]\n"
                "  - observation/image_primary: bytes (JPEG)\n"
                "========================================\n"
            )

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
        with stats_path.open("w") as f:
            json.dump(full_stats, f, indent=2)
        print(f"Saved dataset statistics to {stats_path}")

        return full_stats

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action using BOUNDS_Q99 (matches OpenVLA).

        Maps [q01, q99] -> [-1, 1] and clips to [-1, 1].

        Args:
            action: Action array with shape (8,)

        Returns:
            Normalized action array
        """
        # Get stats for this dataset (dataset_statistics is nested: {dataset_name: {action: {...}}})
        stats = self.dataset_statistics[self.data_mix]

        q01 = np.array(stats["action"]["q01"], dtype=np.float32)
        q99 = np.array(stats["action"]["q99"], dtype=np.float32)

        # Normalize to [-1, 1]
        normalized = 2.0 * (action - q01) / (q99 - q01 + 1e-8) - 1.0

        # Clip to [-1, 1]
        normalized = np.clip(normalized, -1.0, 1.0)

        # Map unused dimensions (where min == max) to 0
        action_min = np.array(stats["action"]["min"], dtype=np.float32)
        action_max = np.array(stats["action"]["max"], dtype=np.float32)
        zeros_mask = action_min == action_max
        normalized = np.where(zeros_mask, 0.0, normalized)

        return normalized

    def _decode_and_resize_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode JPEG image and resize to target resolution."""
        image = decode_jpeg(image_bytes, channels=3)
        image = resize_image(image, self.resize_resolution)
        return image

    def _process_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Process a single example: decode image, normalize action, apply augmentation."""
        # Decode and resize primary image
        image_bytes = example.get("observation/image_primary", b"")
        image_primary = self._decode_and_resize_image(image_bytes)

        # Apply image augmentation if enabled
        if self.image_aug:
            image_primary = self._augmentor(image_primary)

        # Get action and normalize
        action = example.get("action")
        if action is not None:
            action = np.array(action, dtype=np.float32).reshape(8)
            action = self._normalize_action(action)
        else:
            action = np.zeros(8, dtype=np.float32)

        # Get proprio
        proprio = example.get("observation/proprio")
        if proprio is not None:
            proprio = np.array(proprio, dtype=np.float32).reshape(8)
        else:
            proprio = np.zeros(8, dtype=np.float32)

        # Get language instruction
        lang = example.get("task/language_instruction", b"manipulate the object")
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8", errors="replace")

        # Get dataset name
        dataset_name = example.get("dataset_name", b"crane_x7")
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8", errors="replace")

        # Restructure to RLDS batch format (with window_size=1, so add batch dim)
        observation = {
            "image_primary": np.expand_dims(image_primary, 0),  # [1, H, W, 3]
            "proprio": np.expand_dims(proprio, 0),  # [1, 8]
        }

        return {
            "observation": observation,
            "task": {
                "language_instruction": lang,
            },
            "action": np.expand_dims(action, 0),  # [1, 8]
            "dataset_name": dataset_name,
        }

    def _should_include_step(self, step_index: int) -> bool:
        """
        Determine if a step should be included based on split type.

        Uses a deterministic hash-based approach to split steps within episodes.
        This ensures reproducible splits regardless of data order.

        Args:
            step_index: Global step index in the dataset

        Returns:
            Boolean indicating if this step should be included
        """
        if self.overfit_split_ratio <= 0:
            return True

        # Use deterministic hash for splitting
        hash_value = step_index ^ self.split_seed
        mod_value = abs(hash_value) % 1000
        threshold = int(self.overfit_split_ratio * 1000)

        if self.split == "overfit":
            return mod_value < threshold
        else:
            return mod_value >= threshold

    def _create_tf_dataset(self):
        """Initialize dataset components (no longer creates TF dataset)."""
        # Initialize image augmentor
        self._augmentor = ImageAugmentor(
            target_size=self.resize_resolution,
            config=ImageAugmentationConfig(),
        )

        # Shard files for distributed training
        files_to_use = self.tfrecord_files
        if self.world_size > 1:
            files_to_use = [f for i, f in enumerate(self.tfrecord_files) if i % self.world_size == self.rank]

        # Store for iteration
        self._files_to_use = files_to_use
        return None  # No TF dataset

    def _generate_samples(self):
        """Generate samples from TFRecord files."""
        # Read all examples with their indices
        samples = []
        reader = TFRecordReader(
            self._files_to_use,
            feature_spec=self.FEATURE_DESCRIPTION,
            use_alternative_keys=True,
        )

        for step_index, example in enumerate(reader):
            if self._should_include_step(step_index):
                samples.append((step_index, example))

        # Shuffle if training
        if self.train:
            np.random.shuffle(samples)

        # Process and yield samples
        for _, example in samples:
            try:
                processed = self._process_example(example)
                yield processed
            except Exception:
                # Skip problematic samples
                continue

    def __iter__(self):
        """Iterate over dataset yielding transformed batches."""
        while True:
            for rlds_batch in self._generate_samples():
                yield self.batch_transform(rlds_batch)

            # If not training, don't repeat
            if not self.train:
                break

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
