# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Dataset for OpenVLA Training.

This dataset loader is specifically designed for CRANE-X7 robot data,
which uses 7-axis joint angles + gripper (8 dimensions total).
The dataset follows OpenVLA's RLDS format and uses joint angle actions directly,
similar to other datasets in the Open X-Embodiment mixture (e.g., Berkeley Cable Routing).
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Whether to normalize actions to [-1, 1]"""

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


class CraneX7BatchTransform:
    """
    Batch transform for CRANE-X7 data.

    Converts CRANE-X7 TFRecord format to OpenVLA expected format,
    using joint angles directly instead of end-effector positions.
    """

    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: type[PromptBuilder],
        config: CraneX7DatasetConfig,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        predict_stop_token: bool = True,
    ):
        """
        Initialize batch transform.

        Args:
            action_tokenizer: Action tokenizer for OpenVLA
            base_tokenizer: Base language tokenizer
            image_transform: Image preprocessing transform
            prompt_builder_fn: Prompt builder class
            config: Dataset configuration
            normalization_stats: Normalization statistics (mean, std, min, max)
            predict_stop_token: Whether to predict stop token
        """
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.config = config
        self.normalization_stats = normalization_stats or {}
        self.predict_stop_token = predict_stop_token

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to [-1, 1] range.

        Uses quantile-based normalization (q01, q99) for robustness.
        """
        if not self.config.normalize_actions or self.normalization_stats is None:
            return action

        if "action" not in self.normalization_stats:
            return action

        stats = self.normalization_stats["action"]

        # Use quantile bounds for normalization (more robust to outliers)
        if "q01" in stats and "q99" in stats:
            q_low = stats["q01"]
            q_high = stats["q99"]
            # Normalize to [-1, 1]
            action_normalized = 2.0 * (action - q_low) / (q_high - q_low + 1e-8) - 1.0
            # Clip to [-1, 1] range
            action_normalized = np.clip(action_normalized, -1.0, 1.0)
            return action_normalized
        elif "min" in stats and "max" in stats:
            # Fall back to min-max normalization
            action_min = stats["min"]
            action_max = stats["max"]
            action_normalized = 2.0 * (action - action_min) / (action_max - action_min + 1e-8) - 1.0
            action_normalized = np.clip(action_normalized, -1.0, 1.0)
            return action_normalized
        else:
            return action

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert CRANE-X7 batch to OpenVLA format.

        Args:
            batch: Batch from CRANE-X7 TFRecord
                - action: [8] joint angles + gripper
                - observation/proprio: [8] joint angles + gripper
                - observation/image_primary: JPEG bytes
                - task/language_instruction: string
                - dataset_name: string

        Returns:
            OpenVLA-formatted batch with:
                - pixel_values: image tensor
                - input_ids: tokenized prompt
                - labels: tokenized actions
                - dataset_name: dataset identifier
        """
        # Extract data from batch
        action = batch["action"]
        image_bytes = batch["observation"]["image_primary"]
        lang_bytes = batch["task"]["language_instruction"]
        dataset_name = batch.get("dataset_name", b"crane_x7")

        # Decode language instruction
        if isinstance(lang_bytes, bytes):
            lang = lang_bytes.decode("utf-8").lower().strip()
        else:
            lang = str(lang_bytes).lower().strip()

        if not lang:
            lang = self.config.default_instruction

        # Decode dataset name
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode("utf-8")

        # Decode and preprocess image
        if isinstance(image_bytes, bytes):
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Assume it's already a numpy array
            image = Image.fromarray(image_bytes.astype(np.uint8))

        # Normalize action
        action_normalized = self.normalize_action(action)

        # Construct prompt (OpenVLA format)
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action_normalized)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(),
            add_special_tokens=True
        ).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] Only compute loss for action tokens
        labels[:-(len(action_normalized) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
        )


class CraneX7Dataset(IterableDataset):
    """
    PyTorch IterableDataset for CRANE-X7 robot data.

    Loads TFRecord files containing joint angle trajectories and converts
    them to the format expected by OpenVLA training pipeline.
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
        config: CraneX7DatasetConfig,
        batch_transform: CraneX7BatchTransform,
        train: bool = True,
        shuffle_buffer_size: int = 1000,
    ):
        """
        Initialize CRANE-X7 dataset.

        Args:
            config: Dataset configuration
            batch_transform: Batch transformation function
            train: Whether this is training set
            shuffle_buffer_size: Buffer size for shuffling
        """
        super().__init__()
        self.config = config
        self.batch_transform = batch_transform
        self.train = train
        self.shuffle_buffer_size = shuffle_buffer_size

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()
        print(f"Found {len(self.tfrecord_files)} TFRecord files in {self.config.data_root}")

        # Load normalization statistics if not provided
        if batch_transform.normalization_stats is None:
            stats_path = self.config.normalization_stats_path
            if stats_path is None:
                # Try to find dataset_statistics.json in data root
                stats_path = self.config.data_root / "dataset_statistics.json"

            if stats_path and stats_path.exists():
                import json
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                    # Convert to numpy arrays
                    for key in stats:
                        if isinstance(stats[key], dict):
                            for subkey in stats[key]:
                                if isinstance(stats[key][subkey], list):
                                    stats[key][subkey] = np.array(stats[key][subkey])
                    batch_transform.normalization_stats = stats
                    print(f"Loaded normalization statistics from {stats_path}")

        # Create TensorFlow dataset
        self.dataset = self._create_tf_dataset()

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files in the data directory."""
        data_root = Path(self.config.data_root)

        # Look for episode directories containing tfrecord files
        tfrecord_files = list(data_root.glob("episode_*/episode_data.tfrecord"))

        # Also try to find tfrecord files directly
        if not tfrecord_files:
            tfrecord_files = list(data_root.glob("**/*.tfrecord"))

        # Filter out .bak files
        tfrecord_files = [f for f in tfrecord_files if not f.name.endswith(".bak")]

        return sorted(tfrecord_files)

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
            if "task/language_instruction" not in parsed:
                parsed["task/language_instruction"] = tf.constant(
                    self.config.default_instruction.encode("utf-8")
                )
            if "dataset_name" not in parsed:
                parsed["dataset_name"] = tf.constant(b"crane_x7")
            if "observation/timestep" not in parsed:
                parsed["observation/timestep"] = tf.constant([0], dtype=tf.int64)

        return parsed

    def _restructure(self, example):
        """Restructure example to match expected format."""
        return {
            "observation": {
                "image_primary": example["observation/image_primary"],
                "proprio": example["observation/proprio"],
                "timestep": example["observation/timestep"][0],
            },
            "task": {
                "language_instruction": example["task/language_instruction"],
            },
            "action": example["action"],
            "dataset_name": example["dataset_name"],
        }

    def _create_tf_dataset(self) -> tf.data.Dataset:
        """Create TensorFlow dataset pipeline."""
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(
            [str(f) for f in self.tfrecord_files],
            num_parallel_reads=tf.data.AUTOTUNE,
        )

        # Parse examples
        dataset = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Restructure to expected format
        dataset = dataset.map(
            self._restructure,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Shuffle if training
        if self.train:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # Repeat for training
        if self.train:
            dataset = dataset.repeat()

        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __iter__(self):
        """Iterate over dataset."""
        for example in self.dataset.as_numpy_iterator():
            yield self.batch_transform(example)

    def __len__(self) -> int:
        """Get approximate dataset length."""
        # This is approximate - count total steps across all files
        if not hasattr(self, "_dataset_length"):
            self._dataset_length = 0
            for tfrecord_file in self.tfrecord_files:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                self._dataset_length += sum(1 for _ in dataset)
        return self._dataset_length
