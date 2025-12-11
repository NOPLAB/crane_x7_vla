# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
CRANE-X7 Dataset for OpenVLA-OFT Training.

This dataset loader is designed for OpenVLA-OFT (Optimized Fine-Tuning) which:
- Returns continuous action chunks instead of tokenized single actions
- Supports proprioceptive (robot state) input
- Supports multiple camera images (e.g., third-person + wrist)

Key differences from standard OpenVLA dataset:
1. Action chunking: Returns (action_horizon, action_dim) instead of (action_dim,)
2. No action tokenization: Returns raw continuous actions for L1 regression
3. Proprio support: Includes normalized robot state
4. Multi-image: Supports primary + wrist camera images
"""

from dataclasses import dataclass
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


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class OpenVLAOFTBatchTransform:
    """
    Batch transform for OpenVLA-OFT training data.

    Key differences from standard OpenVLA transform:
    - Returns continuous action chunks (not tokenized)
    - Includes proprioceptive state
    - Supports multiple images
    - No action token generation
    """

    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    action_horizon: int = 8
    action_dim: int = 8
    include_proprio: bool = True
    include_wrist_image: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert RLDS batch to OpenVLA-OFT format.

        Args:
            rlds_batch: RLDS-formatted batch containing:
                - observation/image_primary: Primary camera image
                - observation/image_wrist: Wrist camera image (optional)
                - observation/proprio: Proprioceptive state
                - action_chunk: Future actions (action_horizon, action_dim)
                - task/language_instruction: Language instruction

        Returns:
            Dict with:
                - pixel_values: Transformed image(s)
                - input_ids: Tokenized prompt (no action tokens)
                - attention_mask: Attention mask
                - labels: Labels for prompt tokens (IGNORE_INDEX for all)
                - proprio: Proprioceptive state (optional)
                - action_chunk: Continuous actions (action_horizon, action_dim)
                - dataset_name: Dataset identifier
        """
        dataset_name = rlds_batch.get("dataset_name", b"crane_x7")
        if isinstance(dataset_name, bytes):
            dataset_name = dataset_name.decode()

        # Get language instruction
        lang = rlds_batch["task"]["language_instruction"]
        if isinstance(lang, bytes):
            lang = lang.decode()
        lang = lang.lower()

        # Process primary image
        img_primary = rlds_batch["observation"]["image_primary"]
        if img_primary.ndim == 4:  # [1, H, W, 3]
            img_primary = img_primary[0]
        img = Image.fromarray(img_primary)

        # Process wrist image if enabled
        wrist_img = None
        if self.include_wrist_image and "image_wrist" in rlds_batch["observation"]:
            img_wrist = rlds_batch["observation"]["image_wrist"]
            if img_wrist is not None and len(img_wrist) > 0:
                if img_wrist.ndim == 4:
                    img_wrist = img_wrist[0]
                wrist_img = Image.fromarray(img_wrist)

        # Build prompt (without action tokens)
        # For OFT, we only need the question part, not action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {lang}?")
        # Don't add GPT response - we predict continuous actions, not tokens
        prompt_text = prompt_builder.get_prompt()

        # Tokenize prompt
        tokenized = self.base_tokenizer(
            prompt_text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        # Labels: all IGNORE_INDEX since we use L1 loss on actions, not token loss
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Apply image transform
        pixel_values = self.image_transform(img)

        # If using wrist image, concatenate transformed images
        if wrist_img is not None:
            wrist_pixel_values = self.image_transform(wrist_img)
            # Concatenate along channel dimension for fused backbone
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = torch.cat([pixel_values, wrist_pixel_values], dim=0)

        # Get action chunk (continuous, not tokenized)
        action_chunk = rlds_batch["action_chunk"]  # (action_horizon, action_dim)
        if isinstance(action_chunk, np.ndarray):
            action_chunk = torch.from_numpy(action_chunk).float()

        # Get proprioceptive state
        proprio = None
        if self.include_proprio:
            proprio = rlds_batch["observation"]["proprio"]
            if proprio.ndim == 2:  # [1, 8]
                proprio = proprio[0]
            if isinstance(proprio, np.ndarray):
                proprio = torch.from_numpy(proprio).float()

        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": action_chunk,  # (action_horizon, action_dim) continuous
            "dataset_name": dataset_name,
        }

        if proprio is not None:
            result["proprio"] = proprio

        return result


class CraneX7OFTDataset(IterableDataset):
    """
    PyTorch IterableDataset for CRANE-X7 OpenVLA-OFT training.

    Key features:
    - Returns action chunks (action_horizon, action_dim) instead of single actions
    - Continuous actions (no tokenization)
    - Proprioceptive state support
    - Multi-image support (primary + wrist)

    The action chunking is done by looking ahead in the episode to get
    future actions. If not enough future actions exist, the last action
    is repeated to fill the chunk.
    """

    FEATURE_DESCRIPTION = {
        "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
        "observation/image_primary": tf.io.FixedLenFeature([], tf.string),
        "observation/timestep": tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        "action": tf.io.FixedLenFeature([8], tf.float32),
        "task/language_instruction": tf.io.FixedLenFeature(
            [], tf.string, default_value=b"manipulate the object"
        ),
        "dataset_name": tf.io.FixedLenFeature([], tf.string, default_value=b"crane_x7"),
        # Multi-camera support
        "observation/image_wrist": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: OpenVLAOFTBatchTransform,
        resize_resolution: Tuple[int, int],
        action_horizon: int = 8,
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
        Initialize OFT dataset.

        Args:
            data_root_dir: Root directory containing TFRecord episode files
            data_mix: Dataset name (typically "crane_x7")
            batch_transform: OpenVLAOFTBatchTransform instance
            resize_resolution: Target image resolution (height, width)
            action_horizon: Number of future actions to include in chunk
            shuffle_buffer_size: Buffer size for shuffling
            train: Whether this is training set
            image_aug: Whether to apply image augmentation
            overfit_split_ratio: Ratio of steps for overfitting detection
            split: Dataset split ("train" or "overfit")
            split_seed: Random seed for splitting
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.batch_transform = batch_transform
        self.resize_resolution = resize_resolution
        self.action_horizon = action_horizon
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.image_aug = image_aug
        self.overfit_split_ratio = overfit_split_ratio
        self.split = split
        self.split_seed = split_seed
        self.rank = rank
        self.world_size = world_size

        # Find TFRecord files
        self.tfrecord_files = self._find_tfrecord_files()

        print(
            f"[Rank {self.rank}/{self.world_size}] Found {len(self.tfrecord_files)} "
            f"TFRecord files for {split} split (OFT, action_horizon={action_horizon})"
        )

        # Load/compute dataset statistics
        self.dataset_statistics = self._get_dataset_statistics()

        # Store dataset length
        stats = self.dataset_statistics.get(self.data_mix, {})
        self.dataset_length = stats.get("num_transitions", 0)

        # Load all episodes into memory for action chunking
        # This is necessary because TFRecords are not random-access
        self._load_episodes()

    def _find_tfrecord_files(self) -> List[Path]:
        """Find all TFRecord files in the data directory."""
        tfrecord_files = list(self.data_root_dir.glob("episode_*/episode_data.tfrecord"))
        if not tfrecord_files:
            tfrecord_files = list(self.data_root_dir.glob("**/*.tfrecord"))
        tfrecord_files = [f for f in tfrecord_files if not f.name.endswith(".bak")]
        return sorted(tfrecord_files)

    def _get_dataset_statistics(self) -> Dict[str, Any]:
        """Load or compute dataset statistics."""
        import json

        if not self.tfrecord_files:
            raise FileNotFoundError(
                f"No TFRecord files found in {self.data_root_dir}"
            )

        stats_path = self.data_root_dir / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                full_stats = json.load(f)
                if self.data_mix in full_stats:
                    return full_stats
                elif "crane_x7" in full_stats:
                    return full_stats
                else:
                    return {self.data_mix: full_stats}

        # Compute statistics
        print("Computing dataset statistics...")
        actions = []
        proprios = []
        num_transitions = 0

        for tfrecord_file in self.tfrecord_files:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            for raw_record in dataset:
                example = tf.io.parse_single_example(
                    raw_record,
                    {
                        "action": tf.io.FixedLenFeature([8], tf.float32),
                        "observation/proprio": tf.io.FixedLenFeature([8], tf.float32),
                    },
                )
                actions.append(example["action"].numpy())
                proprios.append(example["observation/proprio"].numpy())
                num_transitions += 1

        if num_transitions == 0:
            raise ValueError(f"No valid data found in TFRecord files")

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

        full_stats = {self.data_mix: stats}

        with open(stats_path, "w") as f:
            json.dump(full_stats, f, indent=2)

        return full_stats

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action using BOUNDS_Q99."""
        stats = self.dataset_statistics[self.data_mix]
        q01 = np.array(stats["action"]["q01"], dtype=np.float32)
        q99 = np.array(stats["action"]["q99"], dtype=np.float32)

        normalized = 2.0 * (action - q01) / (q99 - q01 + 1e-8) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        # Map unused dimensions to 0
        action_min = np.array(stats["action"]["min"], dtype=np.float32)
        action_max = np.array(stats["action"]["max"], dtype=np.float32)
        zeros_mask = action_min == action_max
        normalized = np.where(zeros_mask, 0.0, normalized)

        return normalized

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Normalize proprioceptive state using BOUNDS_Q99."""
        stats = self.dataset_statistics[self.data_mix]
        q01 = np.array(stats["proprio"]["q01"], dtype=np.float32)
        q99 = np.array(stats["proprio"]["q99"], dtype=np.float32)

        normalized = 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def _load_episodes(self) -> None:
        """Load all episodes into memory for action chunking."""
        self.episodes = []

        for tfrecord_file in self.tfrecord_files:
            episode_steps = []
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))

            for raw_record in dataset:
                example = tf.io.parse_single_example(raw_record, self.FEATURE_DESCRIPTION)
                step = {
                    "action": example["action"].numpy(),
                    "proprio": example["observation/proprio"].numpy(),
                    "image_primary": example["observation/image_primary"].numpy(),
                    "image_wrist": example["observation/image_wrist"].numpy(),
                    "language_instruction": example["task/language_instruction"].numpy(),
                    "dataset_name": example["dataset_name"].numpy(),
                }
                episode_steps.append(step)

            if episode_steps:
                self.episodes.append(episode_steps)

        print(f"Loaded {len(self.episodes)} episodes with {sum(len(e) for e in self.episodes)} total steps")

    def _get_action_chunk(
        self, episode: List[Dict], step_idx: int
    ) -> np.ndarray:
        """
        Get action chunk starting from step_idx.

        If not enough future actions, repeat the last action.

        Args:
            episode: List of episode steps
            step_idx: Current step index

        Returns:
            action_chunk: (action_horizon, action_dim) normalized actions
        """
        actions = []
        for i in range(self.action_horizon):
            future_idx = step_idx + i
            if future_idx < len(episode):
                action = episode[future_idx]["action"]
            else:
                # Repeat last action if not enough future steps
                action = episode[-1]["action"]

            # Normalize action
            action = self._normalize_action(action)
            actions.append(action)

        return np.stack(actions, axis=0)  # (action_horizon, action_dim)

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode JPEG image and resize."""
        if not image_bytes:
            return None
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, self.resize_resolution)
        image = tf.cast(image, tf.uint8)
        return image.numpy()

    def _apply_image_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply image augmentation."""
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32) / 255.0

        # Random crop
        crop_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * 0.9, tf.int32)
        image = tf.image.random_crop(image, [crop_size[0], crop_size[1], 3])
        image = tf.image.resize(image, self.resize_resolution)

        # Color augmentations
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.05)

        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.cast(image * 255.0, tf.uint8)
        return image.numpy()

    def _should_include_step(self, episode_idx: int, step_idx: int) -> bool:
        """Determine if step should be included based on split."""
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

    def _generate_samples(self):
        """Generate samples with action chunking."""
        # Create list of all (episode_idx, step_idx) pairs
        all_indices = []
        for ep_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                if self._should_include_step(ep_idx, step_idx):
                    all_indices.append((ep_idx, step_idx))

        # Shard for distributed training
        if self.world_size > 1:
            all_indices = all_indices[self.rank :: self.world_size]

        # Shuffle if training
        if self.train:
            np.random.shuffle(all_indices)

        # Generate samples
        for ep_idx, step_idx in all_indices:
            episode = self.episodes[ep_idx]
            step = episode[step_idx]

            # Decode images
            image_primary = self._decode_image(step["image_primary"])
            if self.image_aug:
                image_primary = self._apply_image_augmentation(image_primary)

            # Decode wrist image if available
            image_wrist = None
            if step["image_wrist"]:
                image_wrist = self._decode_image(step["image_wrist"])
                if self.image_aug:
                    image_wrist = self._apply_image_augmentation(image_wrist)

            # Get action chunk
            action_chunk = self._get_action_chunk(episode, step_idx)

            # Normalize proprio
            proprio = self._normalize_proprio(step["proprio"])

            # Build RLDS-like batch
            rlds_batch = {
                "observation": {
                    "image_primary": np.expand_dims(image_primary, 0),
                    "proprio": np.expand_dims(proprio, 0),
                },
                "action_chunk": action_chunk,
                "task": {
                    "language_instruction": step["language_instruction"],
                },
                "dataset_name": step["dataset_name"],
            }

            if image_wrist is not None:
                rlds_batch["observation"]["image_wrist"] = np.expand_dims(image_wrist, 0)

            yield rlds_batch

    def __iter__(self):
        """Iterate over dataset samples."""
        for rlds_batch in self._generate_samples():
            yield self.batch_transform(rlds_batch)

        # Repeat if training
        if self.train:
            while True:
                for rlds_batch in self._generate_samples():
                    yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length


class PaddedCollatorForOFT:
    """
    Collator for OpenVLA-OFT batches.

    Handles padding of variable-length sequences and stacking of tensors.
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int,
        padding_side: str = "right",
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        # Stack pixel values
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # Pad input_ids and attention_mask
        max_len = max(item["input_ids"].shape[0] for item in batch)
        max_len = min(max_len, self.max_length)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            if seq_len > max_len:
                # Truncate
                input_ids_list.append(item["input_ids"][:max_len])
                attention_mask_list.append(item["attention_mask"][:max_len])
                labels_list.append(item["labels"][:max_len])
            else:
                # Pad
                pad_len = max_len - seq_len
                if self.padding_side == "right":
                    input_ids_list.append(
                        torch.cat([
                            item["input_ids"],
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        ])
                    )
                    attention_mask_list.append(
                        torch.cat([
                            item["attention_mask"],
                            torch.zeros(pad_len, dtype=torch.long),
                        ])
                    )
                    labels_list.append(
                        torch.cat([
                            item["labels"],
                            torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                        ])
                    )
                else:
                    input_ids_list.append(
                        torch.cat([
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                            item["input_ids"],
                        ])
                    )
                    attention_mask_list.append(
                        torch.cat([
                            torch.zeros(pad_len, dtype=torch.long),
                            item["attention_mask"],
                        ])
                    )
                    labels_list.append(
                        torch.cat([
                            torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                            item["labels"],
                        ])
                    )

        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)

        # Stack actions (action chunks)
        actions = torch.stack([item["actions"] for item in batch])

        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
        }

        # Stack proprio if present
        if "proprio" in batch[0]:
            proprio = torch.stack([item["proprio"] for item in batch])
            result["proprio"] = proprio

        return result
