#!/usr/bin/env python3
"""
Dataset loader for CRANE-X7 TFRecord data.

This module provides a PyTorch Dataset that loads CRANE-X7 robot demonstration
data stored in TFRecord format and prepares it for OpenVLA fine-tuning.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CraneX7Dataset(Dataset):
    """
    PyTorch Dataset for loading CRANE-X7 TFRecord episodes.

    The dataset expects the following structure:
    data_root/
        episode_0000_TIMESTAMP/
            episode_data.tfrecord
        episode_0001_TIMESTAMP/
            episode_data.tfrecord
        ...

    Each TFRecord contains (RLDS format):
        - observation/proprio: Joint positions (8-DOF: 7 arm joints + 1 gripper)
        - observation/image_primary: RGB image (optional)
        - observation/depth_primary: Depth image (optional)
        - observation/timestep: Integer timestep
        - action: Next state (8-DOF: 7 arm joints + 1 gripper)
        - dataset_name: Dataset identifier
        - task/language_instruction: Language instruction
    """

    def __init__(
        self,
        data_root: str,
        instruction: str = "Pick and place the object",
        image_size: Tuple[int, int] = (224, 224),
        use_image: bool = True,
    ):
        """
        Initialize CRANE-X7 dataset.

        Args:
            data_root: Root directory containing episode folders
            instruction: Task instruction for VLA conditioning
            image_size: Target image size (H, W)
            use_image: Whether to load and use images
        """
        self.data_root = Path(data_root)
        self.instruction = instruction
        self.image_size = image_size
        self.use_image = use_image

        # Find all episode directories
        self.episode_dirs = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir() and d.name.startswith('episode_')
        ])

        if len(self.episode_dirs) == 0:
            raise ValueError(f"No episode directories found in {data_root}")

        # Load all episodes and create flat list of steps
        self.samples = []
        self._load_all_episodes()

        print(f"Loaded {len(self.episode_dirs)} episodes with {len(self.samples)} total steps")

    def _load_all_episodes(self):
        """Load all episodes and create a flat list of (episode_path, step_index) tuples."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required to read TFRecord files. "
                "Install with: pip install tensorflow"
            )

        for episode_dir in self.episode_dirs:
            tfrecord_path = episode_dir / 'episode_data.tfrecord'
            if not tfrecord_path.exists():
                print(f"Warning: No tfrecord file found in {episode_dir}")
                continue

            # Count steps in this episode
            step_count = 0
            for _ in tf.data.TFRecordDataset(str(tfrecord_path)):
                step_count += 1

            # Add all steps from this episode (excluding last step which has no valid action)
            for step_idx in range(step_count - 1):
                self.samples.append((tfrecord_path, step_idx))

    def _parse_tfrecord_step(self, example_bytes: bytes) -> Dict:
        """Parse a single TFRecord example in RLDS format."""
        import tensorflow as tf

        feature_description = {
            'observation/proprio': tf.io.FixedLenFeature([8], tf.float32),
            'action': tf.io.FixedLenFeature([8], tf.float32),
            'observation/timestep': tf.io.FixedLenFeature([1], tf.int64),
            'dataset_name': tf.io.FixedLenFeature([], tf.string),
            'task/language_instruction': tf.io.FixedLenFeature([], tf.string),
        }

        if self.use_image:
            feature_description.update({
                'observation/image_primary': tf.io.FixedLenFeature([], tf.string),
            })
            # Optional depth
            feature_description.update({
                'observation/depth_primary': tf.io.FixedLenFeature([], tf.string, default_value=''),
            })

        example = tf.io.parse_single_example(example_bytes, feature_description)

        parsed = {
            'state': example['observation/proprio'].numpy(),
            'action': example['action'].numpy(),
            'timestep': example['observation/timestep'].numpy()[0],
            'language_instruction': example['task/language_instruction'].numpy().decode('utf-8'),
        }

        if self.use_image:
            # Decode JPEG image
            import cv2
            img_bytes = example['observation/image_primary'].numpy()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            parsed['image'] = img

            # Decode depth if present
            depth_bytes = example['observation/depth_primary'].numpy()
            if len(depth_bytes) > 0:
                depth = np.frombuffer(depth_bytes, dtype=np.float32)
                # Reshape depth if needed (assuming same dimensions as RGB image)
                if img is not None:
                    depth = depth.reshape(img.shape[:2])
                parsed['depth'] = depth

        return parsed

    def _load_step(self, tfrecord_path: Path, step_idx: int) -> Dict:
        """Load a specific step from a TFRecord file."""
        import tensorflow as tf

        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        # Skip to the desired step
        for idx, raw_record in enumerate(dataset):
            if idx == step_idx:
                return self._parse_tfrecord_step(raw_record.numpy())

        raise IndexError(f"Step {step_idx} not found in {tfrecord_path}")

    def __len__(self) -> int:
        """Return total number of steps across all episodes."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary containing:
                - image: RGB image tensor [C, H, W] (if use_image=True)
                - depth: Depth tensor [H, W] (if present)
                - state: Joint state tensor [8]
                - action: Action tensor [8]
                - instruction: Task instruction string from RLDS data
                - timestep: Integer timestep
        """
        tfrecord_path, step_idx = self.samples[idx]
        step_data = self._load_step(tfrecord_path, step_idx)

        sample = {
            'state': torch.from_numpy(step_data['state']).float(),
            'action': torch.from_numpy(step_data['action']).float(),
            'instruction': step_data.get('language_instruction', self.instruction),
            'timestep': step_data['timestep'],
        }

        if self.use_image and 'image' in step_data:
            # Convert image to PIL, resize, and convert to tensor
            img = Image.fromarray(step_data['image'])
            img = img.resize(self.image_size, Image.BILINEAR)
            img_array = np.array(img)
            # Convert to [C, H, W] and normalize to [0, 1]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            sample['image'] = img_tensor

        if 'depth' in step_data:
            # Convert depth to tensor
            depth_tensor = torch.from_numpy(step_data['depth']).float()
            sample['depth'] = depth_tensor

        return sample


def create_crane_x7_dataloader(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    instruction: str = "Pick and place the object",
    image_size: Tuple[int, int] = (224, 224),
    use_image: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for CRANE-X7 dataset.

    Args:
        data_root: Root directory containing episode folders
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the dataset
        instruction: Task instruction
        image_size: Target image size
        use_image: Whether to use images

    Returns:
        DataLoader instance
    """
    dataset = CraneX7Dataset(
        data_root=data_root,
        instruction=instruction,
        image_size=image_size,
        use_image=use_image,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == '__main__':
    # Test dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python crane_x7_dataset.py <data_root>")
        sys.exit(1)

    data_root = sys.argv[1]

    print(f"Loading dataset from {data_root}")
    dataset = CraneX7Dataset(data_root)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of episodes: {len(dataset.episode_dirs)}")

    # Load first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nFirst sample:")
        print(f"  State shape: {sample['state'].shape}")
        print(f"  Action shape: {sample['action'].shape}")
        print(f"  Timestep: {sample['timestep']}")
        print(f"  Instruction: {sample['instruction']}")
        if 'image' in sample:
            print(f"  Image shape: {sample['image'].shape}")
        if 'depth' in sample:
            print(f"  Depth shape: {sample['depth'].shape}")
