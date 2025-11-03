"""
Data format converters.

Converts between different VLA data formats (TFRecord, LeRobot, etc.).
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import logging

from crane_x7_vla.data.adapters import CraneX7DataAdapter
from crane_x7_vla.transforms.action_transforms import (
    ActionPadder, ActionChunker, ActionNormalizer
)
from crane_x7_vla.transforms.image_transforms import MultiCameraProcessor
from crane_x7_vla.transforms.state_transforms import StateNormalizer, StatePadder

logger = logging.getLogger(__name__)


class TFRecordToLeRobotConverter:
    """
    Converts CRANE-X7 TFRecord data to LeRobot format for OpenPI.

    LeRobot format expected by OpenPI:
    {
        'observation/state': [state_dim] float32,
        'observation/image': {cam_name: [H, W, 3] uint8, ...},
        'observation/image_mask': {cam_name: bool, ...},
        'actions': [action_horizon, action_dim] float32,
        'prompt': string,
    }
    """

    def __init__(
        self,
        source_action_dim: int = 8,
        target_action_dim: int = 32,
        action_horizon: int = 50,
        camera_names: List[str] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        image_size: Tuple[int, int] = (224, 224),
        chunk_interpolation: Literal["repeat", "linear"] = "linear",
        normalize_actions: bool = True,
        normalization_mode: Literal["quantile", "zscore"] = "quantile",
    ):
        """
        Initialize converter.

        Args:
            source_action_dim: Source action dimension (CRANE-X7 = 8)
            target_action_dim: Target action dimension (OpenPI = 32)
            action_horizon: Number of future actions to predict
            camera_names: Camera names following LeRobot convention
            image_size: Target image size
            chunk_interpolation: How to generate action chunks
            normalize_actions: Whether to normalize actions
            normalization_mode: Normalization mode
        """
        self.source_action_dim = source_action_dim
        self.target_action_dim = target_action_dim
        self.action_horizon = action_horizon
        self.camera_names = camera_names
        self.image_size = image_size

        # Initialize transformers
        self.action_padder = ActionPadder(source_action_dim, target_action_dim)
        self.state_padder = StatePadder(source_action_dim, target_action_dim)
        self.action_chunker = ActionChunker(action_horizon, chunk_interpolation)
        self.multi_camera_processor = MultiCameraProcessor(
            camera_names, image_size, pad_missing=True
        )

        # Normalizers (will be fitted later)
        self.normalize_actions = normalize_actions
        self.action_normalizer = ActionNormalizer(normalization_mode) if normalize_actions else None
        self.state_normalizer = StateNormalizer(normalization_mode)

    def fit_normalizers(self, data_adapter: CraneX7DataAdapter) -> None:
        """
        Fit normalizers on the dataset.

        Args:
            data_adapter: Data adapter for the dataset
        """
        logger.info("Fitting normalizers on dataset...")

        # Collect actions and states
        actions = []
        states = []

        for episode in data_adapter.iterate_episodes():
            if 'action' in episode:
                actions.append(episode['action'])
            if 'observation/state' in episode:
                states.append(episode['observation/state'])

        actions = np.array(actions)
        states = np.array(states)

        # Fit action normalizer
        if self.normalize_actions and self.action_normalizer is not None:
            self.action_normalizer.fit(actions)
            logger.info("Fitted action normalizer")

        # Fit state normalizer
        self.state_normalizer.fit(states)
        logger.info("Fitted state normalizer")

    def convert_episode(
        self,
        episode: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert a single episode from TFRecord format to LeRobot format.

        Args:
            episode: Episode in TFRecord format

        Returns:
            Episode in LeRobot format
        """
        # Extract components
        state = episode.get('observation/state')  # [8]
        action = episode.get('action')  # [8]
        image = episode.get('observation/image')  # [H, W, 3]
        prompt = episode.get('prompt', b'').decode() if isinstance(
            episode.get('prompt', ''), bytes
        ) else episode.get('prompt', '')

        # Pad state and action to target dimension
        state_padded = self.state_padder.pad(state)  # [32]
        action_padded = self.action_padder.pad(action)  # [32]

        # Normalize state (always)
        state_normalized = self.state_normalizer.normalize(state_padded)

        # Normalize action (optional)
        if self.normalize_actions and self.action_normalizer is not None:
            action_normalized = self.action_normalizer.normalize(action_padded)
        else:
            action_normalized = action_padded

        # Create action chunk (note: we only have single action, so we'll repeat/interpolate)
        # This is a limitation - ideally we'd have access to the next action
        action_chunk = self.action_chunker.chunk_single_action(action_normalized)  # [50, 32]

        # Process image (map to multiple cameras)
        # We only have one camera, so we'll use it for the primary camera and pad others
        camera_images = {self.camera_names[0]: image}  # Use first camera as primary
        processed_images, image_masks = self.multi_camera_processor.process(camera_images)

        # Build LeRobot format
        lerobot_episode = {
            'observation/state': state_normalized.astype(np.float32),
            'observation/image': processed_images,  # Dict of images
            'observation/image_mask': image_masks,  # Dict of masks
            'actions': action_chunk.astype(np.float32),  # [50, 32]
            'prompt': prompt,
        }

        return lerobot_episode

    def convert_batch(
        self,
        episodes: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Convert a batch of episodes.

        Args:
            episodes: List of episodes in TFRecord format

        Returns:
            Batch in LeRobot format with batched tensors
        """
        converted = [self.convert_episode(ep) for ep in episodes]

        # Stack into batches
        batch = {
            'observation/state': np.stack([ep['observation/state'] for ep in converted]),
            'observation/image': {
                cam: np.stack([ep['observation/image'][cam] for ep in converted])
                for cam in self.camera_names
            },
            'observation/image_mask': {
                cam: np.stack([ep['observation/image_mask'][cam] for ep in converted])
                for cam in self.camera_names
            },
            'actions': np.stack([ep['actions'] for ep in converted]),
            'prompt': [ep['prompt'] for ep in converted],
        }

        return batch

    def save_normalization_stats(self, output_dir: Path) -> None:
        """
        Save normalization statistics.

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.action_normalizer is not None:
            self.action_normalizer.save_stats(output_dir / "action_norm_stats.npz")
            logger.info(f"Saved action normalization stats to {output_dir / 'action_norm_stats.npz'}")

        self.state_normalizer.save_stats(output_dir / "state_norm_stats.npz")
        logger.info(f"Saved state normalization stats to {output_dir / 'state_norm_stats.npz'}")

    def load_normalization_stats(self, stats_dir: Path) -> None:
        """
        Load normalization statistics.

        Args:
            stats_dir: Directory containing normalization stats
        """
        stats_dir = Path(stats_dir)

        if self.action_normalizer is not None:
            action_stats_path = stats_dir / "action_norm_stats.npz"
            if action_stats_path.exists():
                self.action_normalizer.load_stats(str(action_stats_path))
                logger.info(f"Loaded action normalization stats from {action_stats_path}")

        state_stats_path = stats_dir / "state_norm_stats.npz"
        if state_stats_path.exists():
            self.state_normalizer.load_stats(str(state_stats_path))
            logger.info(f"Loaded state normalization stats from {state_stats_path}")


class LeRobotDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for LeRobot-formatted data.

    Converts TFRecord data on-the-fly to LeRobot format.
    """

    def __init__(
        self,
        data_adapter: CraneX7DataAdapter,
        converter: TFRecordToLeRobotConverter,
        cache_episodes: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_adapter: Data adapter for loading TFRecord data
            converter: Converter to LeRobot format
            cache_episodes: Whether to cache all episodes in memory
        """
        self.data_adapter = data_adapter
        self.converter = converter
        self.cache_episodes = cache_episodes

        # Load all episodes into memory if caching
        if self.cache_episodes:
            logger.info("Caching all episodes in memory...")
            self.episodes = list(self.data_adapter.iterate_episodes())
            logger.info(f"Cached {len(self.episodes)} episodes")
        else:
            # Count episodes without loading
            logger.info("Counting episodes...")
            self.num_episodes = len(self.data_adapter)
            logger.info(f"Found {self.num_episodes} episodes")

    def __len__(self) -> int:
        """Get dataset size."""
        if self.cache_episodes:
            return len(self.episodes)
        else:
            return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.

        Args:
            idx: Episode index

        Returns:
            Episode in LeRobot format with PyTorch tensors
        """
        # Get episode
        if self.cache_episodes:
            episode = self.episodes[idx]
        else:
            # This is inefficient - we iterate from the beginning each time
            # A better approach would be to implement indexed access or use caching
            for i, ep in enumerate(self.data_adapter.iterate_episodes()):
                if i == idx:
                    episode = ep
                    break

        # Convert to LeRobot format
        lerobot_episode = self.converter.convert_episode(episode)

        # Convert to PyTorch tensors
        torch_episode = {
            'observation/state': torch.from_numpy(lerobot_episode['observation/state']),
            'observation/image': {
                cam: torch.from_numpy(img)
                for cam, img in lerobot_episode['observation/image'].items()
            },
            'observation/image_mask': {
                cam: torch.tensor(mask)
                for cam, mask in lerobot_episode['observation/image_mask'].items()
            },
            'actions': torch.from_numpy(lerobot_episode['actions']),
            'prompt': lerobot_episode['prompt'],
        }

        return torch_episode
