"""
OpenPI backend implementation.

Integrates OpenPI training pipeline with the unified VLA backend interface.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Add OpenPI source path
openpi_path = Path(__file__).parent.parent.parent / "openpi"
sys.path.insert(0, str(openpi_path))

from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.config.openpi_config import OpenPIConfig
from crane_x7_vla.data.adapters import CraneX7DataAdapter
from crane_x7_vla.data.converters import TFRecordToLeRobotConverter, LeRobotDataset

try:
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
    import openpi.training.config as openpi_config
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False
    logger.warning("OpenPI modules not available. OpenPI backend will not work.")


class OpenPIBackend(VLABackend):
    """
    OpenPI backend implementation.

    Integrates OpenPI's training pipeline for CRANE-X7 data.
    """

    def __init__(self, config: OpenPIConfig):
        """
        Initialize OpenPI backend.

        Args:
            config: OpenPI configuration
        """
        if not OPENPI_AVAILABLE:
            raise ImportError(
                "OpenPI modules not available. Please ensure OpenPI is properly installed."
            )

        super().__init__(config)
        self.openpi_config = config
        self._action_dim = config.model_action_dim
        self._action_horizon = config.action_horizon
        self._image_size = config.openpi.image_size

        # Data converter
        self.converter = TFRecordToLeRobotConverter(
            source_action_dim=config.crane_x7_action_dim,
            target_action_dim=config.model_action_dim,
            action_horizon=config.action_horizon,
            camera_names=config.openpi.camera_names,
            image_size=config.openpi.image_size,
            chunk_interpolation=config.openpi.chunk_interpolation,
            normalize_actions=config.openpi.normalize_actions,
            normalization_mode=config.openpi.normalization_mode,
        )

        self.device = None
        self.data_loader = None

    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")

    def _prepare_data(self):
        """
        Prepare dataset for training.

        Loads TFRecord data, converts to LeRobot format, and creates DataLoader.
        """
        logger.info("Preparing data...")

        # Create data adapter
        data_adapter = CraneX7DataAdapter(
            data_root=self.openpi_config.data.data_root,
            split="train",
            shuffle=self.openpi_config.data.shuffle,
            buffer_size=1000,
            include_depth=self.openpi_config.openpi.use_depth
        )

        # Fit normalizers on dataset
        logger.info("Fitting normalizers...")
        self.converter.fit_normalizers(data_adapter)

        # Save normalization stats
        stats_dir = self.openpi_config.output_dir / "normalization_stats"
        self.converter.save_normalization_stats(stats_dir)
        logger.info(f"Saved normalization stats to {stats_dir}")

        # Create PyTorch dataset
        dataset = LeRobotDataset(
            data_adapter=data_adapter,
            converter=self.converter,
            cache_episodes=True  # Cache for faster training
        )

        # Create DataLoader
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.openpi_config.training.batch_size,
            shuffle=self.openpi_config.data.shuffle,
            num_workers=self.openpi_config.data.num_workers,
            pin_memory=True,
        )

        logger.info(f"Dataset prepared: {len(dataset)} episodes")

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.

        Returns:
            Dictionary containing training metrics and results
        """
        self._setup_device()
        self._prepare_data()

        logger.info("Starting OpenPI training...")
        logger.info(f"Model type: {self.openpi_config.openpi.model_type}")
        logger.info(f"Action dim: {self._action_dim}")
        logger.info(f"Action horizon: {self._action_horizon}")
        logger.info(f"Batch size: {self.openpi_config.training.batch_size}")

        # TODO: Implement full OpenPI training pipeline
        # This requires:
        # 1. Creating OpenPI model config
        # 2. Initializing PI0 model
        # 3. Setting up optimizer
        # 4. Running training loop with action chunking
        # 5. Handling multi-camera inputs

        raise NotImplementedError(
            "Full OpenPI training pipeline is complex and requires additional integration work. "
            "Key steps needed:\n"
            "1. Convert UnifiedVLAConfig to OpenPI's TrainConfig format\n"
            "2. Initialize PI0Pytorch model with correct architecture\n"
            "3. Implement custom training loop for action chunking\n"
            "4. Handle multi-camera image inputs\n"
            "5. Integrate with OpenPI's checkpoint/logging system\n\n"
            "For now, please use OpenPI's native training script or contribute to complete this integration."
        )

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Path to test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError("Evaluation not yet implemented for OpenPI backend")

    def infer(
        self,
        observation: Dict[str, np.ndarray],
        language_instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform inference on a single observation.

        Args:
            observation: Dictionary containing:
                - 'state': Robot state [8]
                - 'image': RGB image(s) (dict or single array)
            language_instruction: Task instruction

        Returns:
            Predicted action chunk as numpy array [action_horizon, 8]
            Note: Returns full chunk but typically only first action is executed
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")

        # Prepare observation
        # Convert to LeRobot format
        state = observation.get('state')
        images = observation.get('image')

        # Handle single image or multi-camera
        if isinstance(images, np.ndarray):
            # Single camera - map to primary camera
            camera_images = {self.openpi_config.openpi.camera_names[0]: images}
        else:
            camera_images = images

        # Process through converter
        # This pads state/action, processes images, etc.
        # TODO: Implement single-step conversion (not full episode)

        raise NotImplementedError(
            "Inference not yet implemented for OpenPI backend. "
            "Requires converting single observation to model input format."
        )

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save")

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.openpi_config.backend_config,
        }, path / "model.pt")

        # Save normalization stats
        if self.converter is not None:
            self.converter.save_normalization_stats(path)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint directory
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Checkpoint path does not exist: {path}")

        logger.info(f"Loading OpenPI model from {path}")

        # Load normalization stats
        if (path / "action_norm_stats.npz").exists():
            self.converter.load_normalization_stats(path)
            logger.info("Loaded normalization stats")

        # TODO: Load OpenPI model
        # This requires:
        # 1. Reconstructing model config from saved config
        # 2. Initializing PI0 model
        # 3. Loading state dict

        raise NotImplementedError(
            "Checkpoint loading not yet implemented for OpenPI backend. "
            "Requires OpenPI model initialization and state loading."
        )

    @property
    def action_dim(self) -> int:
        """Get the action dimension (CRANE-X7 native dimension)."""
        return self.openpi_config.crane_x7_action_dim

    @property
    def action_horizon(self) -> int:
        """Get the action horizon (OpenPI uses action chunking)."""
        return self._action_horizon

    @property
    def expected_image_size(self) -> tuple:
        """Get the expected image size for the model."""
        return self._image_size
