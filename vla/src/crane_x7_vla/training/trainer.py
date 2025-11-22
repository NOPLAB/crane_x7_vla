# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Unified VLA trainer.

Provides a single interface for training different VLA backends.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from crane_x7_vla.config.base import UnifiedVLAConfig
from crane_x7_vla.config.openvla_config import OpenVLAConfig
from crane_x7_vla.config.openpi_config import OpenPIConfig
from crane_x7_vla.backends.base import VLABackend
from crane_x7_vla.backends.openvla import OpenVLABackend
from crane_x7_vla.backends.openpi import OpenPIBackend

logger = logging.getLogger(__name__)


class VLATrainer:
    """
    Unified trainer for VLA models.

    Automatically selects and configures the appropriate backend based on configuration.
    """

    def __init__(
        self,
        config: Union[UnifiedVLAConfig, OpenVLAConfig, OpenPIConfig]
    ):
        """
        Initialize VLA trainer.

        Args:
            config: Unified VLA configuration or backend-specific configuration
        """
        self.config = config
        self.backend: Optional[VLABackend] = None

        # Create backend
        self._create_backend()

    def _create_backend(self) -> None:
        """Create the appropriate backend based on configuration."""
        backend_type = self.config.backend

        logger.info(f"Creating {backend_type} backend...")

        if backend_type == "openvla":
            # Convert to OpenVLA config if needed
            if not isinstance(self.config, OpenVLAConfig):
                # Convert UnifiedVLAConfig to OpenVLAConfig
                openvla_config = self._convert_to_openvla_config(self.config)
            else:
                openvla_config = self.config

            self.backend = OpenVLABackend(openvla_config)

        elif backend_type == "openpi":
            # Convert to OpenPI config if needed
            if not isinstance(self.config, OpenPIConfig):
                # Convert UnifiedVLAConfig to OpenPIConfig
                openpi_config = self._convert_to_openpi_config(self.config)
            else:
                openpi_config = self.config

            self.backend = OpenPIBackend(openpi_config)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        logger.info(f"Backend created: {self.backend}")

    def _convert_to_openvla_config(self, config: UnifiedVLAConfig) -> OpenVLAConfig:
        """Convert UnifiedVLAConfig to OpenVLAConfig."""
        from crane_x7_vla.config.openvla_config import OpenVLASpecificConfig

        # Create OpenVLA-specific config from backend_config if available
        openvla_specific = OpenVLASpecificConfig()
        if config.backend_config:
            for key, value in config.backend_config.items():
                if hasattr(openvla_specific, key):
                    setattr(openvla_specific, key, value)

        # Create OpenVLAConfig
        openvla_config = OpenVLAConfig(
            backend="openvla",
            data=config.data,
            training=config.training,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            seed=config.seed,
            resume_from_checkpoint=config.resume_from_checkpoint,
            openvla=openvla_specific
        )

        return openvla_config

    def _convert_to_openpi_config(self, config: UnifiedVLAConfig) -> OpenPIConfig:
        """Convert UnifiedVLAConfig to OpenPIConfig."""
        from crane_x7_vla.config.openpi_config import OpenPISpecificConfig

        # Create OpenPI-specific config from backend_config if available
        openpi_specific = OpenPISpecificConfig()
        if config.backend_config:
            for key, value in config.backend_config.items():
                if hasattr(openpi_specific, key):
                    setattr(openpi_specific, key, value)

        # Create OpenPIConfig
        openpi_config = OpenPIConfig(
            backend="openpi",
            data=config.data,
            training=config.training,
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            seed=config.seed,
            resume_from_checkpoint=config.resume_from_checkpoint,
            openpi=openpi_specific
        )

        return openpi_config

    def train(self) -> Dict[str, Any]:
        """
        Execute training.

        Returns:
            Dictionary containing training results
        """
        logger.info("=" * 60)
        logger.info(f"Starting training with {self.config.backend} backend")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 60)

        results = self.backend.train()

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        logger.info("=" * 60)

        return results

    def evaluate(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        test_data_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            checkpoint_path: Path to model checkpoint
            test_data_path: Path to test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation...")
        metrics = self.backend.evaluate(checkpoint_path, test_data_path)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_config(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        self.config.to_yaml(path)
        logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "VLATrainer":
        """
        Create trainer from configuration file.

        Args:
            config_path: Path to configuration YAML file

        Returns:
            VLATrainer instance
        """
        config = UnifiedVLAConfig.from_yaml(config_path)
        return cls(config)
