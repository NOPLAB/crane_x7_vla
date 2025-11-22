# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Unified configuration system for VLA training.

This module provides a common configuration interface that works across
different VLA backends (OpenVLA, OpenPI, etc.).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import yaml


@dataclass
class CameraConfig:
    """Configuration for camera setup."""

    name: str
    """Camera name (e.g., 'primary', 'wrist_left', 'wrist_right')"""

    topic: str
    """ROS topic for camera stream"""

    enabled: bool = True
    """Whether this camera is enabled"""

    width: int = 640
    """Image width"""

    height: int = 480
    """Image height"""

    calibration_file: Optional[str] = None
    """Path to camera calibration file"""


@dataclass
class DataConfig:
    """Configuration for dataset."""

    data_root: Union[str, Path]
    """Root directory containing training data"""

    format: Literal["tfrecord", "lerobot"] = "tfrecord"
    """Data format"""

    train_split: float = 0.9
    """Fraction of data to use for training"""

    val_split: float = 0.1
    """Fraction of data to use for validation"""

    shuffle: bool = True
    """Whether to shuffle the dataset"""

    num_workers: int = 4
    """Number of data loading workers"""

    prefetch_factor: int = 2
    """Number of batches to prefetch per worker"""

    cameras: List[CameraConfig] = field(default_factory=list)
    """Camera configurations"""

    def __post_init__(self):
        """Convert string path to Path object."""
        self.data_root = Path(self.data_root)


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    batch_size: int = 16
    """Training batch size"""

    num_epochs: int = 100
    """Number of training epochs"""

    learning_rate: float = 5e-4
    """Learning rate"""

    weight_decay: float = 0.01
    """Weight decay for regularization"""

    warmup_steps: int = 1000
    """Number of warmup steps"""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps"""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping"""

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    """Mixed precision training mode"""

    gradient_checkpointing: bool = False
    """Enable gradient checkpointing"""

    save_interval: int = 1000
    """Save checkpoint every N steps"""

    eval_interval: int = 500
    """Evaluate every N steps"""

    log_interval: int = 10
    """Log metrics every N steps"""


@dataclass
class UnifiedVLAConfig:
    """
    Unified configuration for VLA training.

    This configuration works across different VLA backends.
    Backend-specific settings are stored in separate config objects.
    """

    backend: Literal["openvla", "openpi"]
    """VLA backend to use"""

    data: DataConfig
    """Data configuration"""

    training: TrainingConfig
    """Training configuration"""

    output_dir: Union[str, Path] = "./outputs"
    """Output directory for checkpoints and logs"""

    experiment_name: str = "crane_x7_vla"
    """Experiment name"""

    seed: int = 42
    """Random seed"""

    resume_from_checkpoint: Optional[Union[str, Path]] = None
    """Path to checkpoint to resume from"""

    # Backend-specific configs (will be populated by subclasses)
    backend_config: Optional[Dict] = None
    """Backend-specific configuration dictionary"""

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.output_dir = Path(self.output_dir)
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "UnifiedVLAConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            UnifiedVLAConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse data config
        data_config_dict = config_dict.pop('data', {})
        cameras_list = data_config_dict.pop('cameras', [])
        cameras = [CameraConfig(**cam) for cam in cameras_list]
        data_config = DataConfig(**data_config_dict, cameras=cameras)

        # Parse training config
        training_config_dict = config_dict.pop('training', {})
        training_config = TrainingConfig(**training_config_dict)

        # Create main config
        return cls(
            data=data_config,
            training=training_config,
            **config_dict
        )

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = {
            'backend': self.backend,
            'output_dir': str(self.output_dir),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'resume_from_checkpoint': str(self.resume_from_checkpoint) if self.resume_from_checkpoint else None,
            'data': {
                'data_root': str(self.data.data_root),
                'format': self.data.format,
                'train_split': self.data.train_split,
                'val_split': self.data.val_split,
                'shuffle': self.data.shuffle,
                'num_workers': self.data.num_workers,
                'prefetch_factor': self.data.prefetch_factor,
                'cameras': [
                    {
                        'name': cam.name,
                        'topic': cam.topic,
                        'enabled': cam.enabled,
                        'width': cam.width,
                        'height': cam.height,
                        'calibration_file': cam.calibration_file
                    }
                    for cam in self.data.cameras
                ]
            },
            'training': {
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'warmup_steps': self.training.warmup_steps,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'max_grad_norm': self.training.max_grad_norm,
                'mixed_precision': self.training.mixed_precision,
                'gradient_checkpointing': self.training.gradient_checkpointing,
                'save_interval': self.training.save_interval,
                'eval_interval': self.training.eval_interval,
                'log_interval': self.training.log_interval,
            },
            'backend_config': self.backend_config
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
