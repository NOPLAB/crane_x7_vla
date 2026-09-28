# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Unified configuration system for VLA training.

This module provides a common configuration interface that works across
different VLA backends (OpenVLA, OpenPI, etc.).
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal

import yaml


def _configuration_from_dict(default, values: dict):
    """Build a nested configuration from plain YAML values."""
    known = {item.name for item in fields(default)}
    converted = {}
    for name, value in values.items():
        if name not in known:
            continue  # Older example configs may contain retired options.
        current = getattr(default, name)
        if is_dataclass(current) and isinstance(value, dict):
            value = _configuration_from_dict(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        converted[name] = value
    return type(default)(**converted)


def _plain_yaml_value(value):
    """Convert tuples and dataclasses to YAML safe types."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain_yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_yaml_value(item) for item in value]
    return value


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

    calibration_file: str | None = None
    """Path to camera calibration file"""


@dataclass
class DataConfig:
    """Configuration for dataset."""

    data_root: str | Path
    """Root directory containing training data"""

    format: Literal["tfrecord", "lerobot"] = "tfrecord"
    """Data format"""

    train_split: float = 0.9
    """Fraction of data to use for training"""

    val_split: float = 0.1
    """Fraction of data to use for validation"""

    shuffle: bool = True
    """Whether to shuffle the dataset"""

    shuffle_buffer_size: int = 1000
    """Shuffle buffer size for dataset loading"""

    num_workers: int = 4
    """Number of data loading workers"""

    prefetch_factor: int = 2
    """Number of batches to prefetch per worker"""

    cameras: list[CameraConfig] = field(default_factory=list)
    """Camera configurations"""

    def __post_init__(self):
        """Convert string path to Path object."""
        self.data_root = Path(self.data_root)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    LoRA enables parameter-efficient fine-tuning by adding low-rank
    decomposition matrices to transformer layers.
    """

    enabled: bool = True
    """Whether to use LoRA for parameter-efficient fine-tuning"""

    rank: int = 32
    """LoRA rank (higher = more parameters, better capacity)"""

    alpha: int = 16
    """LoRA alpha scaling factor (effective scaling = alpha/rank)"""

    dropout: float = 0.05
    """Dropout applied to LoRA weights"""

    target_modules: list[str] | None = None
    """Target modules for LoRA. None = use backend defaults.
    Common: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']"""

    use_rslora: bool = False
    """Use Rank-Stabilized LoRA (rsLoRA) for better training stability"""

    use_dora: bool = False
    """Use Weight-Decomposed LoRA (DoRA) for improved fine-tuning"""

    bias: Literal["none", "all", "lora_only"] = "none"
    """Which bias to train: 'none', 'all', or 'lora_only'"""

    skip_merge_on_save: bool = True
    """Skip LoRA merge during checkpoint saving to avoid timeout.
    Set to True for multi-GPU training; merge post-training with merge_lora script."""


@dataclass
class OverfittingConfig:
    """Configuration for overfitting detection during training.

    Uses step-level splitting (not episode-level) to properly detect
    memorization by holding out random steps from the same episodes.
    """

    overfit_split_ratio: float = 0.1
    """Ratio of steps to use for overfitting detection (0.0 to disable)"""

    overfit_check_interval: int = 500
    """Check overfitting every N gradient steps"""

    overfit_check_steps: int = 50
    """Number of steps per overfitting check"""


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    batch_size: int = 16
    """Training batch size"""

    num_epochs: int = 100
    """Number of training epochs"""

    max_steps: int = 200_000
    """Maximum number of training steps"""

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

    overfitting: OverfittingConfig = field(default_factory=OverfittingConfig)
    """Overfitting detection configuration"""

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    """LoRA configuration for parameter-efficient fine-tuning"""

    output_dir: str | Path = "./outputs"
    """Output directory for checkpoints and logs"""

    experiment_name: str = "crane_x7_vla"
    """Experiment name"""

    seed: int = 42
    """Random seed"""

    resume_from_checkpoint: str | Path | None = None
    """Path to checkpoint to resume from"""

    # Logging settings
    wandb_project: str = "crane-x7-vla"
    """Weights & Biases project name"""

    wandb_entity: str | None = None
    """Weights & Biases entity (username or team)"""

    # Backend-specific configs (will be populated by subclasses)
    backend_config: dict | None = None
    """Backend-specific configuration dictionary"""

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.output_dir = Path(self.output_dir)
        if self.resume_from_checkpoint is not None:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "UnifiedVLAConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            UnifiedVLAConfig instance
        """
        with Path(yaml_path).open() as f:
            config_dict = yaml.safe_load(f)

        # Parse data config
        data_config_dict = config_dict.pop("data", {})
        cameras_list = data_config_dict.pop("cameras", [])
        cameras = [CameraConfig(**cam) for cam in cameras_list]
        data_config = DataConfig(**data_config_dict, cameras=cameras)

        # Parse training config
        training_config_dict = config_dict.pop("training", {})
        training_config = TrainingConfig(**training_config_dict)

        # Parse overfitting config
        overfitting_config_dict = config_dict.pop("overfitting", {})
        overfitting_config = (
            OverfittingConfig(**overfitting_config_dict) if overfitting_config_dict else OverfittingConfig()
        )

        # Parse LoRA config
        lora_config_dict = config_dict.pop("lora", {})
        lora_config = LoRAConfig(**lora_config_dict) if lora_config_dict else LoRAConfig()

        # The CLI first reads the base class to discover the backend, then
        # reads its subclass. Accept both current nested settings and older
        # flat backend_config files in either pass.
        backend_settings = config_dict.pop("backend_config", None) or {}
        nested_name = {
            "openvla": "openvla",
            "openvla-oft": "openvla_oft",
            "minivla": "minivla",
            "pi0": "pi0",
            "pi0.5": "pi0",
        }.get(config_dict.get("backend"))
        for name in ("openvla", "openvla_oft", "minivla", "pi0"):
            if name != nested_name:
                config_dict.pop(name, None)
        legacy_vq = config_dict.pop("vq", None)
        legacy_multi_image = config_dict.pop("multi_image", None)
        if nested_name and any(item.name == nested_name for item in fields(cls)):
            specific_field = next(item for item in fields(cls) if item.name == nested_name)
            default_specific = specific_field.default_factory()
            settings = dict(backend_settings)
            settings.update(config_dict.pop(nested_name, None) or {})
            if nested_name == "minivla":
                for prefix, key in (("vq_", "vq"), ("multi_image_", "multi_image")):
                    nested = settings.setdefault(key, {})
                    for setting in list(settings):
                        if setting.startswith(prefix):
                            nested[setting[len(prefix):]] = settings.pop(setting)
                if legacy_vq:
                    settings.setdefault("vq", {}).update(legacy_vq)
                if legacy_multi_image:
                    settings.setdefault("multi_image", {}).update(legacy_multi_image)
            elif nested_name == "openvla_oft":
                for key, field_name in (("film_enabled", "film"), ("proprio_enabled", "proprio"),
                                        ("multi_image_enabled", "multi_image"), ("num_images", "multi_image")):
                    if key in settings:
                        target = "num_images" if key == "num_images" else "enabled"
                        settings.setdefault(field_name, {})[target] = settings.pop(key)
            if nested_name == "pi0" and config_dict["backend"] == "pi0.5":
                settings.setdefault("model_type", "pi0.5")
            config_dict[nested_name] = _configuration_from_dict(default_specific, settings)
        else:
            config_dict.pop(nested_name, None)
            config_dict["backend_config"] = backend_settings or None

        # Create main config
        return cls(
            data=data_config,
            training=training_config,
            overfitting=overfitting_config,
            lora=lora_config,
            **config_dict,
        )

    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = {
            "backend": self.backend,
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "resume_from_checkpoint": str(self.resume_from_checkpoint) if self.resume_from_checkpoint else None,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "data": {
                "data_root": str(self.data.data_root),
                "format": self.data.format,
                "train_split": self.data.train_split,
                "val_split": self.data.val_split,
                "shuffle": self.data.shuffle,
                "shuffle_buffer_size": self.data.shuffle_buffer_size,
                "num_workers": self.data.num_workers,
                "prefetch_factor": self.data.prefetch_factor,
                "cameras": [
                    {
                        "name": cam.name,
                        "topic": cam.topic,
                        "enabled": cam.enabled,
                        "width": cam.width,
                        "height": cam.height,
                        "calibration_file": cam.calibration_file,
                    }
                    for cam in self.data.cameras
                ],
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "max_steps": self.training.max_steps,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "warmup_steps": self.training.warmup_steps,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "max_grad_norm": self.training.max_grad_norm,
                "mixed_precision": self.training.mixed_precision,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "save_interval": self.training.save_interval,
                "eval_interval": self.training.eval_interval,
                "log_interval": self.training.log_interval,
            },
            "overfitting": {
                "overfit_split_ratio": self.overfitting.overfit_split_ratio,
                "overfit_check_interval": self.overfitting.overfit_check_interval,
                "overfit_check_steps": self.overfitting.overfit_check_steps,
            },
            "lora": {
                "enabled": self.lora.enabled,
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
                "use_rslora": self.lora.use_rslora,
                "use_dora": self.lora.use_dora,
                "bias": self.lora.bias,
                "skip_merge_on_save": self.lora.skip_merge_on_save,
            },
        }

        for name in ("openvla", "openvla_oft", "minivla", "pi0"):
            if hasattr(self, name):
                config_dict[name] = _plain_yaml_value(getattr(self, name))
                break
        else:
            config_dict["backend_config"] = _plain_yaml_value(self.backend_config)

        with Path(yaml_path).open("w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
