# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenPI-specific configuration.

Extends the unified configuration with OpenPI-specific settings.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from crane_x7_vla.config.base import UnifiedVLAConfig


@dataclass
class OpenPISpecificConfig:
    """OpenPI-specific configuration parameters."""

    model_type: Literal["pi0", "pi0_fast", "pi0.5"] = "pi0_fast"
    """OpenPI model type"""

    pretrained_model_path: Optional[str] = None
    """Path to pretrained model checkpoint"""

    action_dim: int = 32
    """Action dimension (OpenPI uses 32, will pad from CRANE-X7's 8)"""

    state_dim: int = 32
    """State dimension (matches action_dim)"""

    action_horizon: int = 50
    """Number of future actions to predict"""

    image_size: tuple = (224, 224)
    """Image size (height, width)"""

    num_cameras: int = 3
    """Number of camera views (base, left_wrist, right_wrist)"""

    camera_names: list = field(default_factory=lambda: [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb"
    ])
    """Camera names following LeRobot convention"""

    # Normalization settings
    normalize_actions: bool = True
    """Whether to normalize actions"""

    normalization_mode: Literal["quantile", "zscore"] = "quantile"
    """Normalization mode for actions and states"""

    quantile_low: float = 0.01
    """Lower quantile for normalization"""

    quantile_high: float = 0.99
    """Upper quantile for normalization"""

    # Model architecture
    use_lora: bool = True
    """Whether to use LoRA"""

    lora_rank: int = 32
    """LoRA rank"""

    lora_alpha: int = 16
    """LoRA alpha"""

    lora_dropout: float = 0.1
    """LoRA dropout"""

    # Training specific
    max_token_len: int = 128
    """Maximum token length for language instructions"""

    use_depth: bool = False
    """Whether to use depth images"""

    chunk_interpolation: Literal["repeat", "linear"] = "linear"
    """How to generate action chunks from single actions"""


@dataclass
class OpenPIConfig(UnifiedVLAConfig):
    """
    Configuration for OpenPI training.

    Extends UnifiedVLAConfig with OpenPI-specific parameters.
    """

    openpi: OpenPISpecificConfig = field(default_factory=OpenPISpecificConfig)
    """OpenPI-specific configuration"""

    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()
        self.backend = "openpi"

        # Store OpenPI-specific config in backend_config dict
        self.backend_config = {
            'model_type': self.openpi.model_type,
            'pretrained_model_path': self.openpi.pretrained_model_path,
            'action_dim': self.openpi.action_dim,
            'state_dim': self.openpi.state_dim,
            'action_horizon': self.openpi.action_horizon,
            'image_size': self.openpi.image_size,
            'num_cameras': self.openpi.num_cameras,
            'camera_names': self.openpi.camera_names,
            'normalize_actions': self.openpi.normalize_actions,
            'normalization_mode': self.openpi.normalization_mode,
            'quantile_low': self.openpi.quantile_low,
            'quantile_high': self.openpi.quantile_high,
            'use_lora': self.openpi.use_lora,
            'lora_rank': self.openpi.lora_rank,
            'lora_alpha': self.openpi.lora_alpha,
            'lora_dropout': self.openpi.lora_dropout,
            'max_token_len': self.openpi.max_token_len,
            'use_depth': self.openpi.use_depth,
            'chunk_interpolation': self.openpi.chunk_interpolation,
        }

    @property
    def crane_x7_action_dim(self) -> int:
        """Get CRANE-X7's native action dimension (8 DOF)."""
        return 8

    @property
    def model_action_dim(self) -> int:
        """Get model's expected action dimension (32 for OpenPI)."""
        return self.openpi.action_dim

    @property
    def action_horizon(self) -> int:
        """OpenPI predicts action chunks."""
        return self.openpi.action_horizon
