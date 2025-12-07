# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenPI PyTorch-specific configuration.

Extends the unified configuration with OpenPI PyTorch-specific settings.
Uses HuggingFace Pi0 model instead of JAX/Flax implementation.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from crane_x7_vla.config.base import UnifiedVLAConfig


@dataclass
class OpenPIPytorchSpecificConfig:
    """OpenPI PyTorch-specific configuration parameters."""

    model_name: str = "lerobot/pi0_base"
    """HuggingFace model ID for Pi0"""

    pretrained_checkpoint: Optional[str] = None
    """Path to pretrained checkpoint (overrides model_name if specified)"""

    action_dim: int = 32
    """Action dimension (Pi0 uses 32, will pad from CRANE-X7's 8)"""

    state_dim: int = 32
    """State dimension (matches action_dim)"""

    action_horizon: int = 50
    """Number of future actions to predict (action chunk size)"""

    image_size: tuple = (224, 224)
    """Image size (height, width)"""

    num_cameras: int = 3
    """Number of camera views (base, left_wrist, right_wrist)"""

    camera_names: List[str] = field(default_factory=lambda: [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb"
    ])
    """Camera names following LeRobot convention"""

    # Flow matching settings
    num_denoise_steps: int = 10
    """Number of denoising steps for flow matching inference"""

    noise_scheduler: Literal["linear", "cosine"] = "linear"
    """Noise scheduler type for flow matching"""

    # Normalization settings
    normalize_actions: bool = True
    """Whether to normalize actions"""

    normalization_mode: Literal["quantile", "zscore"] = "quantile"
    """Normalization mode for actions and states"""

    quantile_low: float = 0.01
    """Lower quantile for normalization"""

    quantile_high: float = 0.99
    """Upper quantile for normalization"""

    # Model architecture (LoRA currently not supported in PyTorch Pi0)
    use_lora: bool = False
    """Whether to use LoRA (currently not supported, use full finetuning)"""

    lora_rank: int = 32
    """LoRA rank (for future use)"""

    lora_alpha: int = 16
    """LoRA alpha (for future use)"""

    lora_dropout: float = 0.1
    """LoRA dropout (for future use)"""

    # Training specific
    max_token_len: int = 128
    """Maximum token length for language instructions"""

    use_depth: bool = False
    """Whether to use depth images"""

    # Precision
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    """Training precision (bfloat16 recommended)"""

    # Checkpointing
    keep_period: int = 5000
    """Keep checkpoints at this interval (steps)"""

    # Default prompt
    default_prompt: str = "manipulate objects"
    """Default language instruction for tasks"""

    # Delta actions
    use_delta_actions: bool = False
    """Whether to use delta actions (action = next_state - current_state)"""


@dataclass
class OpenPIPytorchConfig(UnifiedVLAConfig):
    """
    Configuration for OpenPI PyTorch training.

    Extends UnifiedVLAConfig with OpenPI PyTorch-specific parameters.
    Uses HuggingFace Pi0 model with flow matching for action prediction.
    """

    openpi_pytorch: OpenPIPytorchSpecificConfig = field(
        default_factory=OpenPIPytorchSpecificConfig
    )
    """OpenPI PyTorch-specific configuration"""

    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()
        self.backend = "openpi-pytorch"

        # Store OpenPI PyTorch-specific config in backend_config dict
        self.backend_config = {
            'model_name': self.openpi_pytorch.model_name,
            'pretrained_checkpoint': self.openpi_pytorch.pretrained_checkpoint,
            'action_dim': self.openpi_pytorch.action_dim,
            'state_dim': self.openpi_pytorch.state_dim,
            'action_horizon': self.openpi_pytorch.action_horizon,
            'image_size': self.openpi_pytorch.image_size,
            'num_cameras': self.openpi_pytorch.num_cameras,
            'camera_names': self.openpi_pytorch.camera_names,
            'num_denoise_steps': self.openpi_pytorch.num_denoise_steps,
            'noise_scheduler': self.openpi_pytorch.noise_scheduler,
            'normalize_actions': self.openpi_pytorch.normalize_actions,
            'normalization_mode': self.openpi_pytorch.normalization_mode,
            'quantile_low': self.openpi_pytorch.quantile_low,
            'quantile_high': self.openpi_pytorch.quantile_high,
            'use_lora': self.openpi_pytorch.use_lora,
            'lora_rank': self.openpi_pytorch.lora_rank,
            'lora_alpha': self.openpi_pytorch.lora_alpha,
            'lora_dropout': self.openpi_pytorch.lora_dropout,
            'max_token_len': self.openpi_pytorch.max_token_len,
            'use_depth': self.openpi_pytorch.use_depth,
            'precision': self.openpi_pytorch.precision,
            'keep_period': self.openpi_pytorch.keep_period,
            'default_prompt': self.openpi_pytorch.default_prompt,
            'use_delta_actions': self.openpi_pytorch.use_delta_actions,
        }

    @property
    def crane_x7_action_dim(self) -> int:
        """Get CRANE-X7's native action dimension (8 DOF)."""
        return 8

    @property
    def model_action_dim(self) -> int:
        """Get model's expected action dimension (32 for Pi0)."""
        return self.openpi_pytorch.action_dim

    @property
    def action_horizon(self) -> int:
        """Pi0 predicts action chunks."""
        return self.openpi_pytorch.action_horizon
