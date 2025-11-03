"""
OpenVLA-specific configuration.

Extends the unified configuration with OpenVLA-specific settings.
"""

from dataclasses import dataclass, field
from typing import Optional
from crane_x7_vla.config.base import UnifiedVLAConfig


@dataclass
class OpenVLASpecificConfig:
    """OpenVLA-specific configuration parameters."""

    model_id: str = "openvla/openvla-7b"
    """HuggingFace model identifier"""

    use_lora: bool = True
    """Whether to use LoRA for parameter-efficient fine-tuning"""

    lora_rank: int = 32
    """LoRA rank"""

    lora_alpha: int = 16
    """LoRA alpha scaling parameter"""

    lora_dropout: float = 0.05
    """LoRA dropout"""

    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    """Target modules for LoRA"""

    action_tokenization_bins: int = 256
    """Number of bins for action discretization"""

    action_range: tuple = (-1.0, 1.0)
    """Range for action values (min, max)"""

    max_sequence_length: int = 512
    """Maximum sequence length"""

    image_size: tuple = (224, 224)
    """Image size (height, width)"""

    use_flash_attention: bool = False
    """Use Flash Attention 2 if available"""

    compile_model: bool = False
    """Use torch.compile() for optimization"""


@dataclass
class OpenVLAConfig(UnifiedVLAConfig):
    """
    Configuration for OpenVLA training.

    Extends UnifiedVLAConfig with OpenVLA-specific parameters.
    """

    openvla: OpenVLASpecificConfig = field(default_factory=OpenVLASpecificConfig)
    """OpenVLA-specific configuration"""

    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()
        self.backend = "openvla"

        # Store OpenVLA-specific config in backend_config dict
        self.backend_config = {
            'model_id': self.openvla.model_id,
            'use_lora': self.openvla.use_lora,
            'lora_rank': self.openvla.lora_rank,
            'lora_alpha': self.openvla.lora_alpha,
            'lora_dropout': self.openvla.lora_dropout,
            'lora_target_modules': self.openvla.lora_target_modules,
            'action_tokenization_bins': self.openvla.action_tokenization_bins,
            'action_range': self.openvla.action_range,
            'max_sequence_length': self.openvla.max_sequence_length,
            'image_size': self.openvla.image_size,
            'use_flash_attention': self.openvla.use_flash_attention,
            'compile_model': self.openvla.compile_model,
        }

    @property
    def action_dim(self) -> int:
        """Get action dimension (CRANE-X7 has 8 DOF)."""
        return 8

    @property
    def action_horizon(self) -> int:
        """OpenVLA predicts single-step actions."""
        return 1
