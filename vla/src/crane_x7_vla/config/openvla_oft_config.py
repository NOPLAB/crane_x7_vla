# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenVLA-OFT (Optimized Fine-Tuning) configuration.

Implements the configuration for OpenVLA-OFT which includes:
- L1 Regression Action Head (continuous actions instead of tokens)
- Action Chunking (predict multiple future actions)
- FiLM (Feature-wise Linear Modulation for language conditioning)
- Proprioceptive Input support
- Multi-image Input support

Reference: https://arxiv.org/abs/2502.19645
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from crane_x7_vla.config.base import UnifiedVLAConfig


@dataclass
class FiLMConfig:
    """Configuration for FiLM (Feature-wise Linear Modulation).

    FiLM modulates visual features using language embeddings:
    output = (1 + gamma) * x + beta

    This enables better language following by injecting language
    information directly into the vision backbone.
    """

    enabled: bool = True
    """Whether to use FiLM for language-vision modulation"""

    language_hidden_dim: int = 4096
    """Hidden dimension for language features (from LLM, Llama-2 7B: 4096)"""


@dataclass
class ActionHeadConfig:
    """Configuration for L1 Regression Action Head.

    Uses MLPResNet architecture to directly predict continuous actions
    from LLM hidden states, bypassing token discretization.
    """

    hidden_dim: int = 4096
    """Hidden dimension for MLP layers (matches LLM hidden dim)"""

    num_blocks: int = 2
    """Number of residual blocks in MLPResNet"""

    use_layer_norm: bool = True
    """Whether to use LayerNorm in MLPResNet"""

    dropout: float = 0.0
    """Dropout rate"""


@dataclass
class ProprioConfig:
    """Configuration for proprioceptive (robot state) input.

    Projects robot state into LLM embedding space using a 2-layer MLP.
    """

    enabled: bool = True
    """Whether to use proprioceptive input"""

    proprio_dim: int = 8
    """Proprioceptive dimension (CRANE-X7: 8 = 7 joints + 1 gripper)"""

    hidden_dim: int = 4096
    """Hidden dimension for proprio projector (matches LLM hidden dim)"""


@dataclass
class MultiImageConfig:
    """Configuration for multi-image input in OFT.

    Supports multiple camera views (e.g., third-person + wrist camera).
    """

    enabled: bool = True
    """Whether to use multi-image input"""

    num_images: int = 2
    """Number of images (e.g., 1=primary only, 2=primary+wrist)"""

    camera_names: List[str] = field(default_factory=lambda: ["primary", "wrist"])
    """Camera names for each image"""


@dataclass
class OpenVLAOFTSpecificConfig:
    """OpenVLA-OFT specific configuration parameters.

    Key differences from standard OpenVLA:
    - Uses L1 regression instead of tokenized actions
    - Predicts action chunks (multiple future actions)
    - Optional FiLM conditioning for better language following
    - Optional proprioceptive input
    - Optional multi-image support
    """

    model_id: str = "openvla/openvla-7b"
    """Base OpenVLA model to fine-tune from HuggingFace Hub"""

    # Action settings
    action_dim: int = 8
    """Action dimension (CRANE-X7: 8 = 7 joints + 1 gripper)"""

    action_horizon: int = 8
    """Action chunk horizon (number of future actions to predict)"""

    # Component configs
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    """L1 regression action head configuration"""

    film: FiLMConfig = field(default_factory=FiLMConfig)
    """FiLM configuration for language-vision modulation"""

    proprio: ProprioConfig = field(default_factory=ProprioConfig)
    """Proprioceptive input configuration"""

    multi_image: MultiImageConfig = field(default_factory=MultiImageConfig)
    """Multi-image input configuration"""

    # LoRA settings (same as OpenVLA)
    use_lora: bool = True
    """Whether to use LoRA for parameter-efficient fine-tuning"""

    lora_rank: int = 32
    """LoRA rank"""

    lora_alpha: int = 16
    """LoRA alpha scaling parameter"""

    lora_dropout: float = 0.0
    """LoRA dropout"""

    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    """Target modules for LoRA (LLM layers only, exclude vision backbone)"""

    # Training settings
    image_size: Tuple[int, int] = (224, 224)
    """Image size (height, width)"""

    use_flash_attention: bool = False
    """Use Flash Attention 2 if available"""

    use_quantization: bool = False
    """Use quantization (4-bit) for memory efficiency"""

    skip_merge_on_save: bool = True
    """Skip LoRA merge during checkpoint saving to avoid NCCL timeout"""

    image_aug: bool = True
    """Whether to use image augmentation during training"""


@dataclass
class OpenVLAOFTConfig(UnifiedVLAConfig):
    """
    Configuration for OpenVLA-OFT training.

    Extends UnifiedVLAConfig with OpenVLA-OFT specific parameters.

    Example usage:
        config = OpenVLAOFTConfig(
            data=DataConfig(data_root="./data"),
            training=TrainingConfig(batch_size=16),
            openvla_oft=OpenVLAOFTSpecificConfig(
                action_horizon=8,
                film=FiLMConfig(enabled=True),
                proprio=ProprioConfig(enabled=True),
            ),
        )
    """

    openvla_oft: OpenVLAOFTSpecificConfig = field(default_factory=OpenVLAOFTSpecificConfig)
    """OpenVLA-OFT specific configuration"""

    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()
        self.backend = "openvla-oft"

        # Store OpenVLA-OFT specific config in backend_config dict
        self.backend_config = {
            'model_id': self.openvla_oft.model_id,
            'action_dim': self.openvla_oft.action_dim,
            'action_horizon': self.openvla_oft.action_horizon,
            'use_lora': self.openvla_oft.use_lora,
            'lora_rank': self.openvla_oft.lora_rank,
            'lora_alpha': self.openvla_oft.lora_alpha,
            'lora_dropout': self.openvla_oft.lora_dropout,
            'lora_target_modules': self.openvla_oft.lora_target_modules,
            'image_size': self.openvla_oft.image_size,
            'use_flash_attention': self.openvla_oft.use_flash_attention,
            'use_quantization': self.openvla_oft.use_quantization,
            'skip_merge_on_save': self.openvla_oft.skip_merge_on_save,
            'image_aug': self.openvla_oft.image_aug,
            # OFT-specific
            'film_enabled': self.openvla_oft.film.enabled,
            'proprio_enabled': self.openvla_oft.proprio.enabled,
            'multi_image_enabled': self.openvla_oft.multi_image.enabled,
            'num_images': self.openvla_oft.multi_image.num_images,
        }

    @property
    def action_dim(self) -> int:
        """Get action dimension (CRANE-X7 has 8 DOF)."""
        return self.openvla_oft.action_dim

    @property
    def action_horizon(self) -> int:
        """Get action horizon (number of future actions predicted)."""
        return self.openvla_oft.action_horizon

    @property
    def use_film(self) -> bool:
        """Whether FiLM is enabled."""
        return self.openvla_oft.film.enabled

    @property
    def use_proprio(self) -> bool:
        """Whether proprioceptive input is enabled."""
        return self.openvla_oft.proprio.enabled

    @property
    def use_multi_image(self) -> bool:
        """Whether multi-image input is enabled."""
        return self.openvla_oft.multi_image.enabled
