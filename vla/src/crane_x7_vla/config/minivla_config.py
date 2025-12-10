# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
MiniVLA-specific configuration.

Extends the unified configuration with MiniVLA-specific settings including:
- Qwen 2.5 0.5B LLM backbone
- VQ Action Chunking (Residual VQ)
- Multi-image support (history and wrist camera)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from crane_x7_vla.config.base import UnifiedVLAConfig


@dataclass
class VQConfig:
    """Configuration for Vector Quantization (VQ) Action Chunking."""

    enabled: bool = True
    """Whether to use VQ action chunking"""

    vq_path: Optional[str] = None
    """Path to pre-trained VQ model (None to train from scratch)"""

    action_horizon: int = 8
    """Action chunk horizon (number of future actions to predict)"""

    n_embed: int = 256
    """Number of embeddings in codebook (vocabulary size)"""

    n_latent: int = 512
    """Latent dimension for VQ encoder/decoder"""

    n_groups: int = 7
    """Number of residual VQ groups (sequential codebooks)"""

    commitment_weight: float = 0.25
    """Commitment loss weight for VQ training"""

    entropy_weight: float = 0.0
    """Entropy regularization weight for codebook usage"""


@dataclass
class MultiImageConfig:
    """Configuration for multi-image input support."""

    enabled: bool = True
    """Whether to use multi-image input"""

    image_history: int = 2
    """Number of historical frames to include (1 = current only)"""

    use_wrist_camera: bool = True
    """Whether to include wrist camera image"""

    camera_names: List[str] = field(default_factory=lambda: [
        "primary",  # Main external camera
        "wrist",    # Wrist-mounted camera
    ])
    """Names of cameras to use"""

    concat_method: str = "sequence"
    """How to combine multiple images: 'sequence' (concat tokens) or 'channel'"""


@dataclass
class MiniVLASpecificConfig:
    """MiniVLA-specific configuration parameters."""

    # Model settings
    llm_model_id: str = "Qwen/Qwen2.5-0.5B"
    """HuggingFace model ID for LLM backbone"""

    vision_backbone: str = "dinosiglip-vit-so-224px"
    """Vision backbone type (DINO-SigLIP for OpenVLA compatibility)"""

    use_pretrained_vlm: bool = True
    """Whether to use pre-trained VLM weights"""

    pretrained_vlm_path: Optional[str] = None
    """Path to pre-trained VLM checkpoint (None to use HF model)"""

    # VQ Action Chunking
    vq: VQConfig = field(default_factory=VQConfig)
    """VQ action chunking configuration"""

    # Multi-image support
    multi_image: MultiImageConfig = field(default_factory=MultiImageConfig)
    """Multi-image input configuration"""

    # LoRA settings
    use_lora: bool = True
    """Whether to use LoRA for parameter-efficient fine-tuning"""

    lora_rank: int = 16
    """LoRA rank (smaller than OpenVLA due to smaller model)"""

    lora_alpha: int = 8
    """LoRA alpha scaling parameter"""

    lora_dropout: float = 0.05
    """LoRA dropout"""

    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    """Target modules for LoRA (Qwen2 architecture)"""

    # Tokenization
    use_extra_tokens: bool = True
    """Use extra tokens for action bins (avoids vocabulary conflicts)"""

    n_extra_tokens: int = 256
    """Number of extra tokens to add for action bins"""

    action_tokenization_bins: int = 256
    """Number of bins for action discretization (non-VQ mode)"""

    action_range: Tuple[float, float] = (-1.0, 1.0)
    """Range for action values (min, max)"""

    # Image settings
    image_size: Tuple[int, int] = (224, 224)
    """Image size (height, width)"""

    # Performance settings
    use_flash_attention: bool = True
    """Use Flash Attention 2 if available (faster on Qwen2)"""

    use_quantization: bool = False
    """Use quantization (4-bit/8-bit) for memory efficiency"""

    compile_model: bool = False
    """Use torch.compile() for optimization"""

    # Checkpoint settings
    skip_merge_on_save: bool = True
    """Skip LoRA merge during checkpoint saving"""

    image_aug: bool = True
    """Whether to use image augmentation during training"""

    @property
    def action_horizon(self) -> int:
        """Get effective action horizon."""
        if self.vq.enabled:
            return self.vq.action_horizon
        return 1

    @property
    def total_image_count(self) -> int:
        """Get total number of images per observation."""
        if not self.multi_image.enabled:
            return 1
        count = self.multi_image.image_history
        if self.multi_image.use_wrist_camera:
            count += 1
        return count


@dataclass
class MiniVLAConfig(UnifiedVLAConfig):
    """
    Configuration for MiniVLA training.

    Extends UnifiedVLAConfig with MiniVLA-specific parameters:
    - Qwen 2.5 0.5B LLM backbone (~1B total params)
    - VQ Action Chunking for multi-step action prediction
    - Multi-image support (history + wrist camera)
    """

    minivla: MiniVLASpecificConfig = field(default_factory=MiniVLASpecificConfig)
    """MiniVLA-specific configuration"""

    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()
        self.backend = "minivla"

        # Store MiniVLA-specific config in backend_config dict
        self.backend_config = {
            # Model settings
            'llm_model_id': self.minivla.llm_model_id,
            'vision_backbone': self.minivla.vision_backbone,
            'use_pretrained_vlm': self.minivla.use_pretrained_vlm,
            'pretrained_vlm_path': self.minivla.pretrained_vlm_path,
            # VQ settings
            'vq_enabled': self.minivla.vq.enabled,
            'vq_path': self.minivla.vq.vq_path,
            'vq_action_horizon': self.minivla.vq.action_horizon,
            'vq_n_embed': self.minivla.vq.n_embed,
            'vq_n_latent': self.minivla.vq.n_latent,
            'vq_n_groups': self.minivla.vq.n_groups,
            # Multi-image settings
            'multi_image_enabled': self.minivla.multi_image.enabled,
            'image_history': self.minivla.multi_image.image_history,
            'use_wrist_camera': self.minivla.multi_image.use_wrist_camera,
            'camera_names': self.minivla.multi_image.camera_names,
            # LoRA settings
            'use_lora': self.minivla.use_lora,
            'lora_rank': self.minivla.lora_rank,
            'lora_alpha': self.minivla.lora_alpha,
            'lora_dropout': self.minivla.lora_dropout,
            'lora_target_modules': self.minivla.lora_target_modules,
            # Tokenization
            'use_extra_tokens': self.minivla.use_extra_tokens,
            'n_extra_tokens': self.minivla.n_extra_tokens,
            'action_tokenization_bins': self.minivla.action_tokenization_bins,
            'action_range': self.minivla.action_range,
            # Image settings
            'image_size': self.minivla.image_size,
            # Performance
            'use_flash_attention': self.minivla.use_flash_attention,
            'use_quantization': self.minivla.use_quantization,
            'compile_model': self.minivla.compile_model,
            'skip_merge_on_save': self.minivla.skip_merge_on_save,
            'image_aug': self.minivla.image_aug,
        }

    @property
    def action_dim(self) -> int:
        """Get action dimension (CRANE-X7 has 8 DOF)."""
        return 8

    @property
    def action_horizon(self) -> int:
        """Get action horizon (VQ chunking returns multiple actions)."""
        return self.minivla.action_horizon
