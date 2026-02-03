# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Pi0/Pi0.5 configuration.

Extends the unified configuration with Pi0/Pi0.5-specific settings.
Uses PaliGemma + Expert Gemma architecture from OpenPI.
"""

from dataclasses import dataclass, field
from typing import Literal

from crane_x7_vla.core.config.base import UnifiedVLAConfig


@dataclass
class Pi0SpecificConfig:
    """Pi0/Pi0.5-specific configuration parameters."""

    # Model variant
    model_type: Literal["pi0", "pi0.5"] = "pi0"
    """Model type: 'pi0' or 'pi0.5' (Pi0.5 uses adaRMSNorm and discrete state)"""

    paligemma_variant: str = "gemma_2b"
    """PaliGemma variant for VLM (gemma_2b, gemma_2b_lora, etc.)"""

    action_expert_variant: str = "gemma_300m"
    """Action expert variant (gemma_300m, gemma_300m_lora, etc.)"""

    pretrained_checkpoint: str | None = None
    """Path to pretrained checkpoint (if any)"""

    use_pretrained: bool = True
    """Whether to load pretrained weights for PaliGemma VLM"""

    paligemma_pretrained_id: str = "google/paligemma-3b-pt-224"
    """HuggingFace model ID for pretrained PaliGemma"""

    openpi_checkpoint: str | None = None
    """OpenPI checkpoint name (e.g., "pi0_base", "pi05_base", "pi0_droid", "pi05_droid").
    If specified, loads the full Pi0/Pi0.5 model pretrained on 10k+ hours of robot data.
    This overrides paligemma_pretrained_id and use_pretrained.
    Available checkpoints:
    - pi0_base: Base Pi0 model for fine-tuning
    - pi05_base: Base Pi0.5 model for fine-tuning
    - pi0_droid: Pi0 fine-tuned on DROID dataset
    - pi05_droid: Pi0.5 fine-tuned on DROID dataset
    """

    # Action configuration
    action_dim: int = 32
    """Action dimension (Pi0 uses 32, will pad from CRANE-X7's 8)"""

    state_dim: int = 32
    """State dimension (matches action_dim)"""

    action_horizon: int = 50
    """Number of future actions to predict (action chunk size)"""

    # Token length
    max_token_len: int | None = None
    """Maximum token length (None = auto-detect: 200 for pi0.5, 48 for pi0)"""

    discrete_state_input: bool | None = None
    """Whether to use discrete state input (None = auto-detect based on model_type)"""

    # Image configuration
    image_size: tuple[int, int] = (224, 224)
    """Image size (height, width)"""

    num_cameras: int = 3
    """Number of camera views"""

    camera_names: list[str] = field(default_factory=lambda: ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"])
    """Camera names following OpenPI convention"""

    # Flow matching settings
    num_denoise_steps: int = 10
    """Number of denoising steps for flow matching inference"""

    # Normalization settings
    normalize_actions: bool = True
    """Whether to normalize actions"""

    normalization_mode: Literal["quantile", "zscore"] = "quantile"
    """Normalization mode for actions and states"""

    quantile_low: float = 0.01
    """Lower quantile for normalization"""

    quantile_high: float = 0.99
    """Upper quantile for normalization"""

    # LoRA settings (OpenPI準拠: スケーリング=1.0)
    use_lora: bool = False
    """Whether to use LoRA for finetuning"""

    # VLM (PaliGemma) LoRA settings
    vlm_lora_rank: int = 16
    """LoRA rank for VLM (OpenPI default: 16)"""

    vlm_lora_alpha: int = 16
    """LoRA alpha for VLM (OpenPI default: 16, scaling = 1.0)"""

    # Expert (Gemma-300M) LoRA settings
    expert_lora_rank: int = 32
    """LoRA rank for Expert (OpenPI default: 32)"""

    expert_lora_alpha: int = 32
    """LoRA alpha for Expert (OpenPI default: 32, scaling = 1.0)"""

    lora_dropout: float = 0.05
    """LoRA dropout rate (OpenPI default: 0.05)"""

    lora_rslora: bool = False
    """Use Rank-Stabilized LoRA (scaling = alpha / sqrt(rank))"""

    freeze_vlm: bool = True
    """Whether to freeze VLM (PaliGemma) weights"""

    freeze_action_expert: bool = False
    """Whether to freeze action expert weights"""

    lora_apply_to_vlm: bool = False
    """Whether to apply LoRA to VLM (PaliGemma)"""

    lora_apply_to_expert: bool = True
    """Whether to apply LoRA to Expert (Gemma-300M)"""

    # EMA settings
    ema_decay: float | None = None
    """EMA decay rate. None to disable EMA. OpenPI default: 0.99"""

    # Precision
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    """Training precision (bfloat16 recommended for A100/H100)"""

    # Default prompt
    default_prompt: str = "manipulate objects"
    """Default language instruction for tasks"""

    # Delta actions
    use_delta_actions: bool = False
    """Whether to use delta actions (action = next_state - current_state)"""

    def __post_init__(self):
        """Set derived values based on model_type."""
        # Auto-detect max_token_len based on model_type
        if self.max_token_len is None:
            self.max_token_len = 200 if self.model_type == "pi0.5" else 48

        # Auto-detect discrete_state_input based on model_type
        if self.discrete_state_input is None:
            self.discrete_state_input = self.model_type == "pi0.5"

    @property
    def pi05(self) -> bool:
        """Whether this is Pi0.5 model."""
        return self.model_type == "pi0.5"


@dataclass
class Pi0Config(UnifiedVLAConfig):
    """
    Configuration for Pi0/Pi0.5 training.

    Extends UnifiedVLAConfig with Pi0/Pi0.5-specific parameters.
    Uses PaliGemma + Expert Gemma architecture with flow matching.

    Key differences between Pi0 and Pi0.5:
    - Pi0: Continuous state input, MLP for timestep processing
    - Pi0.5: Discrete state tokens, adaRMSNorm for timestep injection
    """

    pi0: Pi0SpecificConfig = field(default_factory=Pi0SpecificConfig)
    """Pi0/Pi0.5-specific configuration"""

    def __post_init__(self):
        """Initialize and validate configuration."""
        super().__post_init__()

        # Set backend name based on model type
        if self.pi0.model_type == "pi0.5":
            self.backend = "pi0.5"
        else:
            self.backend = "pi0"

        # Apply pi0 post_init to set derived values
        self.pi0.__post_init__()

        # Store Pi0-specific config in backend_config dict for serialization
        self.backend_config = {
            "model_type": self.pi0.model_type,
            "paligemma_variant": self.pi0.paligemma_variant,
            "action_expert_variant": self.pi0.action_expert_variant,
            "pretrained_checkpoint": self.pi0.pretrained_checkpoint,
            "use_pretrained": self.pi0.use_pretrained,
            "paligemma_pretrained_id": self.pi0.paligemma_pretrained_id,
            "openpi_checkpoint": self.pi0.openpi_checkpoint,
            "action_dim": self.pi0.action_dim,
            "state_dim": self.pi0.state_dim,
            "action_horizon": self.pi0.action_horizon,
            "max_token_len": self.pi0.max_token_len,
            "discrete_state_input": self.pi0.discrete_state_input,
            "image_size": self.pi0.image_size,
            "num_cameras": self.pi0.num_cameras,
            "camera_names": self.pi0.camera_names,
            "num_denoise_steps": self.pi0.num_denoise_steps,
            "normalize_actions": self.pi0.normalize_actions,
            "normalization_mode": self.pi0.normalization_mode,
            "quantile_low": self.pi0.quantile_low,
            "quantile_high": self.pi0.quantile_high,
            "use_lora": self.pi0.use_lora,
            "vlm_lora_rank": self.pi0.vlm_lora_rank,
            "vlm_lora_alpha": self.pi0.vlm_lora_alpha,
            "expert_lora_rank": self.pi0.expert_lora_rank,
            "expert_lora_alpha": self.pi0.expert_lora_alpha,
            "lora_dropout": self.pi0.lora_dropout,
            "lora_rslora": self.pi0.lora_rslora,
            "lora_apply_to_vlm": self.pi0.lora_apply_to_vlm,
            "lora_apply_to_expert": self.pi0.lora_apply_to_expert,
            "freeze_vlm": self.pi0.freeze_vlm,
            "freeze_action_expert": self.pi0.freeze_action_expert,
            "ema_decay": self.pi0.ema_decay,
            "precision": self.pi0.precision,
            "default_prompt": self.pi0.default_prompt,
            "use_delta_actions": self.pi0.use_delta_actions,
        }

    @property
    def crane_x7_action_dim(self) -> int:
        """Get CRANE-X7's native action dimension (8 DOF)."""
        return 8

    @property
    def model_action_dim(self) -> int:
        """Get model's expected action dimension (32 for Pi0)."""
        return self.pi0.action_dim

    @property
    def action_horizon(self) -> int:
        """Pi0 predicts action chunks."""
        return self.pi0.action_horizon

    @property
    def is_pi05(self) -> bool:
        """Check if this is Pi0.5 configuration."""
        return self.pi0.pi05
