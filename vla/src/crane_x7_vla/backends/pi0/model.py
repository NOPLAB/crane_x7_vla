# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Pi0/Pi0.5 Model Implementation.

Based on OpenPI's architecture with joint layer-by-layer attention between
PaliGemma (VLM) and Expert Gemma (Action Expert).

Key components:
- PaliGemmaWithExpertModel from models_pytorch/gemma_pytorch.py
- Flow matching for action prediction
- Support for both Pi0 (continuous state) and Pi0.5 (discrete state + adaRMSNorm)
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from .models_pytorch.pi0_pytorch import get_gemma_config


logger = logging.getLogger(__name__)


@dataclass
class Pi0ModelConfig:
    """Configuration for Pi0 model."""

    pi05: bool = False
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    dtype: str = "bfloat16"
    use_pretrained: bool = True
    paligemma_pretrained_id: str | None = None  # None = use default
    openpi_checkpoint: str | None = None  # OpenPI checkpoint name (e.g., "pi0_base", "pi05_base")


# =============================================================================
# Utility Functions
# =============================================================================


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "cpu" and target_dtype == torch.bfloat16:
        return torch.float32
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    """Compute sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape (batch_size,)")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    """Sample from Beta distribution."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Create 2D attention masks from padding and attention masks.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention patterns.

    Args:
        pad_masks: bool[B, N] - True if part of the input, False if padding
        att_masks: int32[B, N] - 1 where previous tokens cannot attend, 0 for shared attention

    Returns:
        bool[B, N, N] - 2D attention mask
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# =============================================================================
# Pi0 Model
# =============================================================================


class Pi0Model(nn.Module):
    """Pi0/Pi0.5 Model for action prediction.

    Implements flow matching for action chunk prediction using:
    - PaliGemma for vision-language understanding
    - Expert Gemma for action prediction
    - Optional adaRMSNorm for Pi0.5 timestep injection

    Key differences between Pi0 and Pi0.5:
    - Pi0: Continuous state input via projection, timestep mixed via MLP
    - Pi0.5: Discrete state tokens, timestep via adaRMSNorm conditioning

    Uses PaliGemmaWithExpertModel from models_pytorch/gemma_pytorch.py
    which implements OpenPI's joint layer-by-layer attention.
    """

    def __init__(self, config: Pi0ModelConfig):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        # Get Gemma configs from models_pytorch
        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        # Determine if we should use OpenPI checkpoint
        openpi_checkpoint = config.openpi_checkpoint

        # Create PaliGemma with Expert using models_pytorch implementation
        use_adarms = [False, True] if config.pi05 else [False, False]
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=use_adarms,
            precision=config.dtype,
            use_pretrained=config.use_pretrained,
            paligemma_pretrained_id=config.paligemma_pretrained_id,
            openpi_checkpoint=openpi_checkpoint,
        )

        # Action projection layers
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # Timestep processing layers (different for Pi0 vs Pi0.5)
        if self.pi05:
            # Pi0.5: Time MLP for adaRMSNorm conditioning
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # Pi0: State projection and action-time MLP
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gradient_checkpointing_enabled = False

        # Load OpenPI checkpoint if specified
        if openpi_checkpoint is not None:
            self._load_openpi_checkpoint(openpi_checkpoint)

    def _load_openpi_checkpoint(self, checkpoint_name: str) -> None:
        """Load weights from an OpenPI checkpoint.

        Args:
            checkpoint_name: Name of the OpenPI checkpoint (e.g., "pi0_base", "pi05_base")
        """
        from .checkpoint_utils import load_openpi_checkpoint, map_openpi_to_crane_x7

        logger.info(f"Loading OpenPI checkpoint: {checkpoint_name}")

        # Load the checkpoint
        state_dict = load_openpi_checkpoint(checkpoint_name, device="cpu")

        # Map keys if needed
        mapped_state_dict = map_openpi_to_crane_x7(state_dict, pi05=self.pi05)

        # Load weights with strict=False to allow for missing/extra keys
        missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading OpenPI checkpoint: {missing_keys[:10]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading OpenPI checkpoint: {unexpected_keys[:10]}...")

        logger.info(f"Successfully loaded OpenPI checkpoint: {checkpoint_name}")

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for Pi0 model")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for Pi0 model")

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        """Convert 2D attention masks to 4D format for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def sample_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample Gaussian noise."""
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from Beta distribution."""
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language tokens for prefix.

        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token IDs [B, seq_len]
            lang_masks: Language attention masks [B, seq_len]

        Returns:
            Tuple of (embeddings, pad_masks, att_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs  # Image tokens attend to each other

        # Process language tokens
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]  # Full attention between image and language

        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: torch.Tensor | None,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed state, noisy actions, and timestep for suffix.

        Args:
            state: Robot state [B, state_dim]. Required for Pi0, None for Pi0.5
                   (Pi0.5 includes state in language tokens)
            noisy_actions: Noisy action chunk [B, horizon, action_dim]
            timestep: Flow matching timestep [B]

        Returns:
            Tuple of (embeddings, pad_masks, att_masks, adarms_cond)
        """
        embs = []
        pad_masks = []
        att_masks = []

        device = timestep.device
        bsize = noisy_actions.shape[0]

        # Pi0: Add state token (Pi0.5 includes state in language tokens)
        if not self.pi05:
            if state is None:
                raise ValueError("state is required for Pi0 mode (discrete_state_input=False)")
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self.state_proj(state)[:, None, :]
            embs.append(state_emb)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]  # State doesn't attend to previous tokens

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.to(dtype=timestep.dtype)

        # Project noisy actions
        action_emb = self.action_in_proj(noisy_actions)

        # Process timestep differently for Pi0 vs Pi0.5
        if not self.pi05:
            # Pi0: Mix timestep with actions via MLP
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)
            action_time_emb = self.action_time_mlp_out(action_time_emb)
            adarms_cond = None
        else:
            # Pi0.5: Time MLP for adaRMSNorm conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        pad_masks.append(torch.ones(bsize, self.action_horizon, dtype=torch.bool, device=device))
        att_masks += [1] + ([0] * (self.action_horizon - 1))  # Causal attention for actions

        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training with flow matching loss.

        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token IDs [B, seq_len]
            lang_masks: Language attention masks [B, seq_len]
            state: Robot state [B, state_dim]. Required for Pi0, None for Pi0.5
            actions: Target action chunk [B, horizon, action_dim]
            noise: Optional pre-sampled noise
            time: Optional pre-sampled timesteps

        Returns:
            Flow matching loss [B, horizon, action_dim]
        """
        device = actions.device
        bsize = actions.shape[0]

        # Sample noise and time if not provided
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)

        # Compute noisy actions: x_t = t * noise + (1 - t) * actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Target velocity: u_t = noise - actions
        u_t = noise - actions

        # Embed prefix (images + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Embed suffix (state + noisy actions + time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Match dtypes
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Create attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Forward through model (joint layer-by-layer via PaliGemmaWithExpertModel)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract action predictions from suffix output
        suffix_out = outputs[1]
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # Compute MSE loss
        return F.mse_loss(u_t, v_t, reduction="none")

    def _get_model_dtype(self) -> torch.dtype:
        """Get model weight dtype (cached)."""
        if not hasattr(self, "_cached_model_dtype"):
            self._cached_model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
        return self._cached_model_dtype

    def _denoise_step(
        self,
        state: torch.Tensor | None,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: list,
    ) -> torch.Tensor:
        """Single denoising step using KV cache.

        This matches OpenPI's inference approach using KV caching.

        Args:
            state: Robot state [B, state_dim]. Required for Pi0, None for Pi0.5
            x_t: Current noisy actions
            timestep: Current timestep
            prefix_pad_masks: Prefix padding masks
            past_key_values: KV cache from prefix computation
        """
        # Embed suffix for current x_t
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Match dtype
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Create attention masks for suffix attending to prefix + suffix
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # Suffix can attend to all prefix tokens
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # Suffix attention to itself
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # Full mask: [prefix, suffix]
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs for suffix
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Forward through Expert Gemma with KV cache
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract velocity prediction
        suffix_out = outputs[1]
        suffix_out = suffix_out[:, -self.action_horizon :]
        return self.action_out_proj(suffix_out.float())

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Sample actions using flow matching ODE integration.

        When use_kv_cache=True (default), computes prefix KV cache once and
        reuses it for all denoising steps, significantly improving inference speed.

        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token IDs [B, seq_len]
            lang_masks: Language attention masks [B, seq_len]
            state: Robot state [B, state_dim]. Required for Pi0, None for Pi0.5
            num_steps: Number of integration steps (default: 10)
            noise: Optional initial noise
            use_kv_cache: Whether to use KV caching for inference (default: True)

        Returns:
            Sampled action chunk [B, horizon, action_dim]
        """
        # Determine device and batch size
        if state is not None:
            device = state.device
            bsize = state.shape[0]
        else:
            # Pi0.5: get device and batch from lang_tokens
            device = lang_tokens.device
            bsize = lang_tokens.shape[0]

        # Initialize from noise
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.action_dim)
            noise = self.sample_noise(actions_shape, device)
        x_t = noise

        # Embed prefix (images + language) - reused for all steps
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Convert prefix to model dtype
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Set eager attention for inference
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        # Euler integration
        dt = -1.0 / num_steps
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        if use_kv_cache:
            # Compute prefix KV cache once
            kv_cache = self._compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self._denoise_step_with_cache(
                    state,
                    x_t,
                    expanded_time,
                    prefix_pad_masks,
                    kv_cache,
                )

                x_t = x_t + dt * v_t
                time = time + dt
        else:
            # Full computation each step (slower but more accurate for debugging)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self._denoise_step_full(
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    state,
                    x_t,
                    expanded_time,
                )

                x_t = x_t + dt * v_t
                time = time + dt

        return x_t

    def _compute_prefix_kv_cache(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> list:
        """Compute KV cache for prefix (images + language).

        Args:
            prefix_embs: Prefix embeddings [B, prefix_len, dim]
            prefix_pad_masks: Prefix padding masks [B, prefix_len]
            prefix_att_masks: Prefix attention masks [B, prefix_len]

        Returns:
            KV cache list for reuse in denoising steps
        """
        # Create prefix attention mask
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prepare 4D attention mask
        att_2d_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Forward through model with cache enabled
        _, kv_cache = self.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        return kv_cache

    def _denoise_step_with_cache(
        self,
        state: torch.Tensor | None,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        kv_cache: list,
    ) -> torch.Tensor:
        """Single denoising step using cached prefix KV.

        Args:
            state: Robot state [B, state_dim]. Required for Pi0, None for Pi0.5
            x_t: Current noisy actions
            timestep: Current timestep
            prefix_pad_masks: Prefix padding masks
            kv_cache: Cached KV from prefix computation

        Returns:
            Velocity prediction [B, horizon, action_dim]
        """
        # Embed suffix for current x_t
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Match dtype
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Build attention mask: suffix attends to [prefix, suffix]
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # Suffix can attend to all prefix tokens
        prefix_att_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # Suffix attention to itself
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # Full attention: [prefix, suffix]
        full_att_2d = torch.cat([prefix_att_2d, suffix_att_2d], dim=2)

        # Position IDs for suffix (continue from prefix)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare 4D attention mask
        full_att_2d_4d = self._prepare_attention_masks_4d(full_att_2d)

        # Forward with KV cache
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_4d,
            position_ids=position_ids,
            past_key_values=kv_cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,  # Don't update cache for suffix
            adarms_cond=[None, adarms_cond],
        )

        # Extract velocity prediction
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        return v_t

    def _denoise_step_full(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        state: torch.Tensor | None,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step processing full prefix+suffix sequence.

        This processes both prefix and suffix through joint attention,
        matching the training-time forward pass.

        Args:
            prefix_embs: Pre-computed prefix embeddings
            prefix_pad_masks: Prefix padding masks
            prefix_att_masks: Prefix attention masks
            state: Robot state [B, state_dim]. Required for Pi0, None for Pi0.5
            x_t: Current noisy actions
            timestep: Current timestep
        """
        # Embed suffix for current x_t
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Match dtype
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Build attention masks - MUST match training-time forward() method
        # Concatenate masks and use make_att_2d_masks for consistency
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        # Position IDs
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare 4D attention mask
        full_att_2d_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Forward through both models with joint attention
        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract velocity prediction from suffix output
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        return v_t
