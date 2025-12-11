# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
OpenVLA-OFT model components.

This module implements the core building blocks for OpenVLA-OFT:
- MLPResNet: MLP with residual connections
- L1RegressionActionHead: Predicts continuous actions via L1 regression
- ProprioProjector: Projects proprioceptive state to LLM embedding space
- FiLMedVisionTransformerBlock: ViT block with FiLM modulation
- FiLMedVisionBackbone: Vision backbone wrapper with FiLM support

Based on the OpenVLA-OFT paper: https://arxiv.org/abs/2502.19645
Reference implementation: https://github.com/moojink/openvla-oft
"""

from functools import partial
from typing import List, Optional, Union, Sequence, Tuple, Any, Callable

import torch
import torch.nn as nn


class MLPResNetBlock(nn.Module):
    """
    One MLP ResNet block with a residual connection.

    Architecture follows "Pre-Layer Normalization" pattern:
    https://arxiv.org/pdf/2002.04745.pdf

    output = x + FFN(LayerNorm(x))
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        """
        Initialize MLPResNet block.

        Args:
            dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (batch_size, hidden_dim)

        Returns:
            Output tensor (batch_size, hidden_dim)
        """
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """
    MLP with residual connection blocks.

    Architecture:
    1. LayerNorm -> Linear(input_dim, hidden_dim) -> ReLU
    2. N x MLPResNetBlock
    3. LayerNorm -> Linear(hidden_dim, output_dim)
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        """
        Initialize MLPResNet.

        Args:
            num_blocks: Number of residual blocks
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.mlp_resnet_blocks = nn.ModuleList([
            MLPResNetBlock(dim=hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLPResNet.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        for block in self.mlp_resnet_blocks:
            x = block(x)

        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """
    L1 Regression Action Head for continuous action prediction.

    Takes LLM hidden states corresponding to action tokens and predicts
    continuous actions via L1 regression, enabling parallel decoding
    (much faster than autoregressive token generation).

    Architecture:
    - Reshape hidden states: (B, chunk_len * action_dim, hidden_dim)
                          -> (B, chunk_len, action_dim * hidden_dim)
    - MLPResNet: (B, chunk_len, action_dim * hidden_dim)
              -> (B, chunk_len, action_dim)
    """

    def __init__(
        self,
        llm_hidden_dim: int = 4096,
        action_dim: int = 8,
        action_horizon: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize L1 Regression Action Head.

        Args:
            llm_hidden_dim: LLM hidden dimension (Llama-2 7B: 4096)
            action_dim: Action dimension per timestep (CRANE-X7: 8)
            action_horizon: Number of future actions to predict (default: 8)
            num_blocks: Number of residual blocks in MLPResNet
            dropout: Dropout rate
        """
        super().__init__()
        self.llm_hidden_dim = llm_hidden_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Input: flattened hidden states for all action tokens in a chunk
        # For each timestep, we have action_dim hidden states
        input_dim = llm_hidden_dim * action_dim

        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=llm_hidden_dim,
            output_dim=action_dim,
            dropout=dropout,
        )

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict action chunk from LLM hidden states.

        Args:
            actions_hidden_states: Hidden states for action tokens
                Shape: (batch_size, action_horizon * action_dim, llm_hidden_dim)

        Returns:
            Predicted actions: (batch_size, action_horizon, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]

        # Reshape: (B, chunk_len * action_dim, hidden_dim)
        #       -> (B, chunk_len, action_dim * hidden_dim)
        x = actions_hidden_states.reshape(
            batch_size, self.action_horizon, -1
        )

        # Predict actions for each timestep
        # (B, chunk_len, action_dim * hidden_dim) -> (B, chunk_len, action_dim)
        actions = self.model(x)

        return actions

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for compatibility with OpenVLA-OFT interface."""
        return self.forward(actions_hidden_states)


class ProprioProjector(nn.Module):
    """
    Projector for proprioceptive (robot state) input.

    Projects robot state into LLM embedding space using a 2-layer MLP.
    The output embedding is appended to visual embeddings.

    Architecture:
    proprio -> Linear -> GELU -> Linear -> projected_proprio
    """

    def __init__(
        self,
        proprio_dim: int = 8,
        llm_dim: int = 4096,
    ):
        """
        Initialize Proprio Projector.

        Args:
            proprio_dim: Proprioceptive state dimension (CRANE-X7: 8)
            llm_dim: LLM hidden dimension (Llama-2 7B: 4096)
        """
        super().__init__()
        self.proprio_dim = proprio_dim
        self.llm_dim = llm_dim

        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Project proprioceptive state to LLM embedding space.

        Args:
            proprio: Proprioceptive state (batch_size, proprio_dim)

        Returns:
            Projected features (batch_size, llm_dim)
        """
        x = self.fc1(proprio)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class FiLMedVisionTransformerBlock(nn.Module):
    """
    Vision Transformer block wrapped with FiLM modulation.

    FiLM (Feature-wise Linear Modulation) modulates visual features
    using language-conditioned gamma and beta:

    output = (1 + gamma) * x + beta

    where gamma and beta are learned projections of the average
    language embedding. The (1 + gamma) formulation ensures identity
    transformation at initialization (when gamma ≈ 0).

    Reference: https://arxiv.org/pdf/1709.07871.pdf (Section 7.2)
    """

    def __init__(
        self,
        block: nn.Module,
        vision_dim: int,
        llm_dim: int,
    ):
        """
        Initialize FiLM ViT block wrapper.

        Args:
            block: Original Vision Transformer block to wrap
            vision_dim: Number of hidden dimensions in visual embeddings
            llm_dim: Number of hidden dimensions in language embeddings
        """
        super().__init__()
        self.block = block
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim

        # Initialize gamma and beta projectors
        # These project average language embedding to visual space
        self.scale = nn.Linear(llm_dim, vision_dim)  # gamma
        self.shift = nn.Linear(llm_dim, vision_dim)  # beta

    def forward(
        self,
        x: torch.Tensor,
        average_language_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with FiLM modulation.

        Args:
            x: Visual input embeddings (batch_size, seq_len, vision_dim)
            average_language_embedding: Average language embedding (batch_size, llm_dim)

        Returns:
            Modulated visual features (batch_size, seq_len, vision_dim)
        """
        # Project language embedding to get gamma and beta
        gamma = self.scale(average_language_embedding)  # (B, vision_dim)
        beta = self.shift(average_language_embedding)   # (B, vision_dim)

        # Pass through attention portion of original block
        # Assuming standard timm ViT block structure
        x = x + self.block.drop_path1(
            self.block.ls1(self.block.attn(self.block.norm1(x)))
        )

        # Apply FiLM modulation
        # gamma and beta: (B, D) -> (B, 1, D) for broadcasting
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # Pass through feedforward portion of original block
        x = x + self.block.drop_path2(
            self.block.ls2(self.block.mlp(self.block.norm2(x)))
        )

        return x


class NullVisionTransformerBlockWrapper(nn.Module):
    """
    Null wrapper for ViT blocks (passes through without modification).

    Useful for applying FiLM only to specific layers.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(
        self,
        x: torch.Tensor,
        average_language_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Just forward through original block, ignoring language."""
        return self.block(x)


def _unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """Utility function for monkey-patching functions that return tuples."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper


class FiLMedVisionBackbone(nn.Module):
    """
    FiLM-conditioned vision backbone.

    Wraps PrismaticVisionBackbone to accept language embeddings
    and apply FiLM modulation at each ViT block.

    Supports:
    - Single image input
    - Multi-image input (e.g., third-person + wrist cameras)
    - Fused vision backbone (SigLIP + DINOv2)
    """

    def __init__(
        self,
        vision_backbone: nn.Module,
        llm_dim: int = 4096,
    ):
        """
        Initialize FiLM wrapper.

        Args:
            vision_backbone: Base PrismaticVisionBackbone to wrap
            llm_dim: Dimension of language model embeddings (Llama-2 7B: 4096)
        """
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim

        # Wrap vision transformers with FiLM
        self._wrap_vit(self.vision_backbone.featurizer)  # Primary (e.g., SigLIP)

        if hasattr(self.vision_backbone, 'use_fused_vision_backbone') and \
           self.vision_backbone.use_fused_vision_backbone:
            self._wrap_vit(self.vision_backbone.fused_featurizer)  # Secondary (e.g., DINOv2)

    def _wrap_vit(self, vit: nn.Module) -> None:
        """
        Wrap individual vision transformer blocks with FiLM.

        Args:
            vit: Vision transformer module to wrap
        """
        # Wrap each block with FiLM
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(
                    block=block,
                    vision_dim=vit.num_features,
                    llm_dim=self.llm_dim,
                )
            )
        vit.blocks = nn.Sequential(*block_wrappers)

        # Store original forward methods
        original_intermediate_layers = vit._intermediate_layers
        original_get_intermediate_layers = vit.get_intermediate_layers

        # Override _intermediate_layers to accept language embeddings
        def _intermediate_layers_with_lang(
            x: torch.Tensor,
            language_embeddings: torch.Tensor,
            n: Union[int, Sequence] = 1,
        ):
            outputs, num_blocks = [], len(vit.blocks)
            take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

            # Forward pass
            x = vit.patch_embed(x)
            x = vit._pos_embed(x)
            x = vit.patch_drop(x)
            x = vit.norm_pre(x)

            for i, blk in enumerate(vit.blocks):
                x = blk(x, language_embeddings)  # Pass language to each block
                if i in take_indices:
                    outputs.append(x)

            return outputs

        vit._intermediate_layers = _intermediate_layers_with_lang

        # Override get_intermediate_layers
        def get_intermediate_layers_with_lang(
            x: torch.Tensor,
            language_embeddings: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
        ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
            outputs = vit._intermediate_layers(x, language_embeddings, n)
            if norm:
                outputs = [vit.norm(out) for out in outputs]
            prefix_tokens = [out[:, 0:vit.num_prefix_tokens] for out in outputs]
            outputs = [out[:, vit.num_prefix_tokens:] for out in outputs]

            if reshape:
                grid_size = vit.patch_embed.grid_size
                outputs = [
                    out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                    .permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]

            if return_prefix_tokens:
                return tuple(zip(outputs, prefix_tokens))
            return tuple(outputs)

        vit.get_intermediate_layers = get_intermediate_layers_with_lang

        # Override forward to use the new get_intermediate_layers
        vit.forward = _unpack_tuple(
            partial(vit.get_intermediate_layers, n={len(vit.blocks) - 2})
        )

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(
        self,
        pixel_values: torch.Tensor,
        language_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with FiLM language conditioning.

        Args:
            pixel_values: Pixels for input image(s) (B, C, H, W) or (B, num_images*C, H, W)
            language_embeddings: Language embeddings (B, seq_len, llm_dim)

        Returns:
            Visual features with language conditioning (B, num_patches, vision_dim)
        """
        # Average language embeddings for FiLM conditioning
        average_language_embedding = language_embeddings.mean(dim=1)  # (B, llm_dim)

        num_images = self.get_num_images_in_input()
        use_fused = hasattr(self.vision_backbone, 'use_fused_vision_backbone') and \
                    self.vision_backbone.use_fused_vision_backbone

        if num_images == 1:
            if not use_fused:
                return self.vision_backbone.featurizer(
                    pixel_values, average_language_embedding
                )

            # Fused backbone: split channels (3 for SigLIP + 3 for DINOv2)
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.vision_backbone.featurizer(img, average_language_embedding)
            patches_fused = self.vision_backbone.fused_featurizer(
                img_fused, average_language_embedding
            )
            return torch.cat([patches, patches_fused], dim=2)

        else:
            # Multi-image input
            assert use_fused, "Multi-image inputs require using fused backbone!"

            # Split into individual images (each with 6 channels)
            images = torch.split(pixel_values, [6] * num_images, dim=1)

            all_patches = []
            for img in images:
                # Split each image into SigLIP and DINOv2 channels
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                patches = self.vision_backbone.featurizer(
                    img_regular, average_language_embedding
                )
                patches_fused = self.vision_backbone.fused_featurizer(
                    img_fused, average_language_embedding
                )

                # Concatenate along hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along patch dimension
            return torch.cat(all_patches, dim=1)
