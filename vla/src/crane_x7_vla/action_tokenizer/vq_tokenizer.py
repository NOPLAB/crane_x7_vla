# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
VQ-based Action Tokenizer for MiniVLA.

Converts action chunks to/from token IDs using Residual VQ,
enabling autoregressive prediction of action sequences in VLA models.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from crane_x7_vla.action_tokenizer.vq import ResidualVQ


class VQActionTokenizer:
    """
    Action tokenizer using Vector Quantization for action chunking.

    Compresses action chunks (H, A) into a sequence of discrete token IDs
    that can be predicted autoregressively by the VLA model. Each token
    corresponds to a codebook index in the Residual VQ.

    This enables MiniVLA to predict multiple future actions efficiently
    by outputting n_groups tokens instead of H*A continuous values.
    """

    def __init__(
        self,
        vq_path: Optional[Union[str, Path]] = None,
        vq_model: Optional[ResidualVQ] = None,
        use_extra_tokens: bool = True,
        extra_token_offset: int = 0,
        vocab_size: int = 151936,  # Qwen2.5 vocab size
    ):
        """
        Initialize VQActionTokenizer.

        Args:
            vq_path: Path to pre-trained ResidualVQ model
            vq_model: Pre-loaded ResidualVQ model (alternative to vq_path)
            use_extra_tokens: If True, use extra tokens for action bins
                             (recommended for Qwen2.5 to avoid vocab conflicts)
            extra_token_offset: Offset for extra tokens in vocabulary
            vocab_size: Vocabulary size of the base LLM
        """
        if vq_model is not None:
            self.vq = vq_model
        elif vq_path is not None:
            self.vq = ResidualVQ.load(vq_path)
        else:
            self.vq = None

        self.use_extra_tokens = use_extra_tokens
        self.extra_token_offset = extra_token_offset
        self.vocab_size = vocab_size

        # Token ID ranges
        if use_extra_tokens:
            # Use extra tokens added to vocabulary
            self.action_token_begin_idx = vocab_size + extra_token_offset
        else:
            # Use last tokens in vocabulary (OpenVLA style)
            self.action_token_begin_idx = vocab_size - 256

    @property
    def n_embed(self) -> int:
        """Number of embeddings per codebook."""
        if self.vq is None:
            return 256
        return self.vq.n_embed

    @property
    def n_groups(self) -> int:
        """Number of VQ groups (tokens per action chunk)."""
        if self.vq is None:
            return 7
        return self.vq.n_groups

    @property
    def action_horizon(self) -> int:
        """Action chunk horizon."""
        if self.vq is None:
            return 8
        return self.vq.action_horizon

    @property
    def action_dim(self) -> int:
        """Action dimension per timestep."""
        if self.vq is None:
            return 8
        return self.vq.action_dim

    def encode(self, actions: np.ndarray) -> List[int]:
        """
        Encode action chunk to token IDs.

        Args:
            actions: Action chunk of shape (H, A) or (B, H, A)

        Returns:
            List of token IDs of length n_groups (or B*n_groups if batched)
        """
        if self.vq is None:
            raise ValueError("VQ model not loaded. Provide vq_path or vq_model.")

        # Convert to tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        # Add batch dim if needed
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)

        # Encode to indices
        with torch.no_grad():
            indices = self.vq.encode(actions)  # (B, n_groups)

        # Convert to token IDs
        token_ids = indices.cpu().numpy() + self.action_token_begin_idx

        # Flatten for single sample
        if token_ids.shape[0] == 1:
            return token_ids[0].tolist()

        return token_ids.flatten().tolist()

    def decode(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Decode token IDs to action chunk.

        Args:
            token_ids: Token IDs of length n_groups (or B*n_groups if batched)

        Returns:
            Action chunk of shape (H, A) or (B, H, A)
        """
        if self.vq is None:
            raise ValueError("VQ model not loaded. Provide vq_path or vq_model.")

        # Convert to numpy
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)

        # Convert to codebook indices
        indices = token_ids - self.action_token_begin_idx

        # Reshape if batched
        if len(indices) > self.n_groups:
            indices = indices.reshape(-1, self.n_groups)
        else:
            indices = indices.reshape(1, -1)

        # Decode
        indices_tensor = torch.from_numpy(indices).long()
        with torch.no_grad():
            actions = self.vq.decode(indices_tensor)  # (B, H, A)

        actions = actions.cpu().numpy()

        # Remove batch dim if single sample
        if actions.shape[0] == 1:
            return actions[0]

        return actions

    def encode_to_tokens(
        self,
        actions: np.ndarray,
        tokenizer: Optional[object] = None,
    ) -> Tuple[List[int], List[str]]:
        """
        Encode actions to token IDs and token strings.

        Args:
            actions: Action chunk of shape (H, A)
            tokenizer: HuggingFace tokenizer (optional, for token strings)

        Returns:
            Tuple of (token_ids, token_strings)
        """
        token_ids = self.encode(actions)

        if tokenizer is not None:
            token_strings = [tokenizer.decode([tid]) for tid in token_ids]
        else:
            token_strings = [f"<action_{i}_{tid}>" for i, tid in enumerate(token_ids)]

        return token_ids, token_strings

    def decode_from_tokens(
        self,
        token_ids: List[int],
        tokenizer: Optional[object] = None,
    ) -> np.ndarray:
        """
        Decode token IDs to actions.

        Args:
            token_ids: Token IDs from model output
            tokenizer: HuggingFace tokenizer (optional, unused)

        Returns:
            Action chunk of shape (H, A)
        """
        return self.decode(token_ids)

    def get_action_token_range(self) -> Tuple[int, int]:
        """
        Get the range of valid action token IDs.

        Returns:
            Tuple of (begin_idx, end_idx)
        """
        begin_idx = self.action_token_begin_idx
        end_idx = begin_idx + self.n_embed
        return begin_idx, end_idx

    def is_action_token(self, token_id: int) -> bool:
        """Check if a token ID is an action token."""
        begin_idx, end_idx = self.get_action_token_range()
        return begin_idx <= token_id < end_idx


class BinActionTokenizer:
    """
    Simple binning-based action tokenizer (OpenVLA style).

    Discretizes continuous actions into bins and maps each bin
    to a token ID. Used as fallback when VQ is not enabled.
    """

    def __init__(
        self,
        action_dim: int = 8,
        n_bins: int = 256,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        use_extra_tokens: bool = True,
        vocab_size: int = 151936,
    ):
        """
        Initialize BinActionTokenizer.

        Args:
            action_dim: Action dimension
            n_bins: Number of bins for discretization
            action_range: (min, max) range for actions
            use_extra_tokens: Use extra tokens for bins
            vocab_size: Base vocabulary size
        """
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.action_min, self.action_max = action_range
        self.use_extra_tokens = use_extra_tokens
        self.vocab_size = vocab_size

        if use_extra_tokens:
            self.action_token_begin_idx = vocab_size
        else:
            self.action_token_begin_idx = vocab_size - n_bins

        # Precompute bin edges
        self.bin_edges = np.linspace(
            self.action_min, self.action_max, n_bins + 1
        )

    def encode(self, actions: np.ndarray) -> List[int]:
        """
        Encode single-step action to token IDs.

        Args:
            actions: Action of shape (A,)

        Returns:
            List of token IDs of length action_dim
        """
        # Clip to valid range
        actions = np.clip(actions, self.action_min, self.action_max)

        # Discretize to bins
        bin_indices = np.digitize(actions, self.bin_edges[1:-1])

        # Convert to token IDs
        token_ids = bin_indices + self.action_token_begin_idx

        return token_ids.tolist()

    def decode(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Decode token IDs to continuous action.

        Args:
            token_ids: Token IDs of length action_dim

        Returns:
            Action of shape (A,)
        """
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)

        # Convert to bin indices
        bin_indices = token_ids - self.action_token_begin_idx

        # Convert to bin centers
        bin_width = (self.action_max - self.action_min) / self.n_bins
        actions = self.action_min + (bin_indices + 0.5) * bin_width

        return actions
