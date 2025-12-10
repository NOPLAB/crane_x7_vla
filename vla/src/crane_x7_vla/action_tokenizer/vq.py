# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Residual Vector Quantization (VQ) implementation for action chunking.

This module implements Residual VQ based on the VQ-BeT architecture,
which compresses action chunks (H, A) into discrete codebook indices
for autoregressive prediction in VLA models.

References:
- VQ-BeT: https://arxiv.org/abs/2403.03181
- MiniVLA: https://ai.stanford.edu/blog/minivla/
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """
    Single-level Vector Quantization layer.

    Implements the basic VQ operation with EMA codebook updates
    and commitment loss.
    """

    def __init__(
        self,
        dim: int,
        n_embed: int = 256,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_weight: float = 0.25,
    ):
        """
        Initialize VectorQuantize layer.

        Args:
            dim: Embedding dimension
            n_embed: Number of embeddings in codebook
            decay: EMA decay rate for codebook updates
            eps: Epsilon for numerical stability
            commitment_weight: Weight for commitment loss
        """
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment_weight = commitment_weight

        # Codebook embeddings
        embed = torch.randn(n_embed, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: quantize input to nearest codebook entry.

        Args:
            x: Input tensor of shape (B, D) or (B, T, D)

        Returns:
            Tuple of (quantized, indices, loss)
            - quantized: Quantized tensor, same shape as input
            - indices: Codebook indices of shape (B,) or (B, T)
            - loss: VQ loss (commitment + optional entropy)
        """
        # Flatten if needed
        flatten = x.dim() == 3
        if flatten:
            B, T, D = x.shape
            x = x.reshape(-1, D)

        # Compute distances to codebook entries
        # dist[i, j] = ||x[i] - embed[j]||^2
        dist = (
            x.pow(2).sum(dim=1, keepdim=True)
            - 2 * x @ self.embed.t()
            + self.embed.pow(2).sum(dim=1, keepdim=True).t()
        )

        # Find nearest codebook entry
        indices = dist.argmin(dim=1)

        # Quantize
        quantized = F.embedding(indices, self.embed)

        # Compute loss
        loss = self.commitment_weight * F.mse_loss(x.detach(), quantized)
        loss += F.mse_loss(x, quantized.detach())

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # Update codebook with EMA (only during training)
        if self.training:
            self._update_codebook(x, indices)

        # Restore shape
        if flatten:
            quantized = quantized.view(B, T, D)
            indices = indices.view(B, T)

        return quantized, indices, loss

    def _update_codebook(self, x: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook using EMA."""
        # One-hot encoding
        encodings = F.one_hot(indices, self.n_embed).float()

        # Update cluster sizes
        self.cluster_size.data.mul_(self.decay).add_(
            encodings.sum(0), alpha=1 - self.decay
        )

        # Update embedding averages
        embed_sum = encodings.t() @ x
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Normalize to get new embeddings
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embed.data.copy_(embed_normalized)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to codebook indices."""
        _, indices, _ = self(x)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to embeddings."""
        return F.embedding(indices, self.embed)


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization for action chunking.

    Compresses action chunks (H, A) into a sequence of codebook indices
    using multiple sequential VQ layers. Each layer quantizes the residual
    from the previous layer, enabling fine-grained reconstruction.

    This is the core component for VQ-based action chunking in MiniVLA.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 8,
        action_horizon: int = 8,
        n_embed: int = 256,
        n_latent: int = 512,
        n_groups: int = 7,
        commitment_weight: float = 0.25,
    ):
        """
        Initialize ResidualVQ.

        Args:
            input_dim: Input dimension (action_dim * action_horizon)
            action_dim: Action dimension per timestep (8 for CRANE-X7)
            action_horizon: Number of future actions (chunk size)
            n_embed: Number of embeddings per codebook
            n_latent: Latent dimension for encoder/decoder
            n_groups: Number of residual VQ groups (sequential codebooks)
            commitment_weight: Weight for commitment loss
        """
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_embed = n_embed
        self.n_latent = n_latent
        self.n_groups = n_groups

        # Encoder: action chunk -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
        )

        # Residual VQ layers
        self.vq_layers = nn.ModuleList([
            VectorQuantize(
                dim=n_latent,
                n_embed=n_embed,
                commitment_weight=commitment_weight,
            )
            for _ in range(n_groups)
        ])

        # Decoder: latent -> action chunk
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, input_dim),
        )

    def forward(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode actions to indices and reconstruct.

        Args:
            actions: Action chunk of shape (B, H, A) or (B, H*A)

        Returns:
            Tuple of (reconstructed, indices, loss)
            - reconstructed: Reconstructed actions (B, H, A)
            - indices: Codebook indices (B, n_groups)
            - loss: Total VQ loss
        """
        # Flatten action chunk
        B = actions.shape[0]
        if actions.dim() == 3:
            actions_flat = actions.view(B, -1)
        else:
            actions_flat = actions

        # Encode
        z = self.encoder(actions_flat)

        # Residual VQ
        indices_list = []
        total_loss = 0.0
        quantized = torch.zeros_like(z)
        residual = z

        for vq in self.vq_layers:
            q, idx, loss = vq(residual)
            quantized = quantized + q
            residual = z - quantized
            indices_list.append(idx)
            total_loss = total_loss + loss

        # Stack indices
        indices = torch.stack(indices_list, dim=1)  # (B, n_groups)

        # Decode
        reconstructed_flat = self.decoder(quantized)
        reconstructed = reconstructed_flat.view(B, self.action_horizon, self.action_dim)

        # Add reconstruction loss
        recon_loss = F.mse_loss(reconstructed_flat, actions_flat)
        total_loss = total_loss + recon_loss

        return reconstructed, indices, total_loss

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode action chunk to codebook indices.

        Args:
            actions: Action chunk of shape (B, H, A) or (B, H*A)

        Returns:
            indices: Codebook indices of shape (B, n_groups)
        """
        B = actions.shape[0]
        if actions.dim() == 3:
            actions_flat = actions.view(B, -1)
        else:
            actions_flat = actions

        z = self.encoder(actions_flat)

        indices_list = []
        quantized = torch.zeros_like(z)
        residual = z

        for vq in self.vq_layers:
            _, idx, _ = vq(residual)
            q = vq.decode(idx)
            quantized = quantized + q
            residual = z - quantized
            indices_list.append(idx)

        return torch.stack(indices_list, dim=1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices to action chunk.

        Args:
            indices: Codebook indices of shape (B, n_groups)

        Returns:
            actions: Reconstructed action chunk of shape (B, H, A)
        """
        B = indices.shape[0]

        # Sum quantized embeddings from all groups
        quantized = torch.zeros(B, self.n_latent, device=indices.device)
        for i, vq in enumerate(self.vq_layers):
            q = vq.decode(indices[:, i])
            quantized = quantized + q

        # Decode to actions
        actions_flat = self.decoder(quantized)
        return actions_flat.view(B, self.action_horizon, self.action_dim)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'action_dim': self.action_dim,
                'action_horizon': self.action_horizon,
                'n_embed': self.n_embed,
                'n_latent': self.n_latent,
                'n_groups': self.n_groups,
            }
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "ResidualVQ":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        return model
