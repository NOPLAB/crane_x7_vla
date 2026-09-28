# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""PPO algorithm configuration."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    # Discount and GAE
    gamma: float = 0.99
    """Discount factor for rewards."""

    gae_lambda: float = 0.95
    """Lambda for Generalized Advantage Estimation."""

    # PPO clipping
    clip_ratio: float = 0.2
    """Clipping parameter for PPO policy loss."""

    # Loss coefficients
    value_loss_coef: float = 0.5
    """Coefficient for value function loss."""

    entropy_coef: float = 0.01
    """Coefficient for entropy bonus."""

    # Optimization
    learning_rate: float = 1e-5
    """Learning rate for optimizer."""

    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping."""

    # Training epochs
    num_epochs: int = 4
    """Number of PPO epochs per update."""

    minibatch_size: int = 4
    """Minibatch size for PPO updates."""

    # Target KL divergence (optional early stopping)
    target_kl: float | None = None
    """Target KL divergence for early stopping (None to disable)."""

    # Value function clipping
    clip_value_loss: bool = True
    """Whether to clip value function loss."""

    value_clip_range: float = 0.2
    """Clipping range for value function."""
