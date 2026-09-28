# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Generalized Advantage Estimation (GAE) implementation."""

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray | torch.Tensor,
    values: np.ndarray | torch.Tensor,
    dones: np.ndarray | torch.Tensor,
    next_value: float | torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a balance between bias and variance in advantage estimation.
    Reference: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    https://arxiv.org/abs/1506.02438

    Args:
        rewards: Rewards at each timestep (T,) or (T, N) for batched.
        values: Value estimates at each timestep (T,) or (T, N).
        dones: Done flags at each timestep (T,) or (T, N).
        next_value: Value estimate for the state after the last timestep.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter (0 = one-step TD, 1 = Monte Carlo).

    Returns:
        Tuple of (advantages, returns) with same shape as inputs.
    """
    use_torch = isinstance(rewards, torch.Tensor)

    if use_torch:
        return _compute_gae_torch(rewards, values, dones, next_value, gamma, gae_lambda)
    else:
        return _compute_gae_numpy(rewards, values, dones, next_value, gamma, gae_lambda)


def _compute_gae_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of GAE."""
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = next_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = values[t + 1]

        # TD error: delta = r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]

        # GAE: A = delta + gamma * lambda * (1 - done) * A_{t+1}
        advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae
        last_gae = advantages[t]

    # Returns = Advantages + Values
    returns = advantages + values

    return advantages, returns


def _compute_gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor | float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of GAE."""
    T = rewards.shape[0]
    device = rewards.device
    dtype = rewards.dtype

    advantages = torch.zeros_like(rewards)
    last_gae = (
        torch.zeros(rewards.shape[1:], device=device, dtype=dtype)
        if rewards.dim() > 1
        else torch.tensor(0.0, device=device, dtype=dtype)
    )

    if not isinstance(next_value, torch.Tensor):
        next_value = torch.tensor(next_value, device=device, dtype=dtype)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae
        last_gae = advantages[t]

    returns = advantages + values

    return advantages, returns


def compute_returns(
    rewards: np.ndarray | torch.Tensor,
    dones: np.ndarray | torch.Tensor,
    next_value: float | torch.Tensor,
    gamma: float = 0.99,
) -> np.ndarray | torch.Tensor:
    """Compute discounted returns (Monte Carlo estimates).

    Args:
        rewards: Rewards at each timestep (T,).
        dones: Done flags at each timestep (T,).
        next_value: Value estimate for the state after the last timestep.
        gamma: Discount factor.

    Returns:
        Discounted returns with same shape as rewards.
    """
    use_torch = isinstance(rewards, torch.Tensor)

    T = len(rewards)

    if use_torch:
        returns = torch.zeros_like(rewards)
        running_return = (
            next_value
            if isinstance(next_value, torch.Tensor)
            else torch.tensor(next_value, device=rewards.device, dtype=rewards.dtype)
        )
    else:
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = float(next_value)

    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1.0 - float(dones[t]))
        returns[t] = running_return

    return returns


def normalize_advantages(
    advantages: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> np.ndarray | torch.Tensor:
    """Normalize advantages to have zero mean and unit variance.

    Args:
        advantages: Advantage estimates.
        eps: Small constant for numerical stability.

    Returns:
        Normalized advantages.
    """
    if isinstance(advantages, torch.Tensor):
        return (advantages - advantages.mean()) / (advantages.std() + eps)
    else:
        return (advantages - advantages.mean()) / (advantages.std() + eps)
