# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Rollout configuration."""

from dataclasses import dataclass


@dataclass
class RolloutConfig:
    """Configuration for environment rollout."""

    # Environment settings
    env_id: str = "PickPlace-CRANE-X7"
    """Environment identifier for lift simulator."""

    simulator: str = "maniskill"
    """Simulator backend (maniskill, genesis, isaacsim)."""

    backend: str = "cpu"
    """Compute backend (cpu, gpu)."""

    render_mode: str = "rgb_array"
    """Render mode (rgb_array, human, none)."""

    # Parallel environments
    num_parallel_envs: int = 8
    """Number of parallel environments for rollout."""

    # Episode settings
    max_steps: int = 200
    """Maximum steps per episode."""

    # Action settings
    action_chunk_size: int = 1
    """Number of actions to execute per VLA inference."""

    # Sampling settings
    num_rollouts_per_update: int = 16
    """Number of rollouts to collect before each PPO update."""

    # VLA inference settings
    temperature: float = 1.0
    """Temperature for action sampling."""

    do_sample: bool = True
    """Whether to sample actions or use greedy decoding."""

    # Reward settings
    use_binary_reward: bool = True
    """Whether to use binary (0/1) reward."""

    dense_reward_weight: float = 0.0
    """Weight for dense reward (0.0 = binary only)."""
