# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""PPO (Proximal Policy Optimization) implementation for VLA-RL.

This is a standalone PPO implementation without veRL dependency,
designed specifically for VLA model fine-tuning.
"""

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from crane_x7_vla_rl.algorithms.advantage import compute_gae, normalize_advantages
from crane_x7_vla_rl.config.ppo_config import PPOConfig

logger = logging.getLogger(__name__)


@dataclass
class PPOBatch:
    """A batch of data for PPO training."""

    observations: dict[str, torch.Tensor]
    """Observation tensors (images, states)."""

    actions: torch.Tensor
    """Action tokens or continuous actions."""

    old_log_probs: torch.Tensor
    """Log probabilities from behavior policy."""

    advantages: torch.Tensor
    """Advantage estimates."""

    returns: torch.Tensor
    """Return targets for value function."""

    old_values: torch.Tensor
    """Value estimates from behavior policy."""


@dataclass
class PPOTrainResult:
    """Result of a PPO training step."""

    policy_loss: float
    """Policy (actor) loss."""

    value_loss: float
    """Value (critic) loss."""

    entropy_loss: float
    """Entropy bonus loss."""

    total_loss: float
    """Total combined loss."""

    approx_kl: float
    """Approximate KL divergence."""

    clip_fraction: float
    """Fraction of clipped policy ratios."""


class PPOTrainer:
    """PPO trainer for VLA models.

    This trainer implements the PPO algorithm for fine-tuning VLA models
    using reinforcement learning. It supports:
    - Policy gradient with clipped objective
    - Value function loss with optional clipping
    - Entropy bonus for exploration
    - Gradient clipping
    - Early stopping based on KL divergence
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        config: PPOConfig,
        device: torch.device | str = "cuda",
    ):
        """Initialize PPO trainer.

        Args:
            policy: VLA policy model (must support forward, compute_log_prob).
            config: PPO configuration.
            device: Device for training.
        """
        self.policy = policy
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device

        # Optimizer
        self.optimizer = AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

        # Training statistics
        self.update_count = 0

    def compute_loss(
        self,
        batch: PPOBatch,
        new_log_probs: torch.Tensor,
        new_values: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute PPO loss components.

        Args:
            batch: PPO training batch.
            new_log_probs: Log probs from current policy.
            new_values: Values from current value function.
            entropy: Entropy of current policy.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Policy loss with clipping
        log_ratio = new_log_probs - batch.old_log_probs
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        advantages = batch.advantages
        if self.config.target_kl is not None:
            advantages = normalize_advantages(advantages)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        )
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss with optional clipping
        if self.config.clip_value_loss:
            value_pred_clipped = batch.old_values + torch.clamp(
                new_values - batch.old_values,
                -self.config.value_clip_range,
                self.config.value_clip_range,
            )
            value_loss1 = F.mse_loss(new_values, batch.returns, reduction="none")
            value_loss2 = F.mse_loss(
                value_pred_clipped, batch.returns, reduction="none"
            )
            value_loss = torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = F.mse_loss(new_values, batch.returns)

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Compute metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = (
                ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()
            )

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "entropy": entropy.mean().item(),
        }

        return total_loss, metrics

    def update(
        self,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        old_log_probs: np.ndarray,
        old_values: np.ndarray,
    ) -> dict[str, float]:
        """Perform PPO update on collected trajectories.

        Args:
            observations: Dict of observation arrays (images, states).
            actions: Action array (T, action_dim).
            rewards: Reward array (T,).
            dones: Done flag array (T,).
            old_log_probs: Log probs from collection (T,).
            old_values: Value estimates from collection (T,).

        Returns:
            Dictionary of training metrics.
        """
        # Convert to tensors
        obs_tensors = {
            k: torch.tensor(v, device=self.device, dtype=torch.float32)
            for k, v in observations.items()
        }
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32)
        old_log_probs_t = torch.tensor(
            old_log_probs, device=self.device, dtype=torch.float32
        )
        old_values_t = torch.tensor(old_values, device=self.device, dtype=torch.float32)

        # Compute advantages and returns using GAE
        with torch.no_grad():
            # Get value of final state
            # (In practice, this should come from the trajectory)
            next_value = 0.0 if dones[-1] else old_values[-1]

            advantages, returns = compute_gae(
                rewards_t,
                old_values_t,
                dones_t,
                next_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

            # Normalize advantages
            advantages = normalize_advantages(advantages)

        # Create batch
        batch = PPOBatch(
            observations=obs_tensors,
            actions=actions_t,
            old_log_probs=old_log_probs_t,
            advantages=advantages,
            returns=returns,
            old_values=old_values_t,
        )

        # PPO epochs
        all_metrics = []

        for epoch in range(self.config.num_epochs):
            # Get minibatches
            for minibatch in self._create_minibatches(batch):
                # Forward pass through policy
                new_log_probs, new_values, entropy = self._forward_policy(minibatch)

                # Compute loss
                loss, metrics = self.compute_loss(
                    minibatch,
                    new_log_probs,
                    new_values,
                    entropy,
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()
                all_metrics.append(metrics)

            # Early stopping based on KL divergence
            if self.config.target_kl is not None:
                avg_kl = np.mean(
                    [
                        m["approx_kl"]
                        for m in all_metrics[
                            -len(list(self._create_minibatches(batch))) :
                        ]
                    ]
                )
                if avg_kl > 1.5 * self.config.target_kl:
                    logger.info(
                        f"Early stopping at epoch {epoch} due to KL divergence: {avg_kl:.4f}"
                    )
                    break

        self.update_count += 1

        # Aggregate metrics
        final_metrics = {
            key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()
        }
        final_metrics["num_epochs"] = epoch + 1

        return final_metrics

    def _forward_policy(
        self,
        batch: PPOBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through policy to get log probs, values, and entropy.

        This method should be overridden for specific VLA architectures.

        Args:
            batch: PPO training batch.

        Returns:
            Tuple of (log_probs, values, entropy).
        """
        # This is a placeholder - actual implementation depends on VLA model
        # For OpenVLA, this would involve:
        # 1. Encode observations (images + language)
        # 2. Get action logits
        # 3. Compute log probs of taken actions
        # 4. Compute value estimate
        # 5. Compute entropy

        # Placeholder implementation
        outputs = self.policy(
            batch.observations,
            batch.actions,
        )

        log_probs = outputs.get("log_probs", torch.zeros_like(batch.old_log_probs))
        values = outputs.get("values", torch.zeros_like(batch.returns))
        entropy = outputs.get("entropy", torch.zeros(1, device=self.device))

        return log_probs, values, entropy

    def _create_minibatches(self, batch: PPOBatch) -> Iterator[PPOBatch]:
        """Create minibatches from a full batch.

        Args:
            batch: Full PPO batch.

        Yields:
            Minibatches of size config.minibatch_size.
        """
        T = batch.actions.shape[0]
        indices = np.arange(T)
        np.random.shuffle(indices)

        for start in range(0, T, self.config.minibatch_size):
            end = min(start + self.config.minibatch_size, T)
            mb_indices = indices[start:end]
            mb_indices_t = torch.tensor(mb_indices, device=self.device)

            yield PPOBatch(
                observations={
                    k: v[mb_indices_t] for k, v in batch.observations.items()
                },
                actions=batch.actions[mb_indices_t],
                old_log_probs=batch.old_log_probs[mb_indices_t],
                advantages=batch.advantages[mb_indices_t],
                returns=batch.returns[mb_indices_t],
                old_values=batch.old_values[mb_indices_t],
            )

    def save_checkpoint(self, path: str) -> None:
        """Save trainer checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load trainer checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint["update_count"]
        logger.info(f"Loaded checkpoint from {path}")
