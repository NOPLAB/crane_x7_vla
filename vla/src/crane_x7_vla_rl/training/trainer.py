# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""VLA-RL Trainer - Main training orchestrator.

This module provides the VLARLTrainer class that coordinates:
- VLA model loading (from vla/ checkpoints or HuggingFace)
- Rollout collection with parallel environments
- PPO training updates
- Logging and checkpointing
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crane_x7_vla_rl.algorithms.ppo import PPOTrainer
from crane_x7_vla_rl.config.base import VLARLConfig
from crane_x7_vla_rl.rollout.rollout_manager import RolloutManager
from crane_x7_vla_rl.vla.openvla_adapter import OpenVLAAdapter

logger = logging.getLogger(__name__)


class VLARLTrainer:
    """Main VLA-RL trainer that orchestrates the entire training process.

    This trainer combines:
    - OpenVLA model with LoRA fine-tuning
    - Parallel environment rollouts using lift simulator
    - PPO updates with GAE
    - W&B logging and checkpointing

    Example:
        >>> config = VLARLConfig()
        >>> trainer = VLARLTrainer(config)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: VLARLConfig,
        resume_from: str | Path | None = None,
    ):
        """Initialize VLA-RL trainer.

        Args:
            config: VLA-RL configuration.
            resume_from: Path to checkpoint to resume from.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds
        self._set_seeds(config.seed)

        # Initialize wandb if enabled
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        # Load VLA model
        logger.info("Initializing VLA model...")
        self.policy = self._load_policy()

        # Create rollout manager
        logger.info("Setting up rollout manager...")
        self.rollout_manager = RolloutManager(
            config=config.rollout,
            policy_fn=self._policy_fn,
            device=self.device,
        )

        # Create PPO trainer
        logger.info("Initializing PPO trainer...")
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            config=config.ppo,
            device=self.device,
        )

        # Training state
        self.global_step = 0
        self.update_count = 0
        self.best_success_rate = 0.0

        # Resume if specified
        if resume_from is not None:
            self._load_checkpoint(resume_from)

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VLARLTrainer initialized. Output: {self.output_dir}")

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__,
            )
            logger.info(f"W&B initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            self.config.use_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False

    def _load_policy(self) -> OpenVLAAdapter:
        """Load VLA policy model."""
        if self.config.sft_checkpoint is not None:
            # Load from vla/ training checkpoint
            logger.info(f"Loading from SFT checkpoint: {self.config.sft_checkpoint}")
            return OpenVLAAdapter.from_vla_checkpoint(
                checkpoint_path=self.config.sft_checkpoint,
                device=self.device,
            )
        else:
            # Load from pretrained
            logger.info(f"Loading from pretrained: {self.config.pretrained_checkpoint}")
            return OpenVLAAdapter(
                model_name_or_path=self.config.pretrained_checkpoint,
                use_lora=True,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                device=self.device,
            )

    def _policy_fn(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        """Policy function for rollout manager.

        Args:
            observation: Dict with 'image' and 'state' arrays.

        Returns:
            Tuple of (action, log_prob, value).
        """
        action, log_prob, value = self.policy.generate_action(
            image=observation["image"],
            instruction=self.config.language_instruction,
            temperature=self.config.rollout.temperature,
            do_sample=True,
        )
        return action, log_prob, value

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Final training metrics.
        """
        logger.info("Starting VLA-RL training...")
        logger.info(f"  Total updates: {self.config.num_updates}")
        logger.info(
            f"  Rollouts per update: {self.config.rollout.num_rollouts_per_update}"
        )
        logger.info(f"  Parallel envs: {self.config.rollout.num_parallel_envs}")

        start_time = time.time()
        all_metrics = []

        try:
            for update in range(self.update_count, self.config.num_updates):
                update_metrics = self._training_step(update)
                all_metrics.append(update_metrics)

                # Periodic evaluation
                if (update + 1) % self.config.eval_interval == 0:
                    eval_metrics = self._evaluate()
                    self._log_metrics({**update_metrics, **eval_metrics}, update)

                    # Save best model
                    if eval_metrics["eval_success_rate"] > self.best_success_rate:
                        self.best_success_rate = eval_metrics["eval_success_rate"]
                        self._save_checkpoint("best")
                        logger.info(
                            f"New best model! Success rate: {self.best_success_rate:.2%}"
                        )
                else:
                    self._log_metrics(update_metrics, update)

                # Periodic checkpoint
                if (update + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(f"update_{update + 1}")

                self.update_count = update + 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Save final checkpoint
            self._save_checkpoint("final")
            self.rollout_manager.close()

            if self.wandb_run is not None:
                self.wandb_run.finish()

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed / 3600:.1f} hours")

        # Return final metrics
        if all_metrics:
            return {
                key: np.mean([m[key] for m in all_metrics if key in m])
                for key in all_metrics[-1].keys()
            }
        return {}

    def _training_step(self, update: int) -> dict[str, float]:
        """Execute a single training update.

        Args:
            update: Current update number.

        Returns:
            Dictionary of training metrics.
        """
        step_start = time.time()

        # Collect rollouts
        logger.debug(f"Update {update + 1}: Collecting rollouts...")
        trajectory_buffer = self.rollout_manager.collect_rollouts(
            seed=self.config.seed + update,
        )

        rollout_stats = trajectory_buffer.get_statistics()
        self.global_step += rollout_stats["total_steps"]

        # Convert to batch for PPO
        batch = trajectory_buffer.to_batch()

        # PPO update
        logger.debug(f"Update {update + 1}: Running PPO update...")
        ppo_metrics = self.ppo_trainer.update(
            observations={"images": batch["images"], "states": batch["states"]},
            actions=batch["actions"],
            rewards=batch["rewards"],
            dones=batch["dones"],
            old_log_probs=batch["log_probs"],
            old_values=batch["values"],
        )

        step_time = time.time() - step_start

        # Combine metrics
        metrics = {
            "global_step": self.global_step,
            "update": update + 1,
            "step_time": step_time,
            "rollout/success_rate": rollout_stats["success_rate"],
            "rollout/mean_reward": rollout_stats["mean_reward"],
            "rollout/mean_length": rollout_stats["mean_length"],
            "rollout/num_trajectories": rollout_stats["num_trajectories"],
            **{f"ppo/{k}": v for k, v in ppo_metrics.items()},
        }

        # Log progress
        if (update + 1) % self.config.log_interval == 0:
            logger.info(
                f"Update {update + 1}/{self.config.num_updates} | "
                f"Steps: {self.global_step} | "
                f"Success: {rollout_stats['success_rate']:.2%} | "
                f"Reward: {rollout_stats['mean_reward']:.2f} | "
                f"Policy Loss: {ppo_metrics['policy_loss']:.4f} | "
                f"Time: {step_time:.1f}s"
            )

        return metrics

    def _evaluate(self) -> dict[str, float]:
        """Run evaluation with deterministic policy.

        Returns:
            Evaluation metrics.
        """
        logger.info("Running evaluation...")

        eval_metrics = self.rollout_manager.evaluate(
            num_episodes=self.config.num_eval_episodes,
            seed=self.config.seed,
            deterministic=True,
        )

        logger.info(
            f"Evaluation: Success={eval_metrics['eval_success_rate']:.2%}, "
            f"Reward={eval_metrics['eval_mean_reward']:.2f}"
        )

        return eval_metrics

    def _log_metrics(self, metrics: dict[str, float], update: int) -> None:
        """Log metrics to W&B and/or console.

        Args:
            metrics: Metrics dictionary.
            update: Current update number.
        """
        if self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=update)

    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint.

        Args:
            name: Checkpoint name (e.g., 'best', 'final', 'update_100').
        """
        checkpoint_dir = self.output_dir / f"checkpoint_{name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy
        self.policy.save_checkpoint(checkpoint_dir)

        # Save trainer state
        trainer_state = {
            "global_step": self.global_step,
            "update_count": self.update_count,
            "best_success_rate": self.best_success_rate,
            "config": self.config.__dict__,
        }
        torch.save(trainer_state, checkpoint_dir / "trainer_state.pt")

        # Save PPO optimizer state
        self.ppo_trainer.save_checkpoint(str(checkpoint_dir / "ppo_checkpoint.pt"))

        logger.info(f"Saved checkpoint: {checkpoint_dir}")

    def _load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint directory.
        """
        path = Path(path)
        logger.info(f"Loading checkpoint from {path}")

        # Load trainer state
        trainer_state = torch.load(path / "trainer_state.pt", map_location=self.device)
        self.global_step = trainer_state["global_step"]
        self.update_count = trainer_state["update_count"]
        self.best_success_rate = trainer_state["best_success_rate"]

        # Load PPO state
        ppo_path = path / "ppo_checkpoint.pt"
        if ppo_path.exists():
            self.ppo_trainer.load_checkpoint(str(ppo_path))

        logger.info(
            f"Resumed from update {self.update_count}, "
            f"global_step {self.global_step}"
        )


def create_trainer(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> VLARLTrainer:
    """Factory function to create VLARLTrainer.

    Args:
        config_path: Path to YAML config file.
        **overrides: Config overrides as keyword arguments.

    Returns:
        Configured VLARLTrainer instance.
    """
    if config_path is not None:
        config = VLARLConfig.from_yaml(config_path)
    else:
        config = VLARLConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config key: {key}")

    return VLARLTrainer(config)
