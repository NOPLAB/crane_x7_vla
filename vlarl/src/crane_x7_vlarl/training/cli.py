# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Command-line interface for VLA-RL training.

Usage:
    # Train from SFT checkpoint
    python -m crane_x7_vlarl.training.cli train \\
        --sft-checkpoint /workspace/vla/outputs/checkpoint \\
        --experiment-name crane_x7_vlarl

    # Train from pretrained
    python -m crane_x7_vlarl.training.cli train \\
        --pretrained openvla/openvla-7b \\
        --simulator maniskill

    # Evaluate checkpoint
    python -m crane_x7_vlarl.training.cli evaluate \\
        --checkpoint /workspace/vlarl/outputs/crane_x7_vlarl/checkpoint_best

    # Generate config file
    python -m crane_x7_vlarl.training.cli config \\
        --output my_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from crane_x7_vlarl.config.base import VLARLConfig
from crane_x7_vlarl.config.ppo_config import PPOConfig
from crane_x7_vlarl.config.rollout_config import RolloutConfig
from crane_x7_vlarl.training.trainer import VLARLTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VLA-RL Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train VLA with RL")
    _add_train_args(train_parser)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    _add_eval_args(eval_parser)

    # Config command
    config_parser = subparsers.add_parser("config", help="Generate config file")
    _add_config_args(config_parser)

    return parser.parse_args()


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training arguments."""
    # Model source
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--sft-checkpoint",
        type=str,
        help="Path to SFT checkpoint from vla/ training",
    )
    model_group.add_argument(
        "--pretrained",
        type=str,
        default="openvla/openvla-7b",
        help="HuggingFace model ID or path (default: openvla/openvla-7b)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="crane_x7_vlarl",
        help="Experiment name for logging and checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Environment
    parser.add_argument(
        "--simulator",
        type=str,
        default="maniskill",
        choices=["maniskill", "genesis", "isaacsim"],
        help="Simulator backend",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickPlace-CRANE-X7",
        help="Environment ID",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Simulation backend",
    )
    parser.add_argument(
        "--num-parallel-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable real-time visualization (human render mode)",
    )

    # Training
    parser.add_argument(
        "--num-updates",
        type=int,
        default=1000,
        help="Total number of PPO updates",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=8,
        help="Rollouts per update",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="PPO learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="PPO epochs per update",
    )

    # Language instruction
    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the object and place it",
        help="Language instruction for the task",
    )

    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="crane-x7-vlarl",
        help="W&B project name",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation arguments."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default="maniskill",
        choices=["maniskill", "genesis", "isaacsim"],
        help="Simulator backend",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickPlace-CRANE-X7",
        help="Environment ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes",
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add config generation arguments."""
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="config.yaml",
        help="Output config file path",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["default", "fast", "full"],
        default="default",
        help="Config template to use",
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Execute training command."""
    logger.info("Starting VLA-RL training...")

    # Load or create config
    if args.config:
        config = VLARLConfig.from_yaml(args.config)
    else:
        config = VLARLConfig()

    # Apply CLI overrides
    config.experiment_name = args.experiment_name
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.num_updates = args.num_updates
    config.language_instruction = args.instruction
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project

    # Model source
    if args.sft_checkpoint:
        config.sft_checkpoint = args.sft_checkpoint
    else:
        config.pretrained_checkpoint = args.pretrained

    # Rollout config
    config.rollout.simulator = args.simulator
    config.rollout.env_id = args.env_id
    config.rollout.backend = args.backend
    config.rollout.num_parallel_envs = args.num_parallel_envs
    config.rollout.num_rollouts_per_update = args.num_rollouts
    config.rollout.render_mode = "human" if args.render else "rgb_array"

    # PPO config
    config.ppo.learning_rate = args.learning_rate
    config.ppo.num_epochs = args.num_epochs

    # Create trainer
    trainer = VLARLTrainer(
        config=config,
        resume_from=args.resume,
    )

    # Run training
    final_metrics = trainer.train()

    # Print final results
    logger.info("Training completed!")
    logger.info("Final metrics:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute evaluation command."""
    import numpy as np
    import torch

    from crane_x7_vlarl.environments.lift_wrapper import LiftRolloutEnvironment
    from crane_x7_vlarl.vla.openvla_adapter import OpenVLAAdapter

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    # Load policy
    checkpoint_path = Path(args.checkpoint)
    policy = OpenVLAAdapter.from_vla_checkpoint(
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = LiftRolloutEnvironment.from_config(
        env_id=args.env_id,
        simulator_name=args.simulator,
        render_mode=render_mode,
    )

    # Run evaluation
    successes = []
    rewards = []
    lengths = []

    for ep in range(args.num_episodes):
        ep_seed = args.seed + ep
        obs, _ = env.reset(seed=ep_seed)

        episode_reward = 0.0
        episode_length = 0
        episode_success = False

        while True:
            action, _, _ = policy.generate_action(
                image=obs.image,
                temperature=0.0,  # Deterministic
                do_sample=False,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if info.get("success", False):
                episode_success = True

            if terminated or truncated:
                break

        successes.append(episode_success)
        rewards.append(episode_reward)
        lengths.append(episode_length)

        logger.info(
            f"Episode {ep + 1}/{args.num_episodes}: "
            f"Success={episode_success}, Reward={episode_reward:.2f}, "
            f"Length={episode_length}"
        )

    env.close()

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Summary:")
    logger.info(f"  Success Rate: {np.mean(successes):.2%}")
    logger.info(f"  Mean Reward:  {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    logger.info(f"  Mean Length:  {np.mean(lengths):.1f}")
    logger.info("=" * 50)

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Generate config file."""
    # Create config based on template
    if args.template == "fast":
        config = VLARLConfig(
            num_updates=100,
            rollout=RolloutConfig(
                num_parallel_envs=2,
                num_rollouts_per_update=4,
                max_steps=50,
            ),
            ppo=PPOConfig(
                num_epochs=2,
                minibatch_size=16,
            ),
        )
    elif args.template == "full":
        config = VLARLConfig(
            num_updates=5000,
            rollout=RolloutConfig(
                num_parallel_envs=8,
                num_rollouts_per_update=16,
                max_steps=200,
            ),
            ppo=PPOConfig(
                num_epochs=10,
                minibatch_size=64,
            ),
        )
    else:
        config = VLARLConfig()

    # Convert to dict
    config_dict = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "device": config.device,
        "pretrained_checkpoint": config.pretrained_checkpoint,
        "sft_checkpoint": config.sft_checkpoint,
        "num_updates": config.num_updates,
        "language_instruction": config.language_instruction,
        "use_wandb": config.use_wandb,
        "wandb_project": config.wandb_project,
        "rollout": {
            "simulator": config.rollout.simulator,
            "env_id": config.rollout.env_id,
            "backend": config.rollout.backend,
            "num_parallel_envs": config.rollout.num_parallel_envs,
            "num_rollouts_per_update": config.rollout.num_rollouts_per_update,
            "max_steps": config.rollout.max_steps,
            "use_binary_reward": config.rollout.use_binary_reward,
        },
        "ppo": {
            "learning_rate": config.ppo.learning_rate,
            "gamma": config.ppo.gamma,
            "gae_lambda": config.ppo.gae_lambda,
            "clip_ratio": config.ppo.clip_ratio,
            "value_loss_coef": config.ppo.value_loss_coef,
            "entropy_coef": config.ppo.entropy_coef,
            "num_epochs": config.ppo.num_epochs,
            "minibatch_size": config.ppo.minibatch_size,
            "max_grad_norm": config.ppo.max_grad_norm,
        },
    }

    # Write to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config written to: {output_path}")
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.command is None:
        logger.error("No command specified. Use --help for usage.")
        return 1

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "config": cmd_config,
    }

    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
