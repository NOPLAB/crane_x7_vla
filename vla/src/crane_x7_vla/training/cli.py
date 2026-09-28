#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Command-line interface for CRANE-X7 VLA training.

Provides a unified CLI for training different VLA backends (OpenVLA, OpenPI).
Supports both simplified mode (--config file) and full mode with backend-specific arguments.

CLI arguments are automatically generated from config dataclasses.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from crane_x7_vla.core.config.base import (
    CameraConfig,
    DataConfig,
    OverfittingConfig,
    TrainingConfig,
    UnifiedVLAConfig,
)
from crane_x7_vla.training.cli_utils import (
    add_dataclass_arguments,
    apply_args_to_config,
    apply_sweep_config_to_config,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _get_backend_configs(backend: str) -> tuple[Any, ...]:
    """
    Lazy import backend-specific configs to avoid loading unnecessary dependencies.

    Returns tuple of (BackendConfig, BackendSpecificConfig, extra_configs...)
    """
    if backend == "openvla":
        from crane_x7_vla.backends.openvla.config import OpenVLAConfig, OpenVLASpecificConfig

        return (OpenVLAConfig, OpenVLASpecificConfig)
    elif backend == "minivla":
        from crane_x7_vla.backends.minivla.config import MiniVLAConfig, MiniVLASpecificConfig

        return (MiniVLAConfig, MiniVLASpecificConfig)
    elif backend == "openvla-oft":
        from crane_x7_vla.backends.openvla_oft.config import OpenVLAOFTConfig, OpenVLAOFTSpecificConfig

        return (OpenVLAOFTConfig, OpenVLAOFTSpecificConfig)
    elif backend in ("pi0", "pi0.5"):
        from crane_x7_vla.backends.pi0.config import Pi0Config, Pi0SpecificConfig

        return (Pi0Config, Pi0SpecificConfig)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_default_config(backend: str, data_root: Path, output_dir: Path, experiment_name: str) -> UnifiedVLAConfig:
    """
    Create default configuration for the specified backend.

    Args:
        backend: Backend type ('openvla', 'openpi', etc.)
        data_root: Path to training data
        output_dir: Path to output directory
        experiment_name: Experiment name

    Returns:
        Backend-specific config instance
    """
    # Default camera configuration
    cameras = [CameraConfig(name="primary", topic="/camera/color/image_raw", width=640, height=480)]

    # Data configuration
    data_config = DataConfig(data_root=data_root, cameras=cameras)

    # Training configuration
    training_config = TrainingConfig(
        batch_size=16,
        num_epochs=100,
        learning_rate=5e-4,
    )

    # Lazy import backend configs
    configs = _get_backend_configs(backend)
    BackendConfig, BackendSpecificConfig = configs[0], configs[1]

    # Create backend-specific config
    if backend == "openvla":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openvla=BackendSpecificConfig(),
        )
    elif backend == "minivla":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            minivla=BackendSpecificConfig(),
        )
    elif backend == "openvla-oft":
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openvla_oft=BackendSpecificConfig(),
        )
    elif backend in ("pi0", "pi0.5"):
        return BackendConfig(
            backend=backend,
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            pi0=BackendSpecificConfig(model_type=backend),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    backend = args.backend

    # Load or create configuration
    if args.config:
        # Load from YAML and determine backend
        config = UnifiedVLAConfig.from_yaml(args.config)
        # Re-create as proper backend-specific config
        configs = _get_backend_configs(config.backend)
        BackendConfig = configs[0]
        config = BackendConfig.from_yaml(args.config)

        # Warn if config backend doesn't match CLI
        if config.backend != backend:
            logging.warning(
                f"Config file specifies backend '{config.backend}' but CLI subcommand is '{backend}'. "
                f"Using config file backend."
            )
            backend = config.backend
    else:
        if not args.data_root:
            raise ValueError("--data-root is required when not using --config")

        config = create_default_config(
            backend=backend,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name,
        )

    # Apply CLI overrides automatically from TrainingConfig, OverfittingConfig, and LoRAConfig
    apply_args_to_config(args, config.training, prefix="training")
    apply_args_to_config(args, config.overfitting, prefix="overfitting")
    _apply_lora_args_to_config(args, config)
    _apply_pi0_args_to_config(args, config)

    # Save configuration
    config_save_path = Path(config.output_dir) / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_save_path)
    logging.info(f"Configuration saved to {config_save_path}")

    # Lazy import trainer to avoid loading all backends
    from crane_x7_vla.training.trainer import VLATrainer

    # Create trainer and start training
    trainer = VLATrainer(config)
    results = trainer.train()

    logging.info(f"Training completed: {results}")


def evaluate_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    if not args.config:
        raise ValueError("--config is required for evaluation")

    config = UnifiedVLAConfig.from_yaml(args.config)
    configs = _get_backend_configs(config.backend)
    BackendConfig = configs[0]
    config = BackendConfig.from_yaml(args.config)

    from crane_x7_vla.training.trainer import VLATrainer

    trainer = VLATrainer(config)
    metrics = trainer.evaluate(checkpoint_path=args.checkpoint, test_data_path=args.test_data)

    logging.info(f"Evaluation metrics: {metrics}")


def config_command(args: argparse.Namespace) -> None:
    """Generate default configuration file."""
    config = create_default_config(
        backend=args.backend,
        data_root=Path(args.data_root) if args.data_root else Path("./data"),
        output_dir=Path(args.output_dir) if args.output_dir else Path("./outputs"),
        experiment_name=args.experiment_name,
    )

    output_path = Path(args.output)
    config.to_yaml(output_path)

    logging.info(f"Default configuration saved to {output_path}")
    logging.info(f"Backend: {args.backend}")
    logging.info("Edit this file to customize your training configuration.")


def agent_command(args: argparse.Namespace) -> None:
    """Execute W&B Sweep agent command."""
    import wandb

    backend = args.backend

    def run_sweep_training():
        """Training function called by wandb.agent()."""
        # Initialize W&B run (agent will set the config)
        run = wandb.init()

        # Get sweep parameters from W&B config
        sweep_config = dict(wandb.config)
        logging.info(f"Sweep parameters: {sweep_config}")

        # Create base configuration
        config = create_default_config(
            backend=backend,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name,
        )

        # Apply sweep parameters automatically
        apply_sweep_config_to_config(sweep_config, config)

        # Apply fixed CLI parameters (override sweep config)
        apply_args_to_config(args, config.training, prefix="training")
        apply_args_to_config(args, config.overfitting, prefix="overfitting")
        _apply_lora_args_to_config(args, config)
        _apply_pi0_args_to_config(args, config)

        # Save configuration
        config_save_path = Path(config.output_dir) / f"config_{run.id}.yaml"
        config_save_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(config_save_path)
        logging.info(f"Configuration saved to {config_save_path}")

        # Lazy import trainer
        from crane_x7_vla.training.trainer import VLATrainer

        # Create trainer and start training
        trainer = VLATrainer(config)
        results = trainer.train()

        logging.info(f"Training completed: {results}")

        # Finish W&B run
        wandb.finish()

    # Build sweep path
    sweep_path = f"{args.entity}/{args.project}/{args.sweep_id}"
    logging.info(f"Starting W&B Sweep agent for: {sweep_path}")

    # Run the agent (count=1 for single run per Slurm job)
    wandb.agent(sweep_path, function=run_sweep_training, count=1)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for data and output paths."""
    parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--data-root", type=str, help="Path to training data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment-name", type=str, default="crane_x7_vla", help="Experiment name")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training configuration arguments automatically from TrainingConfig."""
    add_dataclass_arguments(
        parser,
        TrainingConfig,
        prefix="training",
        # Include commonly used training parameters
        include_fields={
            "batch_size",
            "learning_rate",
            "num_epochs",
            "max_steps",
            "gradient_accumulation_steps",
            "weight_decay",
            "warmup_steps",
            "max_grad_norm",
            "gradient_checkpointing",
            "save_interval",
            "eval_interval",
            "log_interval",
            "mixed_precision",
        },
    )


def _add_overfitting_arguments(parser: argparse.ArgumentParser) -> None:
    """Add overfitting configuration arguments automatically from OverfittingConfig."""
    add_dataclass_arguments(
        parser,
        OverfittingConfig,
        prefix="overfitting",
    )


def _add_pi0_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Pi0/Pi0.5 specific configuration arguments."""
    pi0_group = parser.add_argument_group("Pi0/Pi0.5 Configuration")

    pi0_group.add_argument(
        "--openpi-checkpoint",
        type=str,
        default=None,
        metavar="NAME",
        help="OpenPI checkpoint name (e.g., 'pi0_base', 'pi05_base', 'pi0_droid', 'pi05_droid')",
    )


def _add_lora_arguments(parser: argparse.ArgumentParser) -> None:
    """Add LoRA configuration arguments."""
    lora_group = parser.add_argument_group("LoRA Configuration")

    lora_group.add_argument(
        "--lora-enabled",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        metavar="BOOL",
        help="Enable LoRA fine-tuning (default: True)",
    )
    lora_group.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        metavar="INT",
        help="LoRA rank (default: 32)",
    )
    lora_group.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        metavar="INT",
        help="LoRA alpha scaling factor (default: 16)",
    )
    lora_group.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        metavar="FLOAT",
        help="LoRA dropout (default: 0.05)",
    )
    lora_group.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        metavar="MODULE",
        help="Target modules for LoRA (default: backend-specific)",
    )
    lora_group.add_argument(
        "--lora-use-rslora",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        metavar="BOOL",
        help="Use Rank-Stabilized LoRA (default: False)",
    )
    lora_group.add_argument(
        "--lora-use-dora",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        metavar="BOOL",
        help="Use Weight-Decomposed LoRA (default: False)",
    )
    lora_group.add_argument(
        "--lora-skip-merge-on-save",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        metavar="BOOL",
        help="Skip LoRA merge during checkpoint saving (default: True)",
    )


def _apply_lora_args_to_config(args: argparse.Namespace, config: UnifiedVLAConfig) -> None:
    """Apply LoRA arguments from CLI to config."""
    options = {
        "lora_enabled": ("enabled", "use_lora"),
        "lora_rank": ("rank", "lora_rank"),
        "lora_alpha": ("alpha", "lora_alpha"),
        "lora_dropout": ("dropout", "lora_dropout"),
        "lora_target_modules": ("target_modules", "lora_target_modules"),
        "lora_use_rslora": ("use_rslora", "lora_rslora"),
        "lora_use_dora": ("use_dora", None),
        "lora_skip_merge_on_save": ("skip_merge_on_save", "skip_merge_on_save"),
    }
    specific = next((getattr(config, name) for name in ("openvla", "openvla_oft", "minivla", "pi0")
                     if hasattr(config, name)), None)
    if getattr(args, "lora_use_dora", None):
        raise ValueError("DoRA is not implemented by the VLA backends")
    if getattr(args, "lora_use_rslora", None) and not hasattr(config, "pi0"):
        raise ValueError("rsLoRA is only implemented by the Pi0/Pi0.5 backend")
    for argument, (base_name, specific_name) in options.items():
        value = getattr(args, argument, None)
        if value is None:
            continue
        setattr(config.lora, base_name, value)
        if specific is not None and specific_name and hasattr(specific, specific_name):
            setattr(specific, specific_name, value)

    # Pi0 has separate adapters for the VLM and action expert.
    if hasattr(config, "pi0"):
        for argument, names in (("lora_rank", ("vlm_lora_rank", "expert_lora_rank")),
                                ("lora_alpha", ("vlm_lora_alpha", "expert_lora_alpha"))):
            value = getattr(args, argument, None)
            if value is not None:
                for name in names:
                    setattr(config.pi0, name, value)


def _apply_pi0_args_to_config(args: argparse.Namespace, config: UnifiedVLAConfig) -> None:
    """Apply Pi0/Pi0.5 arguments from CLI to config."""
    # Only apply if this is a Pi0/Pi0.5 config
    if not hasattr(config, "pi0"):
        return

    if hasattr(args, "openpi_checkpoint") and args.openpi_checkpoint is not None:
        config.pi0.openpi_checkpoint = args.openpi_checkpoint


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CRANE-X7 VLA Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with OpenVLA using default settings
  python -m crane_x7_vla.training.cli train openvla --data-root ./data

  # Train with overrides (auto-generated from config)
  python -m crane_x7_vla.training.cli train openvla --data-root ./data \\
      --training-batch-size 8 --training-max-steps 1000

  # Train with configuration file
  python -m crane_x7_vla.training.cli train openvla --config my_config.yaml

  # Generate default configuration
  python -m crane_x7_vla.training.cli config --backend openvla --output openvla_config.yaml

  # Evaluate trained model
  python -m crane_x7_vla.training.cli evaluate --config my_config.yaml --checkpoint ./outputs/checkpoint

  # Run W&B Sweep agent
  python -m crane_x7_vla.training.cli agent pi0.5 --sweep-id abc123 --entity myteam --project myproject --data-root ./data
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Define available backends
    backends = ["openvla", "minivla", "openvla-oft", "pi0", "pi0.5"]

    # =====================
    # Train command
    # =====================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a VLA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_subparsers = train_parser.add_subparsers(dest="backend", help="VLA backend to use")

    for backend in backends:
        backend_parser = train_subparsers.add_parser(
            backend,
            help=f"Train with {backend} backend",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_common_arguments(backend_parser)
        _add_training_arguments(backend_parser)
        _add_overfitting_arguments(backend_parser)
        _add_lora_arguments(backend_parser)
        # Add Pi0/Pi0.5 specific arguments
        if backend in ("pi0", "pi0.5"):
            _add_pi0_arguments(backend_parser)

    # =====================
    # Evaluate command
    # =====================
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to configuration file (YAML)")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--test-data", type=str, help="Path to test dataset")

    # =====================
    # Config command
    # =====================
    config_parser = subparsers.add_parser("config", help="Generate default configuration file")
    config_parser.add_argument(
        "--backend",
        type=str,
        choices=backends,
        required=True,
        help="VLA backend",
    )
    config_parser.add_argument("--output", type=str, default="config.yaml", help="Output configuration file path")
    config_parser.add_argument("--data-root", type=str, help="Path to training data directory")
    config_parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints and logs")
    config_parser.add_argument("--experiment-name", type=str, default="crane_x7_vla", help="Experiment name")

    # =====================
    # Agent command (W&B Sweep)
    # =====================
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run W&B Sweep agent for hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    agent_subparsers = agent_parser.add_subparsers(dest="backend", help="VLA backend to use")

    for backend in backends:
        agent_backend_parser = agent_subparsers.add_parser(
            backend,
            help=f"Run sweep agent with {backend} backend",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # W&B Sweep arguments
        agent_backend_parser.add_argument("--sweep-id", type=str, required=True, help="W&B Sweep ID")
        agent_backend_parser.add_argument("--entity", type=str, required=True, help="W&B entity (username or team)")
        agent_backend_parser.add_argument("--project", type=str, required=True, help="W&B project name")

        # Data and output arguments
        agent_backend_parser.add_argument(
            "--data-root", type=str, required=True, help="Path to training data directory"
        )
        agent_backend_parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
        agent_backend_parser.add_argument(
            "--experiment-name", type=str, default="crane_x7_vla_sweep", help="Experiment name"
        )

        # Auto-generate training, overfitting, and LoRA arguments
        _add_training_arguments(agent_backend_parser)
        _add_overfitting_arguments(agent_backend_parser)
        _add_lora_arguments(agent_backend_parser)
        # Add Pi0/Pi0.5 specific arguments
        if backend in ("pi0", "pi0.5"):
            _add_pi0_arguments(agent_backend_parser)

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "train":
        if args.backend is None:
            train_parser.print_help()
            sys.exit(1)
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "config":
        config_command(args)
    elif args.command == "agent":
        if args.backend is None:
            agent_parser.print_help()
            sys.exit(1)
        agent_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
