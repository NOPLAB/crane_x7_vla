#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Command-line interface for CRANE-X7 VLA training.

Provides a unified CLI for training different VLA backends (OpenVLA, OpenPI).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from crane_x7_vla.config.base import UnifiedVLAConfig, DataConfig, TrainingConfig, CameraConfig
from crane_x7_vla.config.openvla_config import OpenVLAConfig, OpenVLASpecificConfig
from crane_x7_vla.config.openpi_config import OpenPIConfig, OpenPISpecificConfig
from crane_x7_vla.training.trainer import VLATrainer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_default_config(
    backend: str,
    data_root: Path,
    output_dir: Path,
    experiment_name: str
) -> UnifiedVLAConfig:
    """
    Create default configuration for the specified backend.

    Args:
        backend: Backend type ('openvla' or 'openpi')
        data_root: Path to training data
        output_dir: Path to output directory
        experiment_name: Experiment name

    Returns:
        UnifiedVLAConfig instance
    """
    # Default camera configuration
    cameras = [
        CameraConfig(
            name="primary",
            topic="/camera/color/image_raw",
            width=640,
            height=480
        )
    ]

    # Data configuration
    data_config = DataConfig(
        data_root=data_root,
        cameras=cameras
    )

    # Training configuration
    training_config = TrainingConfig(
        batch_size=16,
        num_epochs=100,
        learning_rate=5e-4,
    )

    if backend == "openvla":
        config = OpenVLAConfig(
            backend="openvla",
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openvla=OpenVLASpecificConfig()
        )
    elif backend == "openpi":
        config = OpenPIConfig(
            backend="openpi",
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openpi=OpenPISpecificConfig()
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return config


def train_command(args):
    """Execute training command."""
    # Load or create configuration
    if args.config:
        config = UnifiedVLAConfig.from_yaml(args.config)
    else:
        if not args.data_root:
            raise ValueError("--data-root is required when not using --config")

        config = create_default_config(
            backend=args.backend,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name
        )

    # Override config with command-line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.gradient_checkpointing:
        config.training.gradient_checkpointing = True

    # Override backend-specific settings (OpenVLA)
    if config.backend == "openvla":
        if args.lora_rank:
            config.openvla.lora_rank = args.lora_rank
            # Update backend_config as well
            if config.backend_config is not None:
                config.backend_config['lora_rank'] = args.lora_rank
        if args.use_quantization:
            config.openvla.use_quantization = True
            # Update backend_config as well
            if config.backend_config is not None:
                config.backend_config['use_quantization'] = True

    # Save configuration
    config_save_path = Path(config.output_dir) / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_save_path)
    logging.info(f"Configuration saved to {config_save_path}")

    # Create trainer and start training
    trainer = VLATrainer(config)
    results = trainer.train()

    logging.info(f"Training completed: {results}")


def evaluate_command(args):
    """Execute evaluation command."""
    # Load configuration
    if args.config:
        config = UnifiedVLAConfig.from_yaml(args.config)
    else:
        raise ValueError("--config is required for evaluation")

    # Create trainer
    trainer = VLATrainer(config)

    # Evaluate
    metrics = trainer.evaluate(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data
    )

    logging.info(f"Evaluation metrics: {metrics}")


def config_command(args):
    """Generate default configuration file."""
    config = create_default_config(
        backend=args.backend,
        data_root=Path(args.data_root) if args.data_root else Path("./data"),
        output_dir=Path(args.output_dir) if args.output_dir else Path("./outputs"),
        experiment_name=args.experiment_name
    )

    output_path = Path(args.output)
    config.to_yaml(output_path)

    logging.info(f"Default configuration saved to {output_path}")
    logging.info(f"Backend: {args.backend}")
    logging.info("Edit this file to customize your training configuration.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CRANE-X7 VLA Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with OpenVLA using default settings
  python -m crane_x7_vla.training.cli train --backend openvla --data-root ./data --experiment-name my_experiment

  # Train with OpenPI
  python -m crane_x7_vla.training.cli train --backend openpi --data-root ./data --experiment-name my_experiment

  # Train with custom configuration file
  python -m crane_x7_vla.training.cli train --config my_config.yaml

  # Generate default configuration
  python -m crane_x7_vla.training.cli config --backend openvla --output openvla_config.yaml

  # Evaluate trained model
  python -m crane_x7_vla.training.cli evaluate --config my_config.yaml --checkpoint ./outputs/checkpoint-1000
        """
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a VLA model")
    train_parser.add_argument(
        "--backend",
        type=str,
        choices=["openvla", "openpi"],
        default="openvla",
        help="VLA backend to use"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)"
    )
    train_parser.add_argument(
        "--data-root",
        type=str,
        help="Path to training data directory"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    train_parser.add_argument(
        "--experiment-name",
        type=str,
        default="crane_x7_vla",
        help="Experiment name"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage"
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        help="LoRA rank (for parameter-efficient fine-tuning)"
    )
    train_parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use quantization (4-bit/8-bit) for memory efficiency"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML)"
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Generate default configuration file")
    config_parser.add_argument(
        "--backend",
        type=str,
        choices=["openvla", "openpi"],
        required=True,
        help="VLA backend"
    )
    config_parser.add_argument(
        "--output",
        type=str,
        default="config.yaml",
        help="Output configuration file path"
    )
    config_parser.add_argument(
        "--data-root",
        type=str,
        help="Path to training data directory"
    )
    config_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for checkpoints and logs"
    )
    config_parser.add_argument(
        "--experiment-name",
        type=str,
        default="crane_x7_vla",
        help="Experiment name"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "config":
        config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
