#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Command-line interface for CRANE-X7 VLA training.

Provides a unified CLI for training different VLA backends (OpenVLA, OpenPI).
CLI arguments are automatically generated from configuration dataclasses.
"""

import argparse
import logging
import sys
from dataclasses import fields, is_dataclass, MISSING
from pathlib import Path
from typing import Dict, get_type_hints, get_origin, get_args, List, Optional, Union, Literal

from crane_x7_vla.config.base import (
    UnifiedVLAConfig, DataConfig, TrainingConfig, CameraConfig, OverfittingConfig
)
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


def _get_field_docstring(cls, field_name: str) -> str:
    """
    Extract docstring for a dataclass field from class source.

    Looks for a docstring immediately following the field definition.
    """
    import inspect
    try:
        source = inspect.getsource(cls)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            # Look for field definition
            if f'{field_name}:' in line or f'{field_name} :' in line:
                # Check next line for docstring
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') and next_line.endswith('"""'):
                        return next_line[3:-3].strip()
                    elif next_line.startswith("'''") and next_line.endswith("'''"):
                        return next_line[3:-3].strip()
        return ""
    except Exception:
        return ""


def _python_name_to_cli_name(name: str) -> str:
    """Convert Python variable name to CLI argument name (snake_case to kebab-case)."""
    return name.replace('_', '-')


def _get_base_type(type_hint) -> type:
    """Extract the base type from a type hint (handling Optional, Union, etc.)."""
    origin = get_origin(type_hint)

    if origin is Union:
        # Handle Optional[X] which is Union[X, None]
        args = get_args(type_hint)
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return _get_base_type(non_none_args[0])

    if origin is Literal:
        # For Literal types, return str
        return str

    if origin is list or origin is List:
        return list

    if type_hint in (int, float, str, bool, Path):
        return type_hint

    if type_hint is type(None):
        return str

    # Default to str for complex types
    if isinstance(type_hint, type):
        return type_hint

    return str


def _should_skip_field(field_name: str, field_type) -> bool:
    """Determine if a field should be skipped for CLI generation."""
    # Skip complex nested types that don't make sense as CLI args
    origin = get_origin(field_type)

    # Skip list types (like cameras)
    if origin is list or origin is List:
        return True

    # Skip dict types
    if origin is dict or origin is Dict:
        return True

    # Skip if the type is a dataclass (nested config)
    try:
        if is_dataclass(field_type):
            return True
    except TypeError:
        pass

    return False


def add_dataclass_args_to_parser(
    parser: argparse.ArgumentParser,
    dataclass_cls,
    prefix: str = "",
    exclude_fields: Optional[List[str]] = None,
) -> Dict[str, tuple]:
    """
    Automatically add CLI arguments for all fields in a dataclass.

    Args:
        parser: ArgumentParser to add arguments to
        dataclass_cls: Dataclass class to extract fields from
        prefix: Prefix for argument names (e.g., "training" -> "--training-batch-size")
        exclude_fields: List of field names to exclude

    Returns:
        Dictionary mapping CLI arg name to (config_path, field_name) for applying overrides
    """
    if exclude_fields is None:
        exclude_fields = []

    arg_mapping = {}
    type_hints = get_type_hints(dataclass_cls)

    for f in fields(dataclass_cls):
        if f.name in exclude_fields:
            continue

        field_type = type_hints.get(f.name, f.type)

        # Skip complex types
        if _should_skip_field(f.name, field_type):
            continue

        # Build argument name
        if prefix:
            arg_name = f"--{_python_name_to_cli_name(prefix)}-{_python_name_to_cli_name(f.name)}"
            config_path = f"{prefix}.{f.name}"
        else:
            arg_name = f"--{_python_name_to_cli_name(f.name)}"
            config_path = f.name

        # Get base type for argparse
        base_type = _get_base_type(field_type)

        # Get docstring as help text
        help_text = _get_field_docstring(dataclass_cls, f.name)

        # Get default value
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:
            default = f.default_factory()
        else:
            default = None

        # Add default to help text
        if default is not None and help_text:
            help_text = f"{help_text} (default: {default})"
        elif default is not None:
            help_text = f"(default: {default})"

        # Handle boolean fields specially
        if base_type is bool:
            parser.add_argument(
                arg_name,
                action="store_true",
                default=None,  # None means "not specified"
                help=help_text
            )
        elif base_type is Path:
            parser.add_argument(
                arg_name,
                type=str,
                default=None,
                help=help_text
            )
        else:
            parser.add_argument(
                arg_name,
                type=base_type,
                default=None,
                help=help_text
            )

        # Store mapping for later use
        arg_mapping[arg_name.lstrip('-').replace('-', '_')] = (config_path, f.name, base_type)

    return arg_mapping


def apply_cli_overrides(config, args: argparse.Namespace, arg_mapping: Dict[str, tuple]):
    """
    Apply CLI argument overrides to a configuration object.

    Args:
        config: Configuration object to modify
        args: Parsed CLI arguments
        arg_mapping: Mapping from arg name to (config_path, field_name, type)
    """
    for arg_name, (config_path, field_name, field_type) in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue

        # Navigate to the correct config object
        parts = config_path.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                break

        if obj is not None:
            # Convert Path if needed
            if field_type is Path and isinstance(value, str):
                value = Path(value)
            setattr(obj, parts[-1], value)


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


def train_command(args, arg_mappings: Dict[str, Dict[str, tuple]]):
    """Execute training command."""
    # Load or create configuration
    if args.config:
        # Load from YAML and determine backend
        config = UnifiedVLAConfig.from_yaml(args.config)

        # Re-create as proper backend-specific config
        if config.backend == "openvla":
            config = OpenVLAConfig.from_yaml(args.config)
        elif config.backend == "openpi":
            config = OpenPIConfig.from_yaml(args.config)
    else:
        if not args.data_root:
            raise ValueError("--data-root is required when not using --config")

        config = create_default_config(
            backend=args.backend,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name
        )

    # Apply all CLI overrides automatically
    for mapping in arg_mappings.values():
        apply_cli_overrides(config, args, mapping)

    # Handle backend-specific overrides for OpenVLA
    if config.backend == "openvla" and hasattr(config, 'openvla'):
        # Also update backend_config dict if it exists
        if config.backend_config is not None:
            for attr in ['lora_rank', 'lora_dropout', 'use_quantization', 'image_aug', 'skip_merge_on_save']:
                if hasattr(config.openvla, attr):
                    config.backend_config[attr] = getattr(config.openvla, attr)

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

  # Train with custom configuration file and override specific settings
  python -m crane_x7_vla.training.cli train --config my_config.yaml --training-batch-size 32 --training-learning-rate 1e-4

  # Override OpenVLA-specific settings
  python -m crane_x7_vla.training.cli train --config my_config.yaml --openvla-lora-rank 16 --openvla-lora-dropout 0.1

  # Override overfitting detection settings
  python -m crane_x7_vla.training.cli train --config my_config.yaml --overfitting-overfit-check-interval 1000

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

    # =====================
    # Train command
    # =====================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a VLA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Basic arguments
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
        help="Path to configuration file (YAML). CLI arguments override config file values."
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

    # Store argument mappings for applying overrides
    arg_mappings = {}

    # Add training config arguments with "training" prefix
    train_group = train_parser.add_argument_group('Training Configuration')
    arg_mappings['training'] = add_dataclass_args_to_parser(
        train_group,
        TrainingConfig,
        prefix="training"
    )

    # Add overfitting config arguments with "overfitting" prefix
    overfit_group = train_parser.add_argument_group('Overfitting Detection Configuration')
    arg_mappings['overfitting'] = add_dataclass_args_to_parser(
        overfit_group,
        OverfittingConfig,
        prefix="overfitting"
    )

    # Add data config arguments with "data" prefix (excluding complex types)
    data_group = train_parser.add_argument_group('Data Configuration')
    arg_mappings['data'] = add_dataclass_args_to_parser(
        data_group,
        DataConfig,
        prefix="data",
        exclude_fields=['cameras', 'data_root']  # data_root handled separately
    )

    # Add OpenVLA-specific arguments
    openvla_group = train_parser.add_argument_group('OpenVLA Configuration')
    arg_mappings['openvla'] = add_dataclass_args_to_parser(
        openvla_group,
        OpenVLASpecificConfig,
        prefix="openvla",
        exclude_fields=['lora_target_modules', 'action_range', 'image_size']  # Complex types
    )

    # Add OpenPI-specific arguments
    openpi_group = train_parser.add_argument_group('OpenPI Configuration')
    arg_mappings['openpi'] = add_dataclass_args_to_parser(
        openpi_group,
        OpenPISpecificConfig,
        prefix="openpi",
        exclude_fields=['image_size']  # Complex types
    )

    # =====================
    # Evaluate command
    # =====================
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

    # =====================
    # Config command
    # =====================
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
        train_command(args, arg_mappings)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "config":
        config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
