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
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

from crane_x7_vla.config.base import CameraConfig, DataConfig, OverfittingConfig, TrainingConfig, UnifiedVLAConfig
from crane_x7_vla.config.openpi_config import OpenPIConfig, OpenPISpecificConfig
from crane_x7_vla.config.openvla_config import OpenVLAConfig, OpenVLASpecificConfig
from crane_x7_vla.training.trainer import VLATrainer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _get_field_docstring(cls, field_name: str) -> str:
    """
    Extract docstring for a dataclass field from class source.

    Looks for a docstring immediately following the field definition.
    """
    import inspect

    try:
        source = inspect.getsource(cls)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            # Look for field definition and check next line for docstring
            if (f"{field_name}:" in line or f"{field_name} :" in line) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (next_line.startswith('"""') and next_line.endswith('"""')) or (
                    next_line.startswith("'''") and next_line.endswith("'''")
                ):
                    return next_line[3:-3].strip()
        return ""
    except Exception:
        return ""


def _python_name_to_cli_name(name: str) -> str:
    """Convert Python variable name to CLI argument name (snake_case to kebab-case)."""
    return name.replace("_", "-")


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

    if origin is list or origin is list:
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
    if origin is list or origin is list:
        return True

    # Skip dict types
    if origin is dict or origin is dict:
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
    exclude_fields: list[str] | None = None,
) -> dict[str, tuple]:
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
                help=help_text,
            )
        elif base_type is Path:
            parser.add_argument(arg_name, type=str, default=None, help=help_text)
        else:
            parser.add_argument(arg_name, type=base_type, default=None, help=help_text)

        # Store mapping for later use
        arg_mapping[arg_name.lstrip("-").replace("-", "_")] = (config_path, f.name, base_type)

    return arg_mapping


def apply_cli_overrides(config, args: argparse.Namespace, arg_mapping: dict[str, tuple]):
    """
    Apply CLI argument overrides to a configuration object.

    Args:
        config: Configuration object to modify
        args: Parsed CLI arguments
        arg_mapping: Mapping from arg name to (config_path, field_name, type)
    """
    for arg_name, (config_path, _field_name, field_type) in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue

        # Navigate to the correct config object
        parts = config_path.split(".")
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


def create_default_config(backend: str, data_root: Path, output_dir: Path, experiment_name: str) -> UnifiedVLAConfig:
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
    cameras = [CameraConfig(name="primary", topic="/camera/color/image_raw", width=640, height=480)]

    # Data configuration
    data_config = DataConfig(data_root=data_root, cameras=cameras)

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
            openvla=OpenVLASpecificConfig(),
        )
    elif backend == "openpi":
        config = OpenPIConfig(
            backend="openpi",
            data=data_config,
            training=training_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            openpi=OpenPISpecificConfig(),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return config


def _build_config_from_args(
    backend: str,
    args: argparse.Namespace,
    arg_mappings: dict[str, dict[str, tuple]],
    sweep_overrides: dict | None = None,
) -> UnifiedVLAConfig:
    """
    Build configuration from CLI arguments and optional sweep overrides.

    Args:
        backend: Backend type ('openvla' or 'openpi')
        args: Parsed CLI arguments
        arg_mappings: Mapping from arg name to (config_path, field_name, type)
        sweep_overrides: Optional dictionary of sweep parameter overrides

    Returns:
        UnifiedVLAConfig instance
    """
    # Load or create configuration
    if hasattr(args, "config") and args.config:
        # Load from YAML and determine backend
        config = UnifiedVLAConfig.from_yaml(args.config)

        # Re-create as proper backend-specific config
        if config.backend == "openvla":
            config = OpenVLAConfig.from_yaml(args.config)
        elif config.backend == "openpi":
            config = OpenPIConfig.from_yaml(args.config)

        # Warn if config backend doesn't match CLI subcommand
        if config.backend != backend:
            logging.warning(
                f"Config file specifies backend '{config.backend}' but CLI subcommand is '{backend}'. "
                f"Using '{backend}' as specified in CLI."
            )
            # Recreate with correct backend
            config = create_default_config(
                backend=backend,
                data_root=config.data.data_root,
                output_dir=config.output_dir,
                experiment_name=config.experiment_name,
            )
    else:
        data_root = getattr(args, "data_root", None)
        if not data_root:
            raise ValueError("--data-root is required when not using --config")

        config = create_default_config(
            backend=backend,
            data_root=Path(data_root),
            output_dir=Path(getattr(args, "output_dir", "./outputs")),
            experiment_name=getattr(args, "experiment_name", "crane_x7_vla"),
        )

    # Apply all CLI overrides automatically
    for mapping in arg_mappings.values():
        apply_cli_overrides(config, args, mapping)

    # Apply sweep overrides (these take precedence)
    if sweep_overrides:
        _apply_sweep_overrides(config, sweep_overrides)

    # Handle backend-specific overrides for OpenVLA
    if config.backend == "openvla" and hasattr(config, "openvla") and config.backend_config is not None:
        for attr in ["lora_rank", "lora_dropout", "use_quantization", "image_aug", "skip_merge_on_save"]:
            if hasattr(config.openvla, attr):
                config.backend_config[attr] = getattr(config.openvla, attr)

    return config


def _apply_sweep_overrides(config: UnifiedVLAConfig, sweep_params: dict) -> None:
    """
    Apply sweep parameter overrides to configuration.

    Maps sweep parameter names to configuration attributes.

    Args:
        config: Configuration to modify
        sweep_params: Dictionary of sweep parameters
    """
    # Mapping from sweep parameter name to config attribute path
    PARAM_MAP = {
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "batch_size"),
        "weight_decay": ("training", "weight_decay"),
        "warmup_steps": ("training", "warmup_steps"),
        "max_grad_norm": ("training", "max_grad_norm"),
        "gradient_accumulation_steps": ("training", "grad_accumulation_steps"),
        "lora_rank": ("openvla", "lora_rank"),
        "lora_alpha": ("openvla", "lora_alpha"),
        "lora_dropout": ("openvla", "lora_dropout"),
        "image_aug": ("openvla", "image_aug"),
    }

    for param_name, value in sweep_params.items():
        if param_name in PARAM_MAP:
            section, attr = PARAM_MAP[param_name]
            obj = getattr(config, section, None)
            if obj is not None and hasattr(obj, attr):
                setattr(obj, attr, value)
                logging.debug(f"Sweep override: {section}.{attr} = {value}")
        else:
            logging.warning(f"Unknown sweep parameter: {param_name}")


def train_command(args, arg_mappings: dict[str, dict[str, tuple]]):
    """Execute training command."""
    backend = args.backend

    # Build configuration from CLI arguments
    config = _build_config_from_args(backend, args, arg_mappings)

    # Save configuration
    config_save_path = Path(config.output_dir) / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_save_path)
    logging.info(f"Configuration saved to {config_save_path}")

    # Create trainer and start training
    trainer = VLATrainer(config)
    results = trainer.train()

    logging.info(f"Training completed: {results}")


def agent_command(args, arg_mappings: dict[str, dict[str, tuple]]):
    """
    Execute W&B sweep agent command.

    Runs wandb.agent() which will:
    1. Connect to the W&B sweep controller
    2. Receive hyperparameters for each run
    3. Execute training with the received parameters
    """
    try:
        import wandb
    except ImportError:
        logging.error("wandb is not installed. Please install it with: pip install wandb")
        sys.exit(1)

    backend = args.backend
    sweep_id = args.sweep_id
    entity = args.entity
    project = args.project
    count = args.count

    logging.info("=" * 60)
    logging.info("W&B Sweep Agent")
    logging.info("=" * 60)
    logging.info(f"Sweep ID: {sweep_id}")
    logging.info(f"Entity: {entity or '(default)'}")
    logging.info(f"Project: {project}")
    logging.info(f"Backend: {backend}")
    logging.info(f"Count: {count}")
    logging.info("=" * 60)

    def run_training():
        """Callback function for wandb.agent()."""
        # Initialize W&B run (automatically connected to sweep)
        run = wandb.init()

        try:
            # Get sweep parameters
            sweep_params = dict(run.config)
            logging.info(f"Sweep parameters: {sweep_params}")

            # Build configuration with sweep overrides
            config = _build_config_from_args(backend, args, arg_mappings, sweep_overrides=sweep_params)

            # Save configuration
            config_save_path = Path(config.output_dir) / "config.yaml"
            config_save_path.parent.mkdir(parents=True, exist_ok=True)
            config.to_yaml(config_save_path)
            logging.info(f"Configuration saved to {config_save_path}")

            # Create trainer and start training
            trainer = VLATrainer(config)
            results = trainer.train()

            logging.info(f"Training completed: {results}")

        except Exception as e:
            logging.exception(f"Training failed: {e}")
            wandb.log({"training_failed": True, "error": str(e)})
            raise
        finally:
            wandb.finish()

    # Run the sweep agent
    wandb.agent(
        sweep_id=sweep_id,
        function=run_training,
        entity=entity if entity else None,
        project=project,
        count=count,
    )


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
    metrics = trainer.evaluate(checkpoint_path=args.checkpoint, test_data_path=args.test_data)

    logging.info(f"Evaluation metrics: {metrics}")


def config_command(args):
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


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments to a parser."""
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (YAML). CLI arguments override config file values."
    )
    parser.add_argument("--data-root", type=str, help="Path to training data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory for checkpoints and logs")
    parser.add_argument("--experiment-name", type=str, default="crane_x7_vla", help="Experiment name")


def _add_common_config_args(parser: argparse.ArgumentParser) -> dict[str, dict[str, tuple]]:
    """Add common configuration arguments to a parser and return arg mappings."""
    arg_mappings = {}

    # Add training config arguments with "training" prefix
    train_group = parser.add_argument_group("Training Configuration")
    arg_mappings["training"] = add_dataclass_args_to_parser(train_group, TrainingConfig, prefix="training")

    # Add overfitting config arguments with "overfitting" prefix
    overfit_group = parser.add_argument_group("Overfitting Detection Configuration")
    arg_mappings["overfitting"] = add_dataclass_args_to_parser(overfit_group, OverfittingConfig, prefix="overfitting")

    # Add data config arguments with "data" prefix (excluding complex types)
    data_group = parser.add_argument_group("Data Configuration")
    arg_mappings["data"] = add_dataclass_args_to_parser(
        data_group,
        DataConfig,
        prefix="data",
        exclude_fields=["cameras", "data_root"],  # data_root handled separately
    )

    return arg_mappings


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CRANE-X7 VLA Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with OpenVLA using default settings
  python -m crane_x7_vla.training.cli train openvla --data-root ./data --experiment-name my_experiment

  # Train with OpenPI
  python -m crane_x7_vla.training.cli train openpi --data-root ./data --experiment-name my_experiment

  # Train with custom configuration file and override specific settings
  python -m crane_x7_vla.training.cli train openvla --config my_config.yaml --training-batch-size 32

  # Override OpenVLA-specific settings (LoRA)
  python -m crane_x7_vla.training.cli train openvla --data-root ./data --lora-rank 16 --lora-dropout 0.1

  # Override OpenPI-specific settings
  python -m crane_x7_vla.training.cli train openpi --data-root ./data --action-chunk-size 50

  # Generate default configuration
  python -m crane_x7_vla.training.cli config --backend openvla --output openvla_config.yaml

  # Evaluate trained model
  python -m crane_x7_vla.training.cli evaluate --config my_config.yaml --checkpoint ./outputs/checkpoint-1000
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # =====================
    # Train command (with backend subcommands)
    # =====================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a VLA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_subparsers = train_parser.add_subparsers(dest="backend", help="VLA backend to use")

    # Store argument mappings for each backend
    all_arg_mappings = {}

    # ----- OpenVLA subcommand -----
    openvla_parser = train_subparsers.add_parser(
        "openvla",
        help="Train with OpenVLA backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_train_args(openvla_parser)
    all_arg_mappings["openvla"] = _add_common_config_args(openvla_parser)

    # Add OpenVLA-specific arguments (without prefix for cleaner CLI)
    openvla_group = openvla_parser.add_argument_group("OpenVLA Configuration")
    all_arg_mappings["openvla"]["openvla"] = add_dataclass_args_to_parser(
        openvla_group,
        OpenVLASpecificConfig,
        prefix="",  # No prefix for backend-specific args
        exclude_fields=["lora_target_modules", "action_range", "image_size"],  # Complex types
    )

    # ----- OpenPI subcommand -----
    openpi_parser = train_subparsers.add_parser(
        "openpi",
        help="Train with OpenPI backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_train_args(openpi_parser)
    all_arg_mappings["openpi"] = _add_common_config_args(openpi_parser)

    # Add OpenPI-specific arguments (without prefix for cleaner CLI)
    openpi_group = openpi_parser.add_argument_group("OpenPI Configuration")
    all_arg_mappings["openpi"]["openpi"] = add_dataclass_args_to_parser(
        openpi_group,
        OpenPISpecificConfig,
        prefix="",  # No prefix for backend-specific args
        exclude_fields=["image_size"],  # Complex types
    )

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
    config_parser.add_argument("--backend", type=str, choices=["openvla", "openpi"], required=True, help="VLA backend")
    config_parser.add_argument("--output", type=str, default="config.yaml", help="Output configuration file path")
    config_parser.add_argument("--data-root", type=str, help="Path to training data directory")
    config_parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints and logs")
    config_parser.add_argument("--experiment-name", type=str, default="crane_x7_vla", help="Experiment name")

    # =====================
    # Agent command (W&B Sweep)
    # =====================
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run W&B sweep agent for hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sweep agent for OpenVLA
  python -m crane_x7_vla.training.cli agent openvla --sweep-id abc123 --data-root ./data

  # Run sweep agent with specific entity/project
  python -m crane_x7_vla.training.cli agent openvla --sweep-id abc123 \\
      --entity my-team --project my-project --data-root ./data

  # Run multiple sweep runs
  python -m crane_x7_vla.training.cli agent openvla --sweep-id abc123 --count 5 --data-root ./data
        """,
    )
    agent_subparsers = agent_parser.add_subparsers(dest="backend", help="VLA backend to use")

    # Store agent argument mappings
    agent_arg_mappings = {}

    # ----- Agent OpenVLA subcommand -----
    agent_openvla_parser = agent_subparsers.add_parser(
        "openvla",
        help="Run sweep agent with OpenVLA backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    agent_openvla_parser.add_argument("--sweep-id", type=str, required=True, help="W&B Sweep ID")
    agent_openvla_parser.add_argument("--entity", type=str, help="W&B entity (team/username)")
    agent_openvla_parser.add_argument("--project", type=str, default="crane_x7", help="W&B project name")
    agent_openvla_parser.add_argument("--count", type=int, default=1, help="Number of runs to execute")
    _add_common_train_args(agent_openvla_parser)
    agent_arg_mappings["openvla"] = _add_common_config_args(agent_openvla_parser)

    # Add OpenVLA-specific arguments
    agent_openvla_group = agent_openvla_parser.add_argument_group("OpenVLA Configuration")
    agent_arg_mappings["openvla"]["openvla"] = add_dataclass_args_to_parser(
        agent_openvla_group,
        OpenVLASpecificConfig,
        prefix="",
        exclude_fields=["lora_target_modules", "action_range", "image_size"],
    )

    # ----- Agent OpenPI subcommand -----
    agent_openpi_parser = agent_subparsers.add_parser(
        "openpi",
        help="Run sweep agent with OpenPI backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    agent_openpi_parser.add_argument("--sweep-id", type=str, required=True, help="W&B Sweep ID")
    agent_openpi_parser.add_argument("--entity", type=str, help="W&B entity (team/username)")
    agent_openpi_parser.add_argument("--project", type=str, default="crane_x7", help="W&B project name")
    agent_openpi_parser.add_argument("--count", type=int, default=1, help="Number of runs to execute")
    _add_common_train_args(agent_openpi_parser)
    agent_arg_mappings["openpi"] = _add_common_config_args(agent_openpi_parser)

    # Add OpenPI-specific arguments
    agent_openpi_group = agent_openpi_parser.add_argument_group("OpenPI Configuration")
    agent_arg_mappings["openpi"]["openpi"] = add_dataclass_args_to_parser(
        agent_openpi_group,
        OpenPISpecificConfig,
        prefix="",
        exclude_fields=["image_size"],
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "train":
        if args.backend is None:
            train_parser.print_help()
            sys.exit(1)
        train_command(args, all_arg_mappings.get(args.backend, {}))
    elif args.command == "agent":
        if args.backend is None:
            agent_parser.print_help()
            sys.exit(1)
        agent_command(args, agent_arg_mappings.get(args.backend, {}))
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "config":
        config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
