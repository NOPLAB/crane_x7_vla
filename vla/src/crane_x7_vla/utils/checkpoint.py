# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Checkpoint validation and management utilities."""

import json
from pathlib import Path
from typing import Literal

from crane_x7_vla.data_types import CheckpointInfo
from crane_x7_vla.utils.logging import get_logger

logger = get_logger(__name__)

BackendType = Literal["openvla", "openpi", "openpi-pytorch"]


def detect_backend(path: Path) -> BackendType | None:
    """
    Detect which backend created a checkpoint.

    Args:
        path: Path to checkpoint directory

    Returns:
        Backend type string or None if cannot be detected
    """
    path = Path(path)

    # OpenVLA: Has adapter_config.json (LoRA) or config.json with OpenVLA structure
    if (path / "adapter_config.json").exists():
        return "openvla"
    if (path / "lora_adapters" / "adapter_config.json").exists():
        return "openvla"

    # OpenPI JAX: Has 'params' directory or 'train_state.msgpack'
    if (path / "params").exists() or (path / "train_state.msgpack").exists():
        return "openpi"

    # OpenPI PyTorch: Has checkpoint.pt
    if (path / "checkpoint.pt").exists():
        return "openpi-pytorch"

    # OpenVLA full model: Has model.safetensors or pytorch_model.bin
    if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
        return "openvla"

    return None


def validate_checkpoint(path: Path | str) -> CheckpointInfo:
    """
    Validate a checkpoint and extract information.

    Args:
        path: Path to checkpoint directory

    Returns:
        CheckpointInfo with validation results

    Raises:
        FileNotFoundError: If checkpoint path does not exist
        ValueError: If checkpoint format cannot be determined
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Detect backend
    backend = detect_backend(path)
    if backend is None:
        raise ValueError(
            f"Cannot determine checkpoint format: {path}. "
            "Expected OpenVLA, OpenPI (JAX), or OpenPI (PyTorch) format."
        )

    # Check for config
    has_config = (
        (path / "config.json").exists()
        or (path / "trainer_config.json").exists()
    )

    # Check for optimizer state
    has_optimizer_state = False
    if backend == "openvla":
        has_optimizer_state = (path / "optimizer.pt").exists()
    elif backend == "openpi":
        has_optimizer_state = (path / "opt_state.msgpack").exists()
    elif backend == "openpi-pytorch":
        # Optimizer state is inside checkpoint.pt
        has_optimizer_state = True

    # Extract step number from directory name or config
    step = _extract_step(path, backend)

    # Validate checkpoint contents
    is_valid = _validate_checkpoint_contents(path, backend)

    return CheckpointInfo(
        path=path,
        backend=backend,
        step=step,
        is_valid=is_valid,
        has_optimizer_state=has_optimizer_state,
        has_config=has_config,
    )


def _extract_step(path: Path, backend: BackendType) -> int:
    """Extract training step from checkpoint path or config."""
    # Try extracting from directory name (e.g., checkpoint-5000)
    dir_name = path.name
    if dir_name.startswith("checkpoint-"):
        try:
            return int(dir_name.split("-")[1])
        except (ValueError, IndexError):
            pass

    # Try extracting from config file
    config_files = ["config.json", "trainer_config.json"]
    for config_file in config_files:
        config_path = path / config_file
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if "global_step" in config:
                    return config["global_step"]
                if "step" in config:
                    return config["step"]
            except (json.JSONDecodeError, KeyError):
                pass

    return 0


def _validate_checkpoint_contents(path: Path, backend: BackendType) -> bool:
    """Validate that checkpoint contains expected files."""
    if backend == "openvla":
        # OpenVLA needs either model weights or LoRA adapters
        has_weights = (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
            or (path / "lora_adapters").exists()
            or (path / "adapter_config.json").exists()
        )
        return has_weights

    elif backend == "openpi":
        # OpenPI JAX needs params
        return (path / "params").exists() or (path / "train_state.msgpack").exists()

    elif backend == "openpi-pytorch":
        # OpenPI PyTorch needs checkpoint.pt
        return (path / "checkpoint.pt").exists()

    return False


def list_checkpoints(run_dir: Path | str) -> list[CheckpointInfo]:
    """
    List all checkpoints in a run directory.

    Args:
        run_dir: Path to run directory containing checkpoints

    Returns:
        List of CheckpointInfo for each valid checkpoint, sorted by step
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        return []

    checkpoints = []

    # Look for checkpoint directories
    for item in run_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint"):
            try:
                info = validate_checkpoint(item)
                checkpoints.append(info)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping invalid checkpoint {item}: {e}")

    # Sort by step
    checkpoints.sort(key=lambda c: c.step)

    return checkpoints


def get_latest_checkpoint(run_dir: Path | str) -> CheckpointInfo | None:
    """
    Get the latest checkpoint from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        CheckpointInfo for latest checkpoint, or None if no checkpoints found
    """
    checkpoints = list_checkpoints(run_dir)
    return checkpoints[-1] if checkpoints else None
