# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""VLA-RL training module."""

__all__ = ["VLARLTrainer"]


def __getattr__(name: str):
    """Load the trainer only when training is requested."""
    if name == "VLARLTrainer":
        from crane_x7_vla_rl.training.trainer import VLARLTrainer

        return VLARLTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
