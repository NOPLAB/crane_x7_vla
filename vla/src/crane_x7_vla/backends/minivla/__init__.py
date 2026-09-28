# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""MiniVLA backend for CRANE-X7 VLA training.

MiniVLA features:
- Qwen 2.5 0.5B LLM backbone (~1B total params, 7x smaller than OpenVLA)
- VQ Action Chunking for efficient multi-step prediction
- Multi-image input support (history + wrist camera)
- ~12.5Hz inference (2.5x faster than OpenVLA)
"""

from crane_x7_vla.backends.minivla.config import (
    MiniVLAConfig,
    MiniVLASpecificConfig,
    MultiImageConfig,
    VQConfig,
)


def __getattr__(name: str):
    """Delay model and dataset imports until the backend is used."""
    from importlib import import_module

    modules = {
        "action_tokenizer": {"BinActionTokenizer", "ResidualVQ", "VectorQuantize", "VQActionTokenizer"},
        "backend": {"MiniVLABackend", "MiniVLAFinetuneConfig", "MiniVLAModel"},
        "dataset": {"MiniVLABatchTransform", "MiniVLADataset", "MiniVLADatasetConfig"},
    }
    for module, names in modules.items():
        if name in names:
            return getattr(import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BinActionTokenizer",
    "MiniVLABackend",
    "MiniVLABatchTransform",
    "MiniVLAConfig",
    "MiniVLADataset",
    "MiniVLADatasetConfig",
    "MiniVLAFinetuneConfig",
    "MiniVLAModel",
    "MiniVLASpecificConfig",
    "MultiImageConfig",
    "ResidualVQ",
    "VQActionTokenizer",
    "VQConfig",
    "VectorQuantize",
]
