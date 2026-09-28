# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""OpenVLA backend for CRANE-X7 VLA training."""

from crane_x7_vla.backends.openvla.config import OpenVLAConfig, OpenVLASpecificConfig


def __getattr__(name: str):
    """Delay model and dataset imports until the backend is used."""
    if name == "OpenVLABackend":
        from crane_x7_vla.backends.openvla.backend import OpenVLABackend

        return OpenVLABackend
    if name in {"CraneX7BatchTransform", "CraneX7Dataset", "CraneX7DatasetConfig"}:
        from crane_x7_vla.backends.openvla import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CraneX7BatchTransform",
    "CraneX7Dataset",
    "CraneX7DatasetConfig",
    "OpenVLABackend",
    "OpenVLAConfig",
    "OpenVLASpecificConfig",
]
