# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""VLA backend implementations for different models."""

from typing import Literal, Type

from crane_x7_vla.backends.base import VLABackend

# Type alias for backend names
BackendType = Literal["openvla", "openpi", "openpi-pytorch"]


def get_backend(backend_type: BackendType) -> Type[VLABackend]:
    """
    Get backend class by name with lazy loading.

    This function delays importing backend modules until they are actually
    needed, avoiding dependency issues when not all backends are installed.

    Args:
        backend_type: Name of the backend ("openvla", "openpi", "openpi-pytorch")

    Returns:
        Backend class (not instantiated)

    Raises:
        ValueError: If backend_type is unknown
        ImportError: If required dependencies for the backend are not installed

    Example:
        >>> BackendClass = get_backend("openvla")
        >>> backend = BackendClass(config)
        >>> backend.train()
    """
    if backend_type == "openvla":
        from crane_x7_vla.backends.openvla import OpenVLABackend

        return OpenVLABackend
    elif backend_type == "openpi":
        from crane_x7_vla.backends.openpi import OpenPIBackend

        return OpenPIBackend
    elif backend_type == "openpi-pytorch":
        from crane_x7_vla.backends.openpi_pytorch import OpenPIPytorchBackend

        return OpenPIPytorchBackend
    else:
        raise ValueError(
            f"Unknown backend: {backend_type}. "
            f"Available backends: openvla, openpi, openpi-pytorch"
        )


__all__ = [
    "VLABackend",
    "get_backend",
    "BackendType",
]
