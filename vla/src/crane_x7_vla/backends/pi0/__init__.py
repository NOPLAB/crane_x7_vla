# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Pi0 Backend for CRANE-X7.

This module implements Pi0 and Pi0.5 models for VLA training using PyTorch.
Based on the OpenPI implementation with PaliGemma + Expert Gemma architecture.

Pi0: Uses continuous state input and MLP for timestep processing
Pi0.5: Uses discrete state tokens and adaRMSNorm for timestep injection
"""

from crane_x7_vla.backends.pi0.config import Pi0Config, Pi0SpecificConfig


def __getattr__(name: str):
    """Delay model imports until the backend is used."""
    if name == "Pi0Backend":
        from crane_x7_vla.backends.pi0.backend import Pi0Backend

        return Pi0Backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Pi0Backend",
    "Pi0Config",
    "Pi0SpecificConfig",
]
