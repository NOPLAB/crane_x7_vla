"""
CRANE-X7 VLA Training Framework

A unified framework for training Vision-Language-Action models on CRANE-X7 robot data.
Supports both OpenVLA and OpenPI backends with automatic data conversion.

Copyright (c) 2025 nop
Licensed under the MIT License
"""

from crane_x7_vla.training.trainer import VLATrainer
from crane_x7_vla.config.base import UnifiedVLAConfig

__version__ = "0.1.0"

__all__ = [
    "VLATrainer",
    "UnifiedVLAConfig",
]
