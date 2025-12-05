# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Data loading and conversion utilities."""

from crane_x7_vla.data.crane_x7_dataset import (
    CraneX7BatchTransform,
    CraneX7Dataset,
    CraneX7DatasetConfig,
)
from crane_x7_vla.data.adapters import CraneX7DataAdapter
from crane_x7_vla.data.converters import (
    LeRobotDataset,
    TFRecordToLeRobotConverter,
)

# OpenPI-specific data config (optional import)
try:
    from crane_x7_vla.data.openpi_data_config import (
        CraneX7DataConfigFactory,
        CraneX7LeRobotDataConfig,
        CraneX7Inputs,
        CraneX7Outputs,
    )
    _OPENPI_DATA_AVAILABLE = True
except ImportError:
    _OPENPI_DATA_AVAILABLE = False

__all__ = [
    "CraneX7Dataset",
    "CraneX7BatchTransform",
    "CraneX7DatasetConfig",
    "CraneX7DataAdapter",
    "LeRobotDataset",
    "TFRecordToLeRobotConverter",
]

if _OPENPI_DATA_AVAILABLE:
    __all__.extend([
        "CraneX7DataConfigFactory",
        "CraneX7LeRobotDataConfig",
        "CraneX7Inputs",
        "CraneX7Outputs",
    ])
