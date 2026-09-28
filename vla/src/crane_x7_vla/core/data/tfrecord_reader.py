# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""
Pure Python TFRecord reader without TensorFlow dependency.

Uses the tfrecord library (pip install tfrecord) to read TFRecord files
in a format compatible with the existing CRANE-X7 dataset structure.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


try:
    from tfrecord import tfrecord_loader
except ImportError as e:
    raise ImportError("tfrecord library is required. Install with: pip install tfrecord") from e


logger = logging.getLogger(__name__)


class TFRecordReader:
    """
    Pure Python TFRecord reader using tfrecord library.

    This class provides a TensorFlow-free way to read TFRecord files
    that were created with the crane_x7_log package.

    Example:
        reader = TFRecordReader(
            file_paths=[Path("episode_001/episode_data.tfrecord")],
            feature_spec=TFRecordReader.CRANE_X7_FEATURES
        )
        for example in reader:
            print(example["action"])
    """

    # Standard CRANE-X7 feature specification for tfrecord library
    # Format: {"key": "type"} where type is "byte", "float", or "int"
    CRANE_X7_FEATURES: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "observation/timestep": "int",
        "action": "float",
        "task/language_instruction": "byte",
        "dataset_name": "byte",
    }

    # Alternative feature names for backward compatibility
    CRANE_X7_FEATURES_ALT: ClassVar[dict[str, str]] = {
        "observation/state": "float",
        "observation/image": "byte",
        "action": "float",
        "prompt": "byte",
        "task": "byte",
    }

    # Extended features including optional multi-camera support
    CRANE_X7_FEATURES_EXTENDED: ClassVar[dict[str, str]] = {
        "observation/proprio": "float",
        "observation/image_primary": "byte",
        "observation/image_secondary": "byte",
        "observation/image_wrist": "byte",
        "observation/timestep": "int",
        "observation/depth": "byte",
        "observation/timestamp": "float",
        "action": "float",
        "task/language_instruction": "byte",
        "dataset_name": "byte",
        "episode_id": "int",
        "step_id": "int",
    }

    def __init__(
        self,
        file_paths: list[Path] | list[str],
        feature_spec: dict[str, str] | None = None,
        use_alternative_keys: bool = False,
    ):
        """
        Initialize TFRecord reader.

        Args:
            file_paths: List of paths to TFRecord files
            feature_spec: Feature specification dict. If None, uses CRANE_X7_FEATURES
            use_alternative_keys: If True, try alternative feature names on failure
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.feature_spec = feature_spec or self.CRANE_X7_FEATURES
        self.use_alternative_keys = use_alternative_keys

        # Validate file paths
        for path in self.file_paths:
            if not path.exists():
                logger.warning(f"TFRecord file not found: {path}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all examples in all TFRecord files."""
        for path in self.file_paths:
            if not path.exists():
                continue
            yield from self._read_file(path)

    def _read_file(self, path: Path) -> Iterator[dict[str, Any]]:
        """
        Read examples from a single TFRecord file.

        Args:
            path: Path to TFRecord file

        Yields:
            Parsed example dictionaries
        """
        # Feature specs in the backends include optional cameras. Reading only
        # fields stored in each example supports episodes with different cameras
        # without restarting a partially consumed file or duplicating samples.
        loader = tfrecord_loader(data_path=str(path), index_path=None, description=None)
        for example in loader:
            if self.use_alternative_keys:
                yield self._normalize_alternative_keys(example)
            else:
                yield self._post_process_example(example)

    def _post_process_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """
        Post-process parsed example.

        Converts numpy arrays to appropriate shapes based on known dimensions.

        Args:
            example: Raw parsed example

        Returns:
            Post-processed example
        """
        result = {}
        for key, value in example.items():
            if value is None:
                result[key] = None
                continue

            if isinstance(value, np.ndarray):
                # Reshape known fixed-length features
                if key in ("observation/proprio", "observation/state", "action"):
                    # These are 8-dimensional vectors
                    if value.size == 8:
                        result[key] = value.reshape(8).astype(np.float32)
                    else:
                        result[key] = value.astype(np.float32)
                elif key == "observation/timestep":
                    result[key] = value.reshape(-1).astype(np.int64)
                elif key == "observation/timestamp":
                    result[key] = value.reshape(-1).astype(np.float32)
                elif key in ("episode_id", "step_id"):
                    result[key] = int(value.flatten()[0]) if value.size > 0 else 0
                else:
                    result[key] = value
            elif isinstance(value, bytes):
                result[key] = value
            else:
                result[key] = value

        return result

    def _normalize_alternative_keys(self, example: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize alternative key names to standard names.

        Args:
            example: Example with alternative keys

        Returns:
            Example with standardized keys
        """
        key_mapping = {
            "observation/state": "observation/proprio",
            "observation/image": "observation/image_primary",
            "prompt": "task/language_instruction",
            "task": "dataset_name",
        }

        result = {}
        for key, value in example.items():
            normalized_key = key_mapping.get(key, key)
            result[normalized_key] = value

        return self._post_process_example(result)

    def count_records(self) -> int:
        """Count total number of records across all files."""
        count = 0
        for path in self.file_paths:
            if not path.exists():
                continue
            try:
                loader = tfrecord_loader(data_path=str(path), index_path=None, description=None)
                count += sum(1 for _ in loader)
            except Exception as e:
                logger.warning(f"Failed to count records in {path}: {e}")
        return count


def find_tfrecord_files(
    data_root: Path | str,
    patterns: list[str] | None = None,
) -> list[Path]:
    """
    Find all TFRecord files in a directory.

    Args:
        data_root: Root directory to search
        patterns: Glob patterns to match. Defaults to common patterns.

    Returns:
        List of found TFRecord file paths, sorted by name
    """
    data_root = Path(data_root)

    if patterns is None:
        patterns = [
            "episode_*/episode_data.tfrecord",
            "**/*.tfrecord",
        ]

    files = set()
    for pattern in patterns:
        files.update(data_root.glob(pattern))

    # Filter out backup files
    files = [f for f in files if not f.name.endswith(".bak")]

    return sorted(files)
