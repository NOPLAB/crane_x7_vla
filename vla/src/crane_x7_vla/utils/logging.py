# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Unified logging utilities for crane_x7_vla."""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if format_string is None:
            format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def set_log_level(name: str, level: int) -> None:
    """
    Set the log level for a specific logger.

    Args:
        name: Logger name
        level: Logging level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)


def disable_duplicate_logs() -> None:
    """
    Disable duplicate logs that may appear from parent loggers.

    Call this after configuring all loggers to prevent duplicate messages.
    """
    # Prevent logs from propagating to the root logger
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("crane_x7_vla"):
            logging.getLogger(name).propagate = False
