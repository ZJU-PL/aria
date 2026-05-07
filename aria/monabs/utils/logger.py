"""Logging helpers for MPA utilities."""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logger(
    name: str = "mpa",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name.
        log_file: Optional file path to log to.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
