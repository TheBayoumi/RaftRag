"""
Logging configuration using Loguru.

This module sets up structured logging with file rotation and different log levels.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path.
        rotation: Log file rotation size.
        retention: Log file retention period.
        compression: Log file compression format.

    Returns:
        None
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        # Log directory is created in ensure_directories() on startup
        # This is defensive in case logging is set up before ensure_directories()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
        )

    logger.info(f"Logging configured at {level} level")


def get_logger(name: str) -> logger:
    """
    Get a logger instance with a specific name.

    Args:
        name: Name for the logger (usually module name).

    Returns:
        logger: Configured logger instance.
    """
    return logger.bind(name=name)
