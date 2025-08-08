"""Logging utilities module for training and debugging.

This module provides centralized logging configuration for the entire
skin lesion classification system. It sets up structured logging with
both console and file outputs, ensuring consistent log formatting and
proper log management across all components.

Key features:
    - Dual output: Console and file logging
    - Configurable log levels
    - Automatic log directory creation
    - Consistent timestamp formatting
    - Colored console output (optional)

The logging system helps with:
    1. Training progress monitoring
    2. Error tracking and debugging
    3. Performance metric logging
    4. System state tracking
    5. Reproducibility through detailed logs

Typical usage:
    logger = setup_logger(\n        name='training',
        log_file='logs/training.log',
        level=logging.INFO
    )
    
    logger.info('Starting training...')
    logger.debug('Model parameters: %d', param_count)
    logger.error('Failed to load checkpoint: %s', error)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optionally file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name."""
    return logging.getLogger(name)