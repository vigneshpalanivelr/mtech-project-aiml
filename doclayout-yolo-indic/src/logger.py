"""
Centralized logging for DocLayout-YOLO-Indic.

Usage:
    >>> from src.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting generation")
"""

import logging
import sys

_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("doclayout_indic")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a namespaced logger under the project root logger."""
    _configure_root(level)
    short = name.split(".")[-1] if name else "main"
    return logging.getLogger(f"doclayout_indic.{short}")
