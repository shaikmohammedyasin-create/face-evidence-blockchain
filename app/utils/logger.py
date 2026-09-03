"""
app/utils/logger.py — Structured, levelled logger for the whole application.
"""
from __future__ import annotations

import logging
import sys
from app.config import LOG_LEVEL

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configured once at process start."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
