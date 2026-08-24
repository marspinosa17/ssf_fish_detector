"""Shared logging setup, matching preprocessing/preprocess.py's setup_logging()."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: Path, log_name: str, logger_name: str) -> logging.Logger:
    """Configure a logger that writes to both stdout and output_dir/{log_name}.

    Returns the configured logger; call once per entry point (train.py, eval.py).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / log_name, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
