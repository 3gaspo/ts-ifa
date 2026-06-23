"""Shared runtime helpers for command-line experiments."""

from __future__ import annotations

import logging
import sys


EXPERIMENT_SEPARATOR = "=" * 72


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


def log_experiment_separator(logger: logging.Logger) -> None:
    """Write the shared visual boundary used between experiment runs."""
    logger.info("%s", EXPERIMENT_SEPARATOR)
