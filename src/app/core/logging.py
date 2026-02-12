from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class LoggingConfig:
    level: int = logging.INFO
    format: str = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


LOGGER_NAME: Final[str] = "antifraud"


def configure_logging(cfg: LoggingConfig | None = None) -> None:
    """Configure root logging with a consistent formatter."""
    config = cfg or LoggingConfig()

    root = logging.getLogger()
    root.setLevel(config.level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(config.level)

    formatter = logging.Formatter(fmt=config.format, datefmt=config.datefmt)
    handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
