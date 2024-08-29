import logging
from typing import Literal

import colorlog


def get_logger(name: str, level: Literal["error", "warning", "info", "debug"] = "info") -> logging.Logger:
    # Convert the level string to the corresponding logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the logger and configure colorlog
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"  # noqa: E501
        )
    )
    logger.addHandler(handler)
    return logger
