"""Logger con el formato común de los pipelines y la indexación."""

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"


def get_logger(name: str, file: str | None = None, propagate: bool = True) -> logging.Logger:
    """Logger a nivel INFO que escribe a `file`, o a stderr si no se pasa ninguno."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = propagate
    if not logger.handlers:
        handler = logging.FileHandler(file) if file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger
