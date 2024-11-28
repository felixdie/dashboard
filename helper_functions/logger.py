import logging


def get_logger() -> logging.Logger:
    """
    Creates a logger to debug in console.

    Parameters:
        None.

    Returns:
        logger: (logging.Logger): Configured logger instance that outputs messages to the console.
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
