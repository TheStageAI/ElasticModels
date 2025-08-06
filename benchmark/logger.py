import logging
import os
import time

# ------------------------------------------------------------------------------------
# Levels, init main library logger
# ------------------------------------------------------------------------------------
_LOGGING_LEVELS_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_LOGGER_MAIN = logging.getLogger("[ElasticModels Benchmark]")


# ------------------------------------------------------------------------------------
# Logging utils
# ------------------------------------------------------------------------------------
def _get_std_formatter() -> logging.Formatter:
    """
    Returns logging formatter 'date time: name: level: msg'.
    """
    formatter = logging.Formatter(
        "%(asctime)s" ": %(name)s: %(levelname)s: %(message)s", "%H:%M:%S"
    )

    return formatter


def _get_console_handler() -> logging.StreamHandler:
    """
    Setups and returns console handler.
    """
    formatter = _get_std_formatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    return handler


def _get_file_handler(
    log_path: str = "./logs", name: str = "log"
) -> logging.FileHandler:
    """
    Setups and returns file handler.
    """
    formatter = _get_std_formatter()
    log_file_name = "{:}_{:}".format(name, time.time())
    log_file_name = os.path.join(log_path, log_file_name)
    handler = logging.FileHandler(log_file_name)
    handler.setFormatter(formatter)

    return handler


def _set_console_handler() -> None:
    """
    Set console handler with std formatter.
    """
    global _LOGGER_MAIN

    handler = _get_console_handler()
    _LOGGER_MAIN.addHandler(handler)


def _set_file_handler(log_path: str = "./logs", name: str = "log") -> None:
    """
    Set file handler with std formatter.
    """
    global _LOGGER_MAIN

    handler = _get_file_handler(log_path, name)
    _LOGGER_MAIN.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Returns child logger from the QLIP main logger.

    Parameters
    ----------
    name: str. Best practise is to pass `__name__` of the module.

    Examples
    --------
    from qlip.logger import get_logger

    MYLOGGER = get_logger(__name__)
    MYLOGGER.info('Logger was initialized.')
    """
    global _LOGGER_MAIN

    return _LOGGER_MAIN.getChild(name)


def set_logging_level(level: str = "INFO") -> None:
    """
    Sets logging level for library logger.

    Parameters
    ----------
    level: str. Available: DEBUG, INFO, WARNING, ERROR.

    Examples
    --------
    from qlip.logger import set_logging_level

    # By default QLIP logger has `INFO` logging level
    MYLOGGER = get_logger(__name__)
    # This message will not be printed, because `INFO` > `DEBUG`
    MYLOGGER.debug('Debug: Logger was initialized.')
    MYLOGGER.info('INFO: Logger was initialized.')

    set_logging_level('DEBUG')
    # Both messages will be printed
    MYLOGGER.debug('Debug: Logger was initialized.')
    MYLOGGER.info('INFO: Logger was initialized.')
    """
    global _LOGGER_MAIN
    global _LOGGING_LEVELS_MAP

    _LOGGER_MAIN.setLevel(_LOGGING_LEVELS_MAP[level])


def set_name(name: str) -> None:
    """ """
    global _LOGGER_MAIN

    _LOGGER_MAIN.name = name


_set_console_handler()
set_logging_level('INFO')