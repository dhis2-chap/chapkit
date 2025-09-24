from contextlib import contextmanager
import copy
import logging.config
import time
from typing import Sequence

import structlog

# Processors for logs from standard library loggers like uvicorn.
stdlib_processors = [
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
]

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "structlog.stdlib.ProcessorFormatter",
            "processor": structlog.dev.ConsoleRenderer(colors=True),
            "foreign_pre_chain": stdlib_processors,
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


STRUCTLOG_PROCESSORS = [
    structlog.stdlib.filter_by_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
]


def configure_logging(
    *,
    use_json: bool = False,
    level: str = "INFO",
    colors: bool = True,
    structlog_processors: Sequence | None = None,
    cache_logger_on_first_use: bool = True,
) -> None:
    """Configure standard logging and structlog with the shared settings.

    Parameters
    ----------
    use_json
        When ``True`` use ``JSONRenderer`` instead of ``ConsoleRenderer`` for log output.
    level
        Minimum log level applied to the default handler and known loggers.
    colors
        Enable ANSI colors for console output (ignored when ``use_json`` is ``True``).
    structlog_processors
        Optional override of the processor pipeline used by structlog.
    cache_logger_on_first_use
        Whether structlog should cache loggers the first time they are created.
    """

    config = copy.deepcopy(LOGGING_CONFIG)

    config["handlers"]["default"]["level"] = level
    config["loggers"][""]["level"] = level

    if "uvicorn.error" in config["loggers"]:
        config["loggers"]["uvicorn.error"]["level"] = level
    if "uvicorn.access" in config["loggers"]:
        config["loggers"]["uvicorn.access"]["level"] = level

    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=colors)

    config["formatters"]["default"]["processor"] = renderer

    logging.config.dictConfig(config)

    processors = list(structlog_processors or STRUCTLOG_PROCESSORS)

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=cache_logger_on_first_use,
    )


@contextmanager
def log_time(event: str, **kwargs):
    start = time.perf_counter()
    log = structlog.get_logger()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info(event, elapsed_seconds=elapsed, **kwargs)
