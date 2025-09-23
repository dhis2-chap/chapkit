from contextlib import contextmanager
import time
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
            # "processor": structlog.processors.JSONRenderer(),
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


@contextmanager
def log_time(event: str, **kwargs):
    start = time.perf_counter()
    log = structlog.get_logger()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info(event, elapsed_seconds=elapsed, **kwargs)
