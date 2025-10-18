"""Gunicorn configuration file with structured logging."""

import logging
import os
import sys

import structlog
from structlog.typing import Processor

# Configure structured logging BEFORE Gunicorn creates any loggers
log_format = os.getenv("LOG_FORMAT", "console").lower()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, log_level, logging.INFO)

# Shared processors for structlog
shared_processors: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
]

# Choose renderer based on format
if log_format == "json":
    formatter_processors = shared_processors + [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]
else:
    formatter_processors = shared_processors + [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        structlog.processors.ExceptionRenderer(),
        structlog.dev.ConsoleRenderer(colors=True),
    ]

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    wrapper_class=structlog.make_filtering_bound_logger(level),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Gunicorn logging configuration - completely override default logging
# This prevents Gunicorn from creating its own handlers
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structlog": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": formatter_processors,
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "structlog",
        },
    },
    "loggers": {
        "gunicorn.error": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["default"],
        "level": log_level,
    },
}
