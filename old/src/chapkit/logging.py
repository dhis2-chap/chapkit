# logging.py
from __future__ import annotations

import copy
import logging
import logging.config
import os
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Sequence
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

# ----------------------------
# Pre-chain for stdlib loggers
# ----------------------------
# Applied by ProcessorFormatter to stdlib LogRecords (uvicorn.*, gunicorn.*).
stdlib_processors: list = [
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]

LOGGING_CONFIG: dict = {
    "version": 1,
    # Critical to avoid duplicate handlers left around by gunicorn/uvicorn defaults
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "()": "structlog.stdlib.ProcessorFormatter",
            # This gets swapped to JSONRenderer if LOG_JSON=1 (see configure_logging)
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
    # IMPORTANT: propagate=False everywhere so each record renders once
    "loggers": {
        # Root (your app)
        "": {"handlers": ["default"], "level": "INFO", "propagate": False},
        # Uvicorn
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "uvicorn.asgi": {"handlers": ["default"], "level": "INFO", "propagate": False},
        # Gunicorn
        "gunicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "gunicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        # Keep gunicorn.access quiet by default (Uvicorn will log access)
        "gunicorn.access": {"handlers": ["default"], "level": "ERROR", "propagate": False},
    },
}


# ----------------------------
# Structlog processor pipeline
# ----------------------------
def _base_structlog_processors(include_callsite: bool) -> list:
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,  # remove if using make_filtering_bound_logger
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if include_callsite:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters={
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                }
            )
        )
    # Must be last so ProcessorFormatter can render stdlib records
    processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)
    return processors


# ----------------------------
# Public API
# ----------------------------
def configure_logging(
    *,
    use_json: bool | None = None,
    level: str | int | None = None,
    colors: bool = True,
    include_callsite: bool = False,
    uvicorn_access_level: str | int | None = None,
    gunicorn_access_level: str | int | None = None,
    structlog_processors: Sequence | None = None,
    cache_logger_on_first_use: bool = True,
    use_filtering_bound_logger: bool = True,
) -> None:
    """
    Configure stdlib logging + structlog so that app, Uvicorn, and Gunicorn logs
    share a single renderer (JSON or console).

    Environment fallbacks:
      LOG_JSON=1|0
      LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
      LOG_UVICORN_ACCESS=INFO|WARNING|ERROR
      LOG_GUNICORN_ACCESS=INFO|WARNING|ERROR
    """
    # Env fallbacks
    if use_json is None:
        use_json = os.getenv("LOG_JSON", "0") == "1"
    lvl = _coerce_level(level or os.getenv("LOG_LEVEL", "INFO"))

    # Tune access noise: default uvicorn.access=WARNING, gunicorn.access=ERROR
    uv_access = _coerce_level(uvicorn_access_level or os.getenv("LOG_UVICORN_ACCESS", "WARNING"))
    gu_access = _coerce_level(gunicorn_access_level or os.getenv("LOG_GUNICORN_ACCESS", "ERROR"))

    # --- stdlib dictConfig
    config = copy.deepcopy(LOGGING_CONFIG)

    # Root + uvicorn core/error levels
    config["handlers"]["default"]["level"] = _level_name(lvl)
    config["loggers"][""]["level"] = _level_name(lvl)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.asgi", "gunicorn", "gunicorn.error"):
        if name in config["loggers"]:
            config["loggers"][name]["level"] = _level_name(lvl)

    # Access loggers set separately
    if "uvicorn.access" in config["loggers"]:
        config["loggers"]["uvicorn.access"]["level"] = _level_name(uv_access)
    if "gunicorn.access" in config["loggers"]:
        config["loggers"]["gunicorn.access"]["level"] = _level_name(gu_access)

    # Renderer selection
    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(colors=colors)
    config["formatters"]["default"]["processor"] = renderer

    logging.config.dictConfig(config)

    # --- structlog.configure
    processors = list(structlog_processors or _base_structlog_processors(include_callsite))
    wrapper_class = structlog.stdlib.BoundLogger
    if use_filtering_bound_logger:
        # Remove filter_by_level (handled by wrapper)
        processors = [p for p in processors if p is not structlog.stdlib.filter_by_level]
        wrapper_class = structlog.make_filtering_bound_logger(lvl)

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=wrapper_class,
        cache_logger_on_first_use=cache_logger_on_first_use,
    )


def bind_request_context(**kwargs) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


# ----------------------------
# Timing helpers
# ----------------------------
@contextmanager
def log_time(event: str, **kwargs):
    log = structlog.get_logger()
    start_ns = time.perf_counter_ns()
    ok = True
    err = None
    try:
        yield
    except Exception as e:  # pragma: no cover
        ok, err = False, repr(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        log.info(event, elapsed_ms=elapsed_ms, ok=ok, error=err, **kwargs)


@asynccontextmanager
async def alog_time(event: str, **kwargs):
    log = structlog.get_logger()
    start_ns = time.perf_counter_ns()
    ok = True
    err = None
    try:
        yield
    except Exception as e:  # pragma: no cover
        ok, err = False, repr(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        log.info(event, elapsed_ms=elapsed_ms, ok=ok, error=err, **kwargs)


# ----------------------------
# Utilities
# ----------------------------
def _coerce_level(lvl: str | int | None) -> int:
    if lvl is None:
        return logging.INFO
    if isinstance(lvl, int):
        return lvl
    if isinstance(lvl, str):
        value = getattr(logging, lvl.upper(), None)
        if isinstance(value, int):
            return value
    return logging.INFO


def _level_name(levelno: int) -> str:
    return logging.getLevelName(levelno)


# ----------------------------
# Optional: per-request context
# ----------------------------
class BindRequestContextMiddleware(BaseHTTPMiddleware):
    """
    Binds a request_id + minimal request info to structlog.contextvars so
    all subsequent logs in the request include that context.
    """

    async def dispatch(self, request, call_next):
        rid = str(uuid4())
        bind_request_context(
            request_id=rid,
            method=request.method,
            path=str(request.url.path),
        )
        try:
            return await call_next(request)
        finally:
            clear_request_context()
