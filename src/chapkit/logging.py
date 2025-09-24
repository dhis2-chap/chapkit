from starlette.middleware.base import BaseHTTPMiddleware
from uuid import uuid4
import copy
import logging
import logging.config
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Sequence

import structlog

# Processors for logs emitted by standard library loggers (e.g., uvicorn),
# before they are handed to structlog's ProcessorFormatter.
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
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            # The ProcessorFormatter turns stdlib LogRecord into an event dict
            # and then renders it with the configured "processor" below.
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
        # Root logger
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        },
        # Uvicorn core logs (errors)
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        # Uvicorn access logs (can be noisy; tune separately)
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# =========================
# Structlog processors
# =========================


def _base_structlog_processors(include_callsite: bool) -> list:
    """
    Build the processor pipeline for *structlog* loggers (not stdlib).
    `wrap_for_formatter` must remain last in the chain.
    """
    processors: list = [
        # Merge any per-request or per-task context bound via structlog.contextvars
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,  # may be removed if using make_filtering_bound_logger
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

    # This must be the last processor so stdlib's ProcessorFormatter can render.
    processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)
    return processors


# =========================
# Public API
# =========================


def configure_logging(
    *,
    use_json: bool = False,
    level: str | int = "INFO",
    colors: bool = True,
    include_callsite: bool = False,
    uvicorn_access_level: str | int | None = None,
    structlog_processors: Sequence | None = None,
    cache_logger_on_first_use: bool = True,
    use_filtering_bound_logger: bool = False,
) -> None:
    """
    Configure stdlib logging + structlog.

    Args:
        use_json: True => JSONRenderer, False => ConsoleRenderer.
        level: Root/uvicorn.error level (str or int, e.g., "INFO", logging.INFO).
        colors: Enable ANSI colors for ConsoleRenderer (ignored when use_json=True).
        include_callsite: Include filename:line and func in logs (handy in dev).
        uvicorn_access_level: Separate level for uvicorn.access (e.g., "WARNING" in prod).
        structlog_processors: Optional override of the structlog processor pipeline.
        cache_logger_on_first_use: Cache structlog loggers after first creation.
        use_filtering_bound_logger: If True, use make_filtering_bound_logger for perf and
            remove filter_by_level from processors.
    """
    lvl = _coerce_level(level)

    # --- stdlib dictConfig setup
    config = copy.deepcopy(LOGGING_CONFIG)

    # Root + uvicorn.error
    config["handlers"]["default"]["level"] = _level_name(lvl)
    config["loggers"][""]["level"] = _level_name(lvl)
    if "uvicorn.error" in config["loggers"]:
        config["loggers"]["uvicorn.error"]["level"] = _level_name(lvl)

    # uvicorn.access can be tuned separately
    access_level = _coerce_level(uvicorn_access_level) if uvicorn_access_level is not None else lvl

    if "uvicorn.access" in config["loggers"]:
        config["loggers"]["uvicorn.access"]["level"] = _level_name(access_level)

    # Renderer selection
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=colors)

    config["formatters"]["default"]["processor"] = renderer

    logging.config.dictConfig(config)

    # --- structlog.configure
    processors = list(structlog_processors or _base_structlog_processors(include_callsite))

    wrapper_class = structlog.stdlib.BoundLogger  # default
    if use_filtering_bound_logger:
        # Remove filter_by_level from the pipeline (it's handled by the wrapper)
        processors = [p for p in processors if p is not structlog.stdlib.filter_by_level]
        wrapper_class = structlog.make_filtering_bound_logger(lvl)

    structlog.configure(
        processors=processors,
        context_class=dict,  # use dict; can be swapped for OrderedDict if you care about key order in text logs
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=wrapper_class,
        cache_logger_on_first_use=cache_logger_on_first_use,
    )


def bind_request_context(**kwargs) -> None:
    """
    Bind contextual values for the current task/request. They will automatically
    appear in subsequent log records (due to merge_contextvars).
    Example:
        bind_request_context(request_id="...", method="GET", path="/api")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """
    Clear previously bound context variables for the current task/request.
    Call this at the end of request handling or background task.
    """
    structlog.contextvars.clear_contextvars()


# =========================
# Timing helpers
# =========================


@contextmanager
def log_time(event: str, **kwargs):
    """
    Measure and log the duration of a code block.
    Emits: {event, elapsed_ms, ok, error?}
    """
    log = structlog.get_logger()
    start_ns = time.perf_counter_ns()
    ok = True
    err_repr = None
    try:
        yield
    except Exception as e:  # re-log outcome, then re-raise
        ok = False
        err_repr = repr(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        log.info(event, elapsed_ms=elapsed_ms, ok=ok, error=err_repr, **kwargs)


@asynccontextmanager
async def alog_time(event: str, **kwargs):
    """
    Async variant of log_time for use in async code.
    """
    log = structlog.get_logger()
    start_ns = time.perf_counter_ns()
    ok = True
    err_repr = None
    try:
        yield
    except Exception as e:
        ok = False
        err_repr = repr(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        log.info(event, elapsed_ms=elapsed_ms, ok=ok, error=err_repr, **kwargs)


# =========================
# Utilities
# =========================


def _coerce_level(lvl: str | int | None) -> int:
    """
    Turn a "level" value into a logging levelno.
    Accepts: None, int (e.g., 20), or str (e.g., "INFO").
    """
    if lvl is None:
        return logging.INFO
    if isinstance(lvl, int):
        return lvl
    if isinstance(lvl, str):
        value = getattr(logging, lvl.upper(), None)
        if isinstance(value, int):
            return value
    # Fallback
    return logging.INFO


def _level_name(levelno: int) -> str:
    """Return the canonical level name for dictConfig (e.g., 20 -> 'INFO')."""
    return logging.getLevelName(levelno)


class BindRequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a request_id and binds minimal request context.
    Add to FastAPI with: app.add_middleware(BindRequestContextMiddleware)
    """

    async def dispatch(self, request, call_next):
        rid = str(uuid4())
        bind_request_context(
            request_id=rid,
            method=request.method,
            path=str(request.url.path),
        )
        try:
            response = await call_next(request)
            return response
        finally:
            clear_request_context()
