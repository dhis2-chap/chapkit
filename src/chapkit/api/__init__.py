"""FastAPI routers and related presentation logic."""

from servicekit.api import CrudPermissions, CrudRouter, Router
from servicekit.api.middleware import (
    add_error_handlers,
    add_logging_middleware,
    database_error_handler,
    validation_error_handler,
)
from servicekit.api.routers import HealthRouter, HealthState, HealthStatus, JobRouter, SystemInfo, SystemRouter
from servicekit.api.service_builder import ServiceInfo
from servicekit.api.utilities import build_location_url, run_app
from servicekit.logging import (
    add_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    reset_request_context,
)

from chapkit.artifact import ArtifactRouter
from chapkit.config import ConfigRouter

from .dependencies import get_artifact_manager, get_config_manager
from .service_builder import AssessedStatus, MLServiceBuilder, MLServiceInfo, ServiceBuilder

__all__ = [
    # Base classes
    "Router",
    "CrudRouter",
    "CrudPermissions",
    # Routers
    "HealthRouter",
    "HealthStatus",
    "HealthState",
    "JobRouter",
    "SystemRouter",
    "SystemInfo",
    "ConfigRouter",
    "ArtifactRouter",
    # Dependencies
    "get_config_manager",
    "get_artifact_manager",
    # Middleware
    "add_error_handlers",
    "add_logging_middleware",
    "database_error_handler",
    "validation_error_handler",
    # Logging
    "configure_logging",
    "get_logger",
    "add_request_context",
    "clear_request_context",
    "reset_request_context",
    # Builders
    "ServiceBuilder",
    "MLServiceBuilder",
    "ServiceInfo",
    "MLServiceInfo",
    "AssessedStatus",
    # Utilities
    "build_location_url",
    "run_app",
]
