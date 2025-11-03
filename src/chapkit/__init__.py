"""Chapkit - ML/data service modules built on servicekit."""

from __future__ import annotations

from pathlib import Path

# Read version from package metadata - must be before internal imports
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("chapkit")
except Exception:
    __version__ = "unknown"

# CLI feature
# Scheduler feature
# Artifact feature
from .artifact import (
    Artifact,
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactOut,
    ArtifactRepository,
    ArtifactRouter,
    ArtifactTreeNode,
)
from .cli import app as cli_app

# Config feature
from .config import (
    BaseConfig,
    Config,
    ConfigIn,
    ConfigManager,
    ConfigOut,
    ConfigRepository,
)

# Data feature
from .data import DataFrame, GroupBy

# ML feature
from .ml import (
    FunctionalModelRunner,
    MLManager,
    MLRouter,
    ModelRunnerProtocol,
    PredictionArtifactData,
    PredictRequest,
    PredictResponse,
    TrainedModelArtifactData,
    TrainRequest,
    TrainResponse,
)
from .scheduler import ChapkitJobRecord, ChapkitJobScheduler

# Task feature
from .task import (
    Task,
    TaskIn,
    TaskManager,
    TaskOut,
    TaskRegistry,
    TaskRepository,
    TaskRouter,
    validate_and_disable_orphaned_tasks,
)


def get_alembic_dir() -> Path:
    """Get the path to chapkit's bundled alembic migrations directory."""
    return Path(__file__).parent / "alembic"


__all__ = [
    # Version
    "__version__",
    # Utils
    "get_alembic_dir",
    # CLI
    "cli_app",
    # Scheduler
    "ChapkitJobRecord",
    "ChapkitJobScheduler",
    # Artifact
    "Artifact",
    "ArtifactHierarchy",
    "ArtifactIn",
    "ArtifactManager",
    "ArtifactOut",
    "ArtifactRepository",
    "ArtifactRouter",
    "ArtifactTreeNode",
    # Config
    "BaseConfig",
    "Config",
    "ConfigIn",
    "ConfigManager",
    "ConfigOut",
    "ConfigRepository",
    # Data
    "DataFrame",
    "GroupBy",
    # ML
    "FunctionalModelRunner",
    "MLManager",
    "MLRouter",
    "ModelRunnerProtocol",
    "PredictionArtifactData",
    "PredictRequest",
    "PredictResponse",
    "TrainedModelArtifactData",
    "TrainRequest",
    "TrainResponse",
    # Task
    "Task",
    "TaskIn",
    "TaskManager",
    "TaskOut",
    "TaskRegistry",
    "TaskRepository",
    "TaskRouter",
    "validate_and_disable_orphaned_tasks",
]
