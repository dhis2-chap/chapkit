"""Chapkit - ML/data service modules built on servicekit."""

from __future__ import annotations

import tomllib
from pathlib import Path

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

# Read version from pyproject.toml
_pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
if _pyproject_path.exists():
    with open(_pyproject_path, "rb") as f:
        _pyproject = tomllib.load(f)
        __version__ = _pyproject["project"]["version"]
else:
    __version__ = "unknown"

__all__ = [
    # Version
    "__version__",
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
