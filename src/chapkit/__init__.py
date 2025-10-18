"""Chapkit - ML/data service modules built on servicekit."""

# Re-export servicekit core for convenience
from servicekit import (
    Base,
    BaseManager,
    BaseRepository,
    Database,
    Entity,
    EntityIn,
    EntityOut,
    Manager,
    Repository,
    SqliteDatabase,
    SqliteDatabaseBuilder,
    ULIDType,
)

# Artifact feature
from .artifact import (
    Artifact,
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactOut,
    ArtifactRepository,
    ArtifactTreeNode,
    PandasDataFrame,
)

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

# Task feature
from .task import (
    Task,
    TaskIn,
    TaskManager,
    TaskOut,
    TaskRegistry,
    TaskRepository,
    validate_and_disable_orphaned_tasks,
)

__all__ = [
    # Re-exported from servicekit
    "Base",
    "BaseManager",
    "BaseRepository",
    "Database",
    "Entity",
    "EntityIn",
    "EntityOut",
    "Manager",
    "Repository",
    "SqliteDatabase",
    "SqliteDatabaseBuilder",
    "ULIDType",
    # Artifact
    "Artifact",
    "ArtifactHierarchy",
    "ArtifactIn",
    "ArtifactManager",
    "ArtifactOut",
    "ArtifactRepository",
    "ArtifactTreeNode",
    "PandasDataFrame",
    # Config
    "BaseConfig",
    "Config",
    "ConfigIn",
    "ConfigManager",
    "ConfigOut",
    "ConfigRepository",
    # Task
    "Task",
    "TaskIn",
    "TaskManager",
    "TaskOut",
    "TaskRegistry",
    "TaskRepository",
    "validate_and_disable_orphaned_tasks",
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
]
