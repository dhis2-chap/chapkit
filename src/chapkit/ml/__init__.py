"""ML module for train/predict operations with artifact-based model storage."""

from .manager import MLManager
from .router import MLRouter
from .runner import BaseModelRunner, FunctionalModelRunner, RunInfo, ShellModelRunner
from .schema_discovery import (
    ModelInfo,
    create_config_from_schema,
    discover_model_info,
    discover_model_info_async,
)
from .schemas import (
    MLPredictionArtifactData,
    MLTrainingArtifactData,
    ModelRunnerProtocol,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

__all__ = [
    "BaseModelRunner",
    "FunctionalModelRunner",
    "MLManager",
    "MLRouter",
    "ModelInfo",
    "ModelRunnerProtocol",
    "PredictRequest",
    "PredictResponse",
    "MLPredictionArtifactData",
    "RunInfo",
    "ShellModelRunner",
    "TrainRequest",
    "TrainResponse",
    "MLTrainingArtifactData",
    "create_config_from_schema",
    "discover_model_info",
    "discover_model_info_async",
]
