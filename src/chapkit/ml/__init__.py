"""ML module for train/predict operations with artifact-based model storage."""

from .manager import MLManager
from .router import MLRouter
from .runner import BaseModelRunner, FunctionalModelRunner, ShellModelRunner
from .schemas import (
    MLPredictionArtifactData,
    MLTrainingWorkspaceArtifactData,
    ModelRunnerProtocol,
    PredictRequest,
    PredictResponse,
    Severity,
    TrainRequest,
    TrainResponse,
    ValidatePredictRequest,
    ValidateRequest,
    ValidateTrainRequest,
    ValidationDiagnostic,
    ValidationResponse,
)

__all__ = [
    "BaseModelRunner",
    "FunctionalModelRunner",
    "MLManager",
    "MLRouter",
    "ModelRunnerProtocol",
    "PredictRequest",
    "PredictResponse",
    "MLPredictionArtifactData",
    "ShellModelRunner",
    "TrainRequest",
    "TrainResponse",
    "MLTrainingWorkspaceArtifactData",
    "ValidatePredictRequest",
    "ValidateRequest",
    "ValidateTrainRequest",
    "ValidationDiagnostic",
    "ValidationResponse",
    "Severity",
]
