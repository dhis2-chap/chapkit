"""Pydantic schemas for ML train/predict operations.

Migration Note:
    TrainedModelArtifactData and PredictionArtifactData have been replaced by
    MLTrainingWorkspaceArtifactData and MLPredictionArtifactData from chapkit.artifact.schemas.

    Key changes:
    - ml_type field renamed to type
    - model field moved to content
    - predictions field moved to content
    - Added nested metadata structure
    - Added content_type and content_size fields
    - Removed training_artifact_id (use parent_id instead)
    - Removed model_type and model_size_bytes (metadata only)
"""

from __future__ import annotations

import datetime
from typing import Annotated, Any, Literal, Protocol, Self, TypeVar

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, Field, model_validator
from ulid import ULID

from chapkit.artifact.schemas import (
    MLPredictionArtifactData,
    MLTrainingWorkspaceArtifactData,
)
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

ConfigT = TypeVar("ConfigT", bound=BaseConfig, contravariant=True)

Severity = Literal["error", "warning", "info"]


class TrainRequest(BaseModel):
    """Request schema for training a model."""

    config_id: ULID = Field(description="ID of the config to use for training")
    data: DataFrame = Field(description="Training data as DataFrame")
    geo: FeatureCollection | None = Field(default=None, description="Optional geospatial data")


class TrainResponse(BaseModel):
    """Response schema for train operation submission."""

    job_id: str = Field(description="ID of the training job in the scheduler")
    artifact_id: str = Field(description="ID that will contain the trained model artifact")
    message: str = Field(description="Human-readable message")


class PredictRequest(BaseModel):
    """Request schema for making predictions."""

    artifact_id: ULID | None = Field(
        default=None,
        description="ID of the trained artifact (train-backed services)",
    )
    config_id: ULID | None = Field(
        default=None,
        description="Config ID to predict against directly (stateless services)",
    )
    historic: DataFrame = Field(description="Historic data as DataFrame")
    future: DataFrame = Field(description="Future/prediction data as DataFrame")
    geo: FeatureCollection | None = Field(default=None, description="Optional geospatial data")

    @model_validator(mode="after")
    def _exactly_one_id(self) -> Self:
        """Require exactly one of artifact_id / config_id to be set."""
        if (self.artifact_id is None) == (self.config_id is None):
            raise ValueError("Exactly one of 'artifact_id' (train-backed) or 'config_id' (stateless) must be provided")
        return self


class PredictResponse(BaseModel):
    """Response schema for predict operation submission."""

    job_id: str = Field(description="ID of the prediction job in the scheduler")
    artifact_id: str = Field(description="ID that will contain the prediction artifact")
    message: str = Field(description="Human-readable message")


class ValidationDiagnostic(BaseModel):
    """A single validation finding surfaced by $validate."""

    severity: Severity = Field(description="error | warning | info")
    code: str = Field(
        description="Stable machine-readable code, e.g. 'n_lags_exceeds_context'",
        max_length=128,
    )
    message: str = Field(description="Human-readable message for display", max_length=2048)
    field: str | None = Field(
        default=None,
        description="Optional dotted path to the offending input, e.g. 'config.n_lags' or 'historic'",
        max_length=256,
    )

    @classmethod
    def error(cls, *, code: str, message: str, field: str | None = None) -> ValidationDiagnostic:
        """Create an error-severity diagnostic."""
        return cls(severity="error", code=code, message=message, field=field)

    @classmethod
    def warning(cls, *, code: str, message: str, field: str | None = None) -> ValidationDiagnostic:
        """Create a warning-severity diagnostic."""
        return cls(severity="warning", code=code, message=message, field=field)

    @classmethod
    def info(cls, *, code: str, message: str, field: str | None = None) -> ValidationDiagnostic:
        """Create an info-severity diagnostic."""
        return cls(severity="info", code=code, message=message, field=field)


class ValidationResponse(BaseModel):
    """Result of a $validate call."""

    valid: bool = Field(description="True iff no diagnostic has severity='error'")
    diagnostics: list[ValidationDiagnostic] = Field(default_factory=list)


class ValidateTrainRequest(BaseModel):
    """Request schema for validating a train payload."""

    type: Literal["train"] = Field(default="train", frozen=True)
    config_id: ULID = Field(description="ID of the config to use for training")
    data: DataFrame = Field(description="Training data as DataFrame")
    geo: FeatureCollection | None = Field(default=None, description="Optional geospatial data")


class ValidatePredictRequest(BaseModel):
    """Request schema for validating a predict payload."""

    type: Literal["predict"] = Field(default="predict", frozen=True)
    artifact_id: ULID | None = Field(
        default=None,
        description="ID of the trained artifact (train-backed services)",
    )
    config_id: ULID | None = Field(
        default=None,
        description="Config ID to validate against directly (stateless services)",
    )
    historic: DataFrame = Field(description="Historic data as DataFrame")
    future: DataFrame = Field(description="Future/prediction data as DataFrame")
    geo: FeatureCollection | None = Field(default=None, description="Optional geospatial data")

    @model_validator(mode="after")
    def _exactly_one_id(self) -> Self:
        """Require exactly one of artifact_id / config_id to be set."""
        if (self.artifact_id is None) == (self.config_id is None):
            raise ValueError("Exactly one of 'artifact_id' (train-backed) or 'config_id' (stateless) must be provided")
        return self


ValidateRequest = Annotated[
    ValidateTrainRequest | ValidatePredictRequest,
    Field(discriminator="type"),
]


class ModelRunnerProtocol(Protocol[ConfigT]):
    """Protocol defining the interface for model runners."""

    @property
    def predict_only(self) -> bool:
        """Return True when this runner has no train step (stateless predict)."""
        ...

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object (must be pickleable)."""
        ...

    async def create_training_artifact(
        self,
        training_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact data structure from training result."""
        ...

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Make predictions using a trained model and return predictions."""
        ...

    async def create_prediction_artifact(
        self,
        prediction_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact data structure from prediction result."""
        ...

    async def on_validate_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Return domain-specific diagnostics for a train payload (default: empty)."""
        ...

    async def on_validate_predict(
        self,
        config: ConfigT,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Return domain-specific diagnostics for a predict payload (default: empty)."""
        ...


__all__ = [
    "TrainRequest",
    "TrainResponse",
    "PredictRequest",
    "PredictResponse",
    "ValidationDiagnostic",
    "ValidationResponse",
    "ValidateTrainRequest",
    "ValidatePredictRequest",
    "ValidateRequest",
    "Severity",
    "ModelRunnerProtocol",
    "MLTrainingWorkspaceArtifactData",
    "MLPredictionArtifactData",
]
