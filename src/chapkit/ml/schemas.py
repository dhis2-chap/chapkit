"""Pydantic schemas for ML train/predict operations.

Migration Note:
    TrainedModelArtifactData and PredictionArtifactData have been replaced by
    MLTrainingArtifactData and MLPredictionArtifactData from chapkit.artifact.data_schemas.

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

from typing import Any, Protocol, TypeVar

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, Field
from ulid import ULID

from chapkit.artifact.schemas import (
    MLPredictionArtifactData,
    MLTrainingArtifactData,
)
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

ConfigT = TypeVar("ConfigT", bound=BaseConfig, contravariant=True)


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

    artifact_id: ULID = Field(description="ID of the artifact containing the trained model")
    historic: DataFrame = Field(description="Historic data as DataFrame")
    future: DataFrame = Field(description="Future/prediction data as DataFrame")
    geo: FeatureCollection | None = Field(default=None, description="Optional geospatial data")


class PredictResponse(BaseModel):
    """Response schema for predict operation submission."""

    job_id: str = Field(description="ID of the prediction job in the scheduler")
    artifact_id: str = Field(description="ID that will contain the prediction artifact")
    message: str = Field(description="Human-readable message")


class ModelRunnerProtocol(Protocol[ConfigT]):
    """Protocol defining the interface for model runners."""

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object (must be pickleable)."""
        ...

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions using a trained model and return predictions as DataFrame."""
        ...


__all__ = [
    "TrainRequest",
    "TrainResponse",
    "PredictRequest",
    "PredictResponse",
    "ModelRunnerProtocol",
    "MLTrainingArtifactData",
    "MLPredictionArtifactData",
]
