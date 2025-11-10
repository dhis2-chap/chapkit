"""Pydantic schemas for ML train/predict operations."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar

from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, Field
from ulid import ULID

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

    training_artifact_id: ULID = Field(description="ID of the artifact containing the trained model")
    historic: DataFrame = Field(description="Historic data as DataFrame")
    future: DataFrame = Field(description="Future/prediction data as DataFrame")
    geo: FeatureCollection | None = Field(default=None, description="Optional geospatial data")


class PredictResponse(BaseModel):
    """Response schema for predict operation submission."""

    job_id: str = Field(description="ID of the prediction job in the scheduler")
    artifact_id: str = Field(description="ID that will contain the prediction artifact")
    message: str = Field(description="Human-readable message")


class TrainedModelArtifactData(BaseModel):
    """Schema for trained model artifact data stored in the artifact system."""

    ml_type: Literal["ml_training"] = Field(description="Artifact type identifier")
    config_id: str = Field(description="ID of the config used for training")
    started_at: str = Field(description="ISO format timestamp when operation started")
    completed_at: str = Field(description="ISO format timestamp when operation completed")
    duration_seconds: float = Field(description="Operation duration in seconds (rounded to 2 decimals)")
    model: Any = Field(description="The trained model object (must be pickleable)")
    model_type: str | None = Field(default=None, description="Fully qualified class name of the model")
    model_size_bytes: int | None = Field(default=None, description="Serialized pickle size of the model in bytes")

    model_config = {"arbitrary_types_allowed": True}


class PredictionArtifactData(BaseModel):
    """Schema for prediction artifact data stored in the artifact system."""

    ml_type: Literal["ml_prediction"] = Field(description="Artifact type identifier")
    config_id: str = Field(description="ID of the config used for prediction")
    training_artifact_id: str = Field(description="ID of the trained model artifact used for prediction")
    started_at: str = Field(description="ISO format timestamp when operation started")
    completed_at: str = Field(description="ISO format timestamp when operation completed")
    duration_seconds: float = Field(description="Operation duration in seconds (rounded to 2 decimals)")
    predictions: DataFrame = Field(description="Prediction results as structured DataFrame")


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
