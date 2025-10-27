"""Manager for ML train/predict operations with artifact-based storage."""

from __future__ import annotations

import datetime
import pickle
from typing import Generic, TypeVar

from servicekit import Database
from ulid import ULID

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.config import ConfigManager, ConfigRepository
from chapkit.config.schemas import BaseConfig
from chapkit.scheduler import ChapkitJobScheduler

from .schemas import (
    ModelRunnerProtocol,
    PredictionArtifactData,
    PredictRequest,
    PredictResponse,
    TrainedModelArtifactData,
    TrainRequest,
    TrainResponse,
)

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


def _extract_model_type(model: object) -> str | None:
    """Extract fully qualified type name from model object."""
    try:
        # Handle dict models (e.g., ml_class.py pattern with {"model": ..., "scaler": ...})
        if isinstance(model, dict) and "model" in model:
            obj = model["model"]
        else:
            obj = model

        # Get fully qualified class name
        module = type(obj).__module__
        qualname = type(obj).__qualname__
        return f"{module}.{qualname}"
    except Exception:
        return None


def _calculate_model_size(model: object) -> int | None:
    """Calculate serialized pickle size of model in bytes."""
    try:
        pickled = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        return len(pickled)
    except Exception:
        return None


class MLManager(Generic[ConfigT]):
    """Manager for ML train/predict operations with job scheduling and artifact storage."""

    def __init__(
        self,
        runner: ModelRunnerProtocol[ConfigT],
        scheduler: ChapkitJobScheduler,
        database: Database,
        config_schema: type[ConfigT],
    ) -> None:
        """Initialize ML manager with runner, scheduler, database, and config schema."""
        self.runner = runner
        self.scheduler = scheduler
        self.database = database
        self.config_schema = config_schema

    async def execute_train(self, request: TrainRequest) -> TrainResponse:
        """Submit a training job to the scheduler and return job/artifact IDs."""
        # Pre-allocate artifact ID for the trained model
        model_artifact_id = ULID()

        # Submit job to scheduler
        job_id = await self.scheduler.add_job(
            self._train_task,
            request,
            model_artifact_id,
        )

        return TrainResponse(
            job_id=str(job_id),
            model_artifact_id=str(model_artifact_id),
            message=f"Training job submitted. Job ID: {job_id}",
        )

    async def execute_predict(self, request: PredictRequest) -> PredictResponse:
        """Submit a prediction job to the scheduler and return job/artifact IDs."""
        # Pre-allocate artifact ID for predictions
        prediction_artifact_id = ULID()

        # Submit job to scheduler
        job_id = await self.scheduler.add_job(
            self._predict_task,
            request,
            prediction_artifact_id,
        )

        return PredictResponse(
            job_id=str(job_id),
            prediction_artifact_id=str(prediction_artifact_id),
            message=f"Prediction job submitted. Job ID: {job_id}",
        )

    async def _train_task(self, request: TrainRequest, model_artifact_id: ULID) -> ULID:
        """Execute training task and store trained model in artifact."""
        # Load config
        async with self.database.session() as session:
            config_repo = ConfigRepository(session)
            config_manager: ConfigManager[ConfigT] = ConfigManager(config_repo, self.config_schema)
            config = await config_manager.find_by_id(request.config_id)

            if config is None:
                raise ValueError(f"Config {request.config_id} not found")

        # Train model with timing
        training_started_at = datetime.datetime.now(datetime.UTC)
        trained_model = await self.runner.on_train(
            config=config.data,
            data=request.data,
            geo=request.geo,
        )
        training_completed_at = datetime.datetime.now(datetime.UTC)
        training_duration = (training_completed_at - training_started_at).total_seconds()

        # Calculate model metrics
        model_type = _extract_model_type(trained_model)
        model_size_bytes = _calculate_model_size(trained_model)

        # Store trained model in artifact with metadata
        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_manager = ArtifactManager(artifact_repo)
            config_repo = ConfigRepository(session)

            # Create and validate artifact data with Pydantic
            artifact_data_model = TrainedModelArtifactData(
                ml_type="trained_model",
                config_id=str(request.config_id),
                model=trained_model,
                started_at=training_started_at.isoformat(),
                completed_at=training_completed_at.isoformat(),
                duration_seconds=round(training_duration, 2),
                model_type=model_type,
                model_size_bytes=model_size_bytes,
            )

            await artifact_manager.save(
                ArtifactIn(
                    id=model_artifact_id,
                    data=artifact_data_model.model_dump(),
                    parent_id=None,
                    level=0,
                )
            )

            # Link config to root artifact for tree traversal
            await config_repo.link_artifact(request.config_id, model_artifact_id)
            await config_repo.commit()

        return model_artifact_id

    async def _predict_task(self, request: PredictRequest, prediction_artifact_id: ULID) -> ULID:
        """Execute prediction task and store predictions in artifact."""
        # Load model artifact
        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_manager = ArtifactManager(artifact_repo)
            model_artifact = await artifact_manager.find_by_id(request.model_artifact_id)

            if model_artifact is None:
                raise ValueError(f"Model artifact {request.model_artifact_id} not found")

        # Extract model and config_id from artifact
        model_data = model_artifact.data
        if not isinstance(model_data, dict) or model_data.get("ml_type") != "trained_model":
            raise ValueError(f"Artifact {request.model_artifact_id} is not a trained model")

        trained_model = model_data["model"]
        config_id = ULID.from_str(model_data["config_id"])

        # Load config
        async with self.database.session() as session:
            config_repo = ConfigRepository(session)
            config_manager: ConfigManager[ConfigT] = ConfigManager(config_repo, self.config_schema)
            config = await config_manager.find_by_id(config_id)

            if config is None:
                raise ValueError(f"Config {config_id} not found")

        # Make predictions with timing
        prediction_started_at = datetime.datetime.now(datetime.UTC)
        predictions = await self.runner.on_predict(
            config=config.data,
            model=trained_model,
            historic=request.historic,
            future=request.future,
            geo=request.geo,
        )
        prediction_completed_at = datetime.datetime.now(datetime.UTC)
        prediction_duration = (prediction_completed_at - prediction_started_at).total_seconds()

        # Store predictions in artifact with parent linkage
        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_manager = ArtifactManager(artifact_repo)

            # Create and validate artifact data with Pydantic
            artifact_data_model = PredictionArtifactData(
                ml_type="prediction",
                model_artifact_id=str(request.model_artifact_id),
                config_id=str(config_id),
                predictions=predictions,
                started_at=prediction_started_at.isoformat(),
                completed_at=prediction_completed_at.isoformat(),
                duration_seconds=round(prediction_duration, 2),
            )

            await artifact_manager.save(
                ArtifactIn(
                    id=prediction_artifact_id,
                    data=artifact_data_model.model_dump(),
                    parent_id=request.model_artifact_id,
                    level=1,
                )
            )

        return prediction_artifact_id
