"""Manager for ML train/predict operations with artifact-based storage."""

from __future__ import annotations

import datetime
from typing import Generic, TypeVar

from servicekit import Database
from ulid import ULID

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.config import ConfigManager, ConfigRepository
from chapkit.config.schemas import BaseConfig
from chapkit.scheduler import ChapkitScheduler

from .schemas import (
    ModelRunnerProtocol,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class MLManager(Generic[ConfigT]):
    """Manager for ML train/predict operations with job scheduling and artifact storage."""

    def __init__(
        self,
        runner: ModelRunnerProtocol[ConfigT],
        scheduler: ChapkitScheduler,
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
        artifact_id = ULID()

        # Submit job to scheduler
        job_id = await self.scheduler.add_job(
            self._train_task,
            request,
            artifact_id,
        )

        return TrainResponse(
            job_id=str(job_id),
            artifact_id=str(artifact_id),
            message=f"Training job submitted. Job ID: {job_id}",
        )

    async def execute_predict(self, request: PredictRequest) -> PredictResponse:
        """Submit a prediction job to the scheduler and return job/artifact IDs."""
        # Pre-allocate artifact ID for predictions
        artifact_id = ULID()

        # Submit job to scheduler
        job_id = await self.scheduler.add_job(
            self._predict_task,
            request,
            artifact_id,
        )

        return PredictResponse(
            job_id=str(job_id),
            artifact_id=str(artifact_id),
            message=f"Prediction job submitted. Job ID: {job_id}",
        )

    async def _train_task(self, request: TrainRequest, artifact_id: ULID) -> ULID:
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

        # Store trained model in artifact with metadata
        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_manager = ArtifactManager(artifact_repo)
            config_repo = ConfigRepository(session)

            # Create metadata
            from chapkit.artifact.schemas import MLMetadata, MLTrainingArtifactData

            metadata = MLMetadata(
                status="success",
                config_id=str(request.config_id),
                started_at=training_started_at.isoformat(),
                completed_at=training_completed_at.isoformat(),
                duration_seconds=round(training_duration, 2),
            )

            # Create and validate artifact data structure with Pydantic
            # Note: We validate but don't serialize to JSON because content contains Python objects
            artifact_data_model = MLTrainingArtifactData(
                type="ml_training",
                metadata=metadata,
                content=trained_model,
                content_type="application/x-pickle",
                content_size=None,
            )

            # Construct dict manually to preserve Python objects (database uses PickleType)
            artifact_data = {
                "type": artifact_data_model.type,
                "metadata": artifact_data_model.metadata.model_dump(),
                "content": trained_model,  # Keep as Python object
                "content_type": artifact_data_model.content_type,
                "content_size": artifact_data_model.content_size,
            }

            await artifact_manager.save(
                ArtifactIn(
                    id=artifact_id,
                    data=artifact_data,
                    parent_id=None,
                    level=0,
                )
            )

            # Link config to root artifact for tree traversal
            await config_repo.link_artifact(request.config_id, artifact_id)
            await config_repo.commit()

        return artifact_id

    async def _predict_task(self, request: PredictRequest, artifact_id: ULID) -> ULID:
        """Execute prediction task and store predictions in artifact."""
        # Load training artifact
        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_manager = ArtifactManager(artifact_repo)
            training_artifact = await artifact_manager.find_by_id(request.artifact_id)

            if training_artifact is None:
                raise ValueError(f"Training artifact {request.artifact_id} not found")

        # Extract model and config_id from artifact
        training_data = training_artifact.data
        if not isinstance(training_data, dict) or training_data.get("type") != "ml_training":
            raise ValueError(f"Artifact {request.artifact_id} is not a training artifact")

        trained_model = training_data["content"]
        config_id = ULID.from_str(training_data["metadata"]["config_id"])

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

            # Create metadata
            from chapkit.artifact.schemas import MLMetadata, MLPredictionArtifactData

            metadata = MLMetadata(
                status="success",
                config_id=str(config_id),
                started_at=prediction_started_at.isoformat(),
                completed_at=prediction_completed_at.isoformat(),
                duration_seconds=round(prediction_duration, 2),
            )

            # Create and validate artifact data structure with Pydantic
            # Note: We validate but don't serialize to JSON because content contains Python objects
            artifact_data_model = MLPredictionArtifactData(
                type="ml_prediction",
                metadata=metadata,
                content=predictions,
                content_type="application/vnd.chapkit.dataframe+json",
                content_size=None,
            )

            # Construct dict manually to preserve Python objects (database uses PickleType)
            artifact_data = {
                "type": artifact_data_model.type,
                "metadata": artifact_data_model.metadata.model_dump(),
                "content": predictions,  # Keep as Python object (DataFrame)
                "content_type": artifact_data_model.content_type,
                "content_size": artifact_data_model.content_size,
            }

            await artifact_manager.save(
                ArtifactIn(
                    id=artifact_id,
                    data=artifact_data,
                    parent_id=request.artifact_id,
                    level=1,
                )
            )

        return artifact_id
