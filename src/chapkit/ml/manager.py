"""Manager for ML train/predict operations with artifact-based storage."""

from __future__ import annotations

import datetime
import pickle
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Generic, TypeVar

from servicekit import Database
from ulid import ULID

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.config import ConfigManager, ConfigRepository
from chapkit.config.schemas import BaseConfig
from chapkit.scheduler import ChapkitScheduler

from .runner import get_temp_dir
from .schemas import (
    ModelRunnerProtocol,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
    ValidatePredictRequest,
    ValidateRequest,
    ValidateTrainRequest,
    ValidationDiagnostic,
    ValidationResponse,
)

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


def _has_error(diagnostics: list[ValidationDiagnostic]) -> bool:
    """Return True if any diagnostic has severity='error'."""
    return any(d.severity == "error" for d in diagnostics)


class MLManager(Generic[ConfigT]):
    """Manager for ML train/predict operations with job scheduling and artifact storage."""

    def __init__(
        self,
        runner: ModelRunnerProtocol[ConfigT],
        scheduler: ChapkitScheduler,
        database: Database,
        config_schema: type[ConfigT],
        min_prediction_periods: int = 0,
        max_prediction_periods: int = 100,
    ) -> None:
        """Initialize ML manager with runner, scheduler, database, and config schema."""
        self.runner = runner
        self.scheduler = scheduler
        self.database = database
        self.config_schema = config_schema
        self.min_prediction_periods = min_prediction_periods
        self.max_prediction_periods = max_prediction_periods

    def _validate_prediction_periods(self, config_data: BaseConfig) -> None:
        """Validate that config prediction_periods is within allowed bounds."""
        periods = config_data.prediction_periods
        if periods < self.min_prediction_periods:
            raise ValueError(
                f"prediction_periods ({periods}) is below the minimum allowed value ({self.min_prediction_periods})"
            )
        if periods > self.max_prediction_periods:
            raise ValueError(
                f"prediction_periods ({periods}) exceeds the maximum allowed value ({self.max_prediction_periods})"
            )

    def _check_prediction_periods(self, config_data: BaseConfig) -> list[ValidationDiagnostic]:
        """Non-raising variant of prediction-periods bounds check returning diagnostics."""
        diagnostics: list[ValidationDiagnostic] = []
        periods = config_data.prediction_periods
        if periods < self.min_prediction_periods:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="prediction_periods_out_of_bounds",
                    message=(
                        f"prediction_periods ({periods}) is below the minimum allowed value "
                        f"({self.min_prediction_periods})"
                    ),
                    field="config.prediction_periods",
                )
            )
        elif periods > self.max_prediction_periods:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="prediction_periods_out_of_bounds",
                    message=(
                        f"prediction_periods ({periods}) exceeds the maximum allowed value "
                        f"({self.max_prediction_periods})"
                    ),
                    field="config.prediction_periods",
                )
            )
        return diagnostics

    async def validate(self, request: ValidateRequest) -> ValidationResponse:
        """Run framework-level and runner validations for a train or predict payload."""
        if request.type == "train":
            diagnostics = await self._validate_train(request)
        else:
            diagnostics = await self._validate_predict(request)
        return ValidationResponse(valid=not _has_error(diagnostics), diagnostics=diagnostics)

    async def _validate_train(self, request: ValidateTrainRequest) -> list[ValidationDiagnostic]:
        """Collect diagnostics for a train payload."""
        diagnostics: list[ValidationDiagnostic] = []

        async with self.database.session() as session:
            config_repo = ConfigRepository(session)
            config_manager: ConfigManager[ConfigT] = ConfigManager(config_repo, self.config_schema)
            config = await config_manager.find_by_id(request.config_id)

        if config is None:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="config_not_found",
                    message=f"Config {request.config_id} not found",
                    field="config_id",
                )
            )
            return diagnostics

        diagnostics.extend(self._check_prediction_periods(config.data))

        if len(request.data) == 0:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="data_empty",
                    message="Training data is empty",
                    field="data",
                )
            )

        if _has_error(diagnostics):
            return diagnostics

        runner_diagnostics = await self.runner.on_validate_train(
            config=config.data,
            data=request.data,
            geo=request.geo,
        )
        diagnostics.extend(runner_diagnostics)
        return diagnostics

    async def _validate_predict(self, request: ValidatePredictRequest) -> list[ValidationDiagnostic]:
        """Collect diagnostics for a predict payload."""
        diagnostics: list[ValidationDiagnostic] = []

        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_manager = ArtifactManager(artifact_repo)
            training_artifact = await artifact_manager.find_by_id(request.artifact_id)

        if training_artifact is None:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="training_artifact_not_found",
                    message=f"Training artifact {request.artifact_id} not found",
                    field="artifact_id",
                )
            )
            return diagnostics

        training_data = training_artifact.data
        if not isinstance(training_data, dict) or training_data.get("type") != "ml_training_workspace":
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="invalid_training_artifact",
                    message=f"Artifact {request.artifact_id} is not a training artifact",
                    field="artifact_id",
                )
            )
            return diagnostics

        training_metadata = training_data.get("metadata", {})
        training_status = training_metadata.get("status", "unknown")
        if training_status == "failed":
            exit_code = training_metadata.get("exit_code", "unknown")
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="training_artifact_failed",
                    message=(
                        f"Training artifact {request.artifact_id} failed training "
                        f"(exit_code={exit_code}); cannot predict with it"
                    ),
                    field="artifact_id",
                )
            )
            return diagnostics

        config_id_str = training_metadata.get("config_id")
        if not config_id_str:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="invalid_training_artifact",
                    message=f"Training artifact {request.artifact_id} is missing config_id in metadata",
                    field="artifact_id",
                )
            )
            return diagnostics

        try:
            config_id = ULID.from_str(config_id_str)
        except ValueError:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="invalid_training_artifact",
                    message=(
                        f"Training artifact {request.artifact_id} has a malformed config_id "
                        f"in metadata: {config_id_str!r}"
                    ),
                    field="artifact_id",
                )
            )
            return diagnostics

        async with self.database.session() as session:
            config_repo = ConfigRepository(session)
            config_manager = ConfigManager(config_repo, self.config_schema)
            config = await config_manager.find_by_id(config_id)

        if config is None:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="config_not_found",
                    message=f"Config {config_id} referenced by training artifact not found",
                    field="artifact_id",
                )
            )
            return diagnostics

        diagnostics.extend(self._check_prediction_periods(config.data))

        diagnostics.extend(self._check_training_workspace(training_data, request.artifact_id))

        if len(request.historic) == 0:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="historic_empty",
                    message="Historic data is empty",
                    field="historic",
                )
            )
        if len(request.future) == 0:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="future_empty",
                    message="Future data is empty",
                    field="future",
                )
            )

        if _has_error(diagnostics):
            return diagnostics

        runner_diagnostics = await self.runner.on_validate_predict(
            config=config.data,
            historic=request.historic,
            future=request.future,
            geo=request.geo,
        )
        diagnostics.extend(runner_diagnostics)
        return diagnostics

    def _check_training_workspace(
        self,
        training_data: dict[str, Any],
        artifact_id: ULID,
    ) -> list[ValidationDiagnostic]:
        """Mirror the deterministic integrity checks from _predict_task."""
        from .runner import ShellModelRunner

        diagnostics: list[ValidationDiagnostic] = []

        if training_data.get("content_type") != "application/zip":
            return diagnostics

        workspace_content = training_data.get("content")
        if not isinstance(workspace_content, (bytes, bytearray)) or len(workspace_content) == 0:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="training_workspace_corrupted",
                    message=f"Training artifact {artifact_id} has empty or non-binary workspace content",
                    field="artifact_id",
                )
            )
            return diagnostics

        try:
            with zipfile.ZipFile(BytesIO(workspace_content), "r") as zf:
                bad_entry = zf.testzip()
                if bad_entry is not None:
                    diagnostics.append(
                        ValidationDiagnostic(
                            severity="error",
                            code="training_workspace_corrupted",
                            message=f"Corrupt file in training workspace ZIP: {bad_entry}",
                            field="artifact_id",
                        )
                    )
                    return diagnostics
                names = set(zf.namelist())
                pickle_bytes = zf.read("model.pickle") if "model.pickle" in names else None
        except zipfile.BadZipFile as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="training_workspace_corrupted",
                    message=f"Training artifact workspace is not a valid ZIP: {exc}",
                    field="artifact_id",
                )
            )
            return diagnostics

        if isinstance(self.runner, ShellModelRunner):
            return diagnostics

        if pickle_bytes is None:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="model_pickle_missing",
                    message="Training artifact workspace is missing model.pickle",
                    field="artifact_id",
                )
            )
            return diagnostics

        try:
            pickle.loads(pickle_bytes)
        except (pickle.UnpicklingError, EOFError, TypeError, AttributeError, ImportError, ValueError) as exc:
            diagnostics.append(
                ValidationDiagnostic(
                    severity="error",
                    code="model_pickle_corrupted",
                    message=f"Training artifact model.pickle is corrupted or incompatible: {exc}",
                    field="artifact_id",
                )
            )

        return diagnostics

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

            self._validate_prediction_periods(config.data)

        # Train model with timing
        training_started_at = datetime.datetime.now(datetime.UTC)
        training_result = await self.runner.on_train(
            config=config.data,
            data=request.data,
            geo=request.geo,
        )
        training_completed_at = datetime.datetime.now(datetime.UTC)
        training_duration = (training_completed_at - training_started_at).total_seconds()

        # Extract workspace_dir before try block (for cleanup in finally)
        workspace_dir_str = training_result.get("workspace_dir") if isinstance(training_result, dict) else None
        workspace_dir = Path(workspace_dir_str) if workspace_dir_str else None

        try:
            # Let runner create artifact structure
            artifact_data_dict = await self.runner.create_training_artifact(
                training_result=training_result,
                config_id=str(request.config_id),
                started_at=training_started_at,
                completed_at=training_completed_at,
                duration_seconds=round(training_duration, 2),
            )

            # Validate artifact structure with Pydantic
            from chapkit.artifact.schemas import MLTrainingWorkspaceArtifactData

            MLTrainingWorkspaceArtifactData.model_validate(artifact_data_dict)

            # Store artifact
            async with self.database.session() as session:
                artifact_repo = ArtifactRepository(session)
                artifact_manager = ArtifactManager(artifact_repo)
                config_repo = ConfigRepository(session)

                await artifact_manager.save(
                    ArtifactIn(
                        id=artifact_id,
                        data=artifact_data_dict,  # Use dict directly (PickleType)
                        parent_id=None,
                        level=0,
                    )
                )

                # Link config to root artifact for tree traversal
                await config_repo.link_artifact(request.config_id, artifact_id)
                await config_repo.commit()

        finally:
            # Cleanup workspace if created by ShellModelRunner
            if workspace_dir and workspace_dir.exists():
                shutil.rmtree(workspace_dir, ignore_errors=True)

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
        if not isinstance(training_data, dict) or training_data.get("type") != "ml_training_workspace":
            raise ValueError(f"Artifact {request.artifact_id} is not a training artifact")

        # Check training status - block prediction on failed training
        training_metadata = training_data.get("metadata", {})
        training_status = training_metadata.get("status", "unknown")

        if training_status == "failed":
            exit_code = training_metadata.get("exit_code", "unknown")
            raise ValueError(
                f"Cannot predict using failed training artifact {request.artifact_id}. "
                f"Training script exited with code {exit_code}."
            )

        # Check if artifact is workspace (ZIP) or pickled model
        is_workspace = training_data.get("content_type") == "application/zip"
        extracted_workspace = None
        prediction_workspace_dir = None

        try:
            if is_workspace:
                # Extract workspace from zip
                import pickle
                import tempfile
                import zipfile
                from io import BytesIO

                workspace_content = training_data["content"]
                extracted_workspace = Path(tempfile.mkdtemp(prefix="chapkit_workspace_extract_", dir=get_temp_dir()))

                # Extract zip to temp directory
                zip_buffer = BytesIO(workspace_content)
                with zipfile.ZipFile(zip_buffer, "r") as zf:
                    zf.extractall(extracted_workspace)

                # Determine how to pass model based on runner type
                from .runner import ShellModelRunner

                if isinstance(self.runner, ShellModelRunner):
                    # ShellModelRunner: pass workspace directory to runner
                    trained_model = {
                        "workspace_dir": str(extracted_workspace),
                    }
                else:
                    # FunctionalModelRunner with workspace: load pickled model from workspace
                    model_pickle_path = extracted_workspace / "model.pickle"
                    if model_pickle_path.exists():
                        try:
                            trained_model = pickle.loads(model_pickle_path.read_bytes())
                        except (pickle.UnpicklingError, EOFError, TypeError) as e:
                            raise ValueError(
                                f"Failed to load model from {model_pickle_path}: "
                                f"corrupted or incompatible pickle file. {e}"
                            ) from e
                    else:
                        raise ValueError(
                            f"Training artifact workspace missing model.pickle file at {model_pickle_path}"
                        )
            else:
                # Pickled model handling (FunctionalModelRunner without workspace)
                trained_model = training_data["content"]

            config_id = ULID.from_str(training_metadata["config_id"])

            # Load config
            async with self.database.session() as session:
                config_repo = ConfigRepository(session)
                config_manager: ConfigManager[ConfigT] = ConfigManager(config_repo, self.config_schema)
                config = await config_manager.find_by_id(config_id)

                if config is None:
                    raise ValueError(f"Config {config_id} not found")

                self._validate_prediction_periods(config.data)

            # Make predictions with timing
            prediction_started_at = datetime.datetime.now(datetime.UTC)
            prediction_result = await self.runner.on_predict(
                config=config.data,
                model=trained_model,
                historic=request.historic,
                future=request.future,
                geo=request.geo,
            )
            prediction_completed_at = datetime.datetime.now(datetime.UTC)
            prediction_duration = (prediction_completed_at - prediction_started_at).total_seconds()

            # Extract predictions from result (handles both new dict format and legacy DataFrame)
            if isinstance(prediction_result, dict) and "content" in prediction_result:
                # New unified format: {content, workspace_dir, exit_code, stdout, stderr}
                predictions = prediction_result["content"]
                workspace_dir_str = prediction_result.get("workspace_dir")
            else:
                # Legacy format: BaseModelRunner subclasses return DataFrame directly
                predictions = prediction_result
                workspace_dir_str = None

            # Create workspace artifact if workspace was created (both runners can produce workspace)
            if workspace_dir_str:
                prediction_workspace_dir = Path(workspace_dir_str)

                # Create workspace artifact data (zips workspace)
                # Runner returns ml_prediction_workspace type for debugging artifacts
                workspace_artifact_dict = await self.runner.create_prediction_artifact(
                    prediction_result=prediction_result,
                    config_id=str(config_id),
                    started_at=prediction_started_at,
                    completed_at=prediction_completed_at,
                    duration_seconds=round(prediction_duration, 2),
                )
            else:
                workspace_artifact_dict = None

            # Create prediction artifact with DataFrame (same for both runners)
            from chapkit.artifact.schemas import MLMetadata, MLPredictionArtifactData

            metadata = MLMetadata(
                status="success",
                config_id=str(config_id),
                started_at=prediction_started_at.isoformat(),
                completed_at=prediction_completed_at.isoformat(),
                duration_seconds=round(prediction_duration, 2),
            )

            # Create and validate artifact data structure with Pydantic
            artifact_data_model = MLPredictionArtifactData(
                type="ml_prediction",
                metadata=metadata,
                content=predictions,
                content_type="application/vnd.chapkit.dataframe+json",
                content_size=None,
            )

            # Construct dict manually to preserve Python objects (database uses PickleType)
            artifact_data_dict = {
                "type": artifact_data_model.type,
                "metadata": artifact_data_model.metadata.model_dump(),
                "content": predictions,  # Keep as Python object (DataFrame)
                "content_type": artifact_data_model.content_type,
                "content_size": artifact_data_model.content_size,
            }

            # Store artifacts
            async with self.database.session() as session:
                artifact_repo = ArtifactRepository(session)
                artifact_manager = ArtifactManager(artifact_repo)

                # 1. Store prediction artifact (level 1, parent = training artifact)
                await artifact_manager.save(
                    ArtifactIn(
                        id=artifact_id,
                        data=artifact_data_dict,
                        parent_id=request.artifact_id,
                        level=1,
                    )
                )

                # 2. Store workspace artifact if workspace was created (level 2, parent = prediction artifact)
                if workspace_artifact_dict is not None:
                    workspace_artifact_id = ULID()
                    await artifact_manager.save(
                        ArtifactIn(
                            id=workspace_artifact_id,
                            data=workspace_artifact_dict,
                            parent_id=artifact_id,
                            level=2,
                        )
                    )

        finally:
            # Cleanup extracted training workspace
            if extracted_workspace and extracted_workspace.exists():
                shutil.rmtree(extracted_workspace, ignore_errors=True)
            # Cleanup prediction workspace (created by ShellModelRunner)
            if prediction_workspace_dir and prediction_workspace_dir.exists():
                shutil.rmtree(prediction_workspace_dir, ignore_errors=True)

        return artifact_id
