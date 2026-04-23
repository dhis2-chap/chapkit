"""Model runner implementations for ML train/predict operations."""

from __future__ import annotations

import datetime
import os
import pickle
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

import yaml
from geojson_pydantic import FeatureCollection
from servicekit.logging import get_logger

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml.schemas import ValidationDiagnostic
from chapkit.utils import run_shell

ConfigT = TypeVar("ConfigT", bound=BaseConfig)

# Type aliases for ML runner functions
type TrainFunction[ConfigT] = Callable[[ConfigT, DataFrame, FeatureCollection | None], Awaitable[Any]]
type PredictFunction[ConfigT] = Callable[
    [ConfigT, Any, DataFrame, DataFrame, FeatureCollection | None], Awaitable[DataFrame]
]
type ValidateTrainFunction[ConfigT] = Callable[
    [ConfigT, DataFrame, FeatureCollection | None], Awaitable[list[ValidationDiagnostic]]
]
type ValidatePredictFunction[ConfigT] = Callable[
    [ConfigT, DataFrame, DataFrame, FeatureCollection | None], Awaitable[list[ValidationDiagnostic]]
]

logger = get_logger(__name__)


def get_temp_dir() -> str | None:
    """Get temp directory from CHAPKIT_TEMP_DIR or use system default."""
    return os.getenv("CHAPKIT_TEMP_DIR")


# Patterns to exclude when copying project to workspace
WORKSPACE_EXCLUDE_PATTERNS = (
    # Python
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    # JavaScript/Node
    "node_modules",
    # Version control
    ".git",
    ".gitignore",
    # IDEs
    ".vscode",
    ".idea",
    ".DS_Store",
    # Build artifacts
    "build",
    "dist",
    "*.so",
    "*.dylib",
    # Databases
    "*.db",
    "*.sqlite",
    "*.sqlite3",
)


def prepare_workspace(source_dir: Path, dest_dir: Path) -> None:
    """Copy project directory to workspace, excluding build artifacts and virtual environments."""
    shutil.copytree(
        source_dir,
        dest_dir,
        ignore=shutil.ignore_patterns(*WORKSPACE_EXCLUDE_PATTERNS),
        dirs_exist_ok=True,
    )
    logger.info("copied_project_directory", src=str(source_dir), dest=str(dest_dir))


def write_training_inputs(
    workspace_dir: Path,
    config: BaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None,
) -> None:
    """Write training input files (config.yml, data.csv, geo.json) to workspace."""
    (workspace_dir / "config.yml").write_text(yaml.safe_dump(config.model_dump(), indent=2))
    data.to_csv(workspace_dir / "data.csv")
    if geo:
        (workspace_dir / "geo.json").write_text(geo.model_dump_json(indent=2))


def write_prediction_inputs(
    workspace_dir: Path,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None,
) -> None:
    """Write prediction input files (historic.csv, future.csv, geo.json) to workspace."""
    historic.to_csv(workspace_dir / "historic.csv")
    future.to_csv(workspace_dir / "future.csv")
    if geo:
        (workspace_dir / "geo.json").write_text(geo.model_dump_json(indent=2))


def zip_workspace(workspace_dir: Path) -> bytes:
    """Zip workspace directory and return bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=get_temp_dir()) as tmp:
        zip_file_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for root, _, files in os.walk(workspace_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(workspace_dir)
                    zf.write(file_path, arcname)

        # Validate zip integrity
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                raise ValueError(f"Corrupted file in workspace zip: {bad_file}")

        return zip_file_path.read_bytes()

    finally:
        if zip_file_path.exists():
            zip_file_path.unlink()


def create_workspace_artifact(
    workspace_content: bytes,
    artifact_type: str,
    config_id: str,
    started_at: datetime.datetime,
    completed_at: datetime.datetime,
    duration_seconds: float,
    status: Literal["success", "failed"] = "success",
    exit_code: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    """Create artifact dict from workspace ZIP content."""
    from chapkit.artifact.schemas import MLMetadata

    metadata = MLMetadata(
        status=status,
        config_id=config_id,
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_seconds=duration_seconds,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )

    return {
        "type": artifact_type,
        "metadata": metadata.model_dump(),
        "content": workspace_content,
        "content_type": "application/zip",
        "content_size": len(workspace_content),
    }


class BaseModelRunner(ABC, Generic[ConfigT]):
    """Abstract base class for model runners with lifecycle hooks."""

    @property
    def supports_train(self) -> bool:
        """Return True when this runner can execute a train step."""
        return True

    async def on_init(self) -> None:
        """Optional initialization hook called before training or prediction."""
        pass

    async def on_cleanup(self) -> None:
        """Optional cleanup hook called after training or prediction."""
        pass

    async def create_training_artifact(
        self,
        training_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact data structure from training result.

        Default implementation assumes training_result is a pickleable object.
        Runners can override to customize artifact creation (e.g., workspace zipping).

        Returns dict compatible with MLTrainingWorkspaceArtifactData structure.
        """
        from chapkit.artifact.schemas import MLMetadata

        metadata = MLMetadata(
            status="success",
            config_id=config_id,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration_seconds,
        )

        return {
            "type": "ml_training_workspace",
            "metadata": metadata.model_dump(),
            "content": training_result,  # Pickled model
            "content_type": "application/x-pickle",
            "content_size": None,
        }

    async def create_prediction_artifact(
        self,
        prediction_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact data structure from prediction result.

        Default implementation assumes prediction_result is a DataFrame.
        Runners can override to customize artifact creation (e.g., workspace zipping).

        Returns dict compatible with MLPredictionArtifactData structure.
        """
        from chapkit.artifact.schemas import MLMetadata

        metadata = MLMetadata(
            status="success",
            config_id=config_id,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration_seconds,
        )

        return {
            "type": "ml_prediction",
            "metadata": metadata.model_dump(),
            "content": prediction_result,  # DataFrame
            "content_type": "application/vnd.chapkit.dataframe+json",
            "content_size": None,
        }

    @abstractmethod
    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object (must be pickleable)."""
        ...

    @abstractmethod
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

    async def on_validate_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Return domain-specific diagnostics for a train payload (default: empty)."""
        return []

    async def on_validate_predict(
        self,
        config: ConfigT,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Return domain-specific diagnostics for a predict payload (default: empty)."""
        return []


class FunctionalModelRunner(BaseModelRunner[ConfigT]):
    """Functional model runner wrapping train and predict functions."""

    def __init__(
        self,
        on_predict: PredictFunction[ConfigT],
        on_train: TrainFunction[ConfigT] | None = None,
        on_validate_train: ValidateTrainFunction[ConfigT] | None = None,
        on_validate_predict: ValidatePredictFunction[ConfigT] | None = None,
    ) -> None:
        """Initialize functional runner with predict, and optional train/validate functions."""
        self._on_train = on_train
        self._on_predict = on_predict
        self._on_validate_train = on_validate_train
        self._on_validate_predict = on_validate_predict

    @property
    def supports_train(self) -> bool:
        """True when a train callback was supplied."""
        return self._on_train is not None

    async def on_validate_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Delegate to the provided validate-train callback, or return no diagnostics."""
        if self._on_validate_train is None:
            return []
        return await self._on_validate_train(config, data, geo)

    async def on_validate_predict(
        self,
        config: ConfigT,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Delegate to the provided validate-predict callback, or return no diagnostics."""
        if self._on_validate_predict is None:
            return []
        return await self._on_validate_predict(config, historic, future, geo)

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return dict with content and workspace.

        Returns:
            Dict with keys: content (model), workspace_dir, exit_code, stdout, stderr
        """
        if self._on_train is None:
            raise RuntimeError("FunctionalModelRunner has no on_train callback; this runner is stateless")

        workspace_dir = Path(tempfile.mkdtemp(prefix="chapkit_functional_train_", dir=get_temp_dir()))
        # Copy full project directory for reproducibility
        prepare_workspace(Path.cwd(), workspace_dir)
        # Write training input files
        write_training_inputs(workspace_dir, config, data, geo)

        # Execute training function
        model = await self._on_train(config, data, geo)

        # Write model.pickle
        (workspace_dir / "model.pickle").write_bytes(pickle.dumps(model))

        return {
            "content": model,
            "workspace_dir": str(workspace_dir),
            "exit_code": None,
            "stdout": None,
            "stderr": None,
        }

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Make predictions and return dict with content and workspace.

        Returns:
            Dict with keys: content (DataFrame), workspace_dir, exit_code, stdout, stderr
        """
        workspace_dir = Path(tempfile.mkdtemp(prefix="chapkit_functional_predict_", dir=get_temp_dir()))
        # Copy full project directory for reproducibility
        prepare_workspace(Path.cwd(), workspace_dir)
        # Write config.yml
        (workspace_dir / "config.yml").write_text(yaml.safe_dump(config.model_dump(), indent=2))
        # Write prediction input files
        write_prediction_inputs(workspace_dir, historic, future, geo)
        # Write model.pickle (input model for prediction; omitted in stateless mode)
        if model is not None:
            (workspace_dir / "model.pickle").write_bytes(pickle.dumps(model))

        # Execute prediction function
        predictions = await self._on_predict(config, model, historic, future, geo)

        # Write predictions.csv
        predictions.to_csv(workspace_dir / "predictions.csv")

        return {
            "content": predictions,
            "workspace_dir": str(workspace_dir),
            "exit_code": None,
            "stdout": None,
            "stderr": None,
        }

    async def create_training_artifact(
        self,
        training_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact from training result by zipping workspace."""
        # Extract content and workspace from unified result dict
        if isinstance(training_result, dict) and "content" in training_result:
            workspace_dir = training_result.get("workspace_dir")
        else:
            workspace_dir = None

        if not workspace_dir:
            raise ValueError(
                "FunctionalModelRunner.create_training_artifact() requires workspace dict from on_train(). "
                "Got result without workspace_dir."
            )

        # Zip workspace like ShellModelRunner
        return await self._create_workspace_artifact(
            workspace_dir=Path(workspace_dir),
            config_id=config_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            artifact_type="ml_training_workspace",
        )

    async def create_prediction_artifact(
        self,
        prediction_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact from prediction result by zipping workspace."""
        # Extract content and workspace from unified result dict
        if isinstance(prediction_result, dict) and "content" in prediction_result:
            workspace_dir = prediction_result.get("workspace_dir")
        else:
            workspace_dir = None

        if not workspace_dir:
            raise ValueError(
                "FunctionalModelRunner.create_prediction_artifact() requires workspace dict from on_predict(). "
                "Got result without workspace_dir."
            )

        # Zip workspace like ShellModelRunner (workspace artifact for debugging)
        return await self._create_workspace_artifact(
            workspace_dir=Path(workspace_dir),
            config_id=config_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            artifact_type="ml_prediction_workspace",
        )

    async def _create_workspace_artifact(
        self,
        workspace_dir: Path,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
        artifact_type: str,
    ) -> dict[str, Any]:
        """Create artifact with workspace zip."""
        workspace_content = zip_workspace(workspace_dir)
        return create_workspace_artifact(
            workspace_content=workspace_content,
            artifact_type=artifact_type,
            config_id=config_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
        )


class ShellModelRunner(BaseModelRunner[ConfigT]):
    """Shell-based model runner that executes external scripts for train/predict operations."""

    def __init__(
        self,
        predict_command: str,
        train_command: str | None = None,
        on_validate_train: ValidateTrainFunction[ConfigT] | None = None,
        on_validate_predict: ValidatePredictFunction[ConfigT] | None = None,
    ) -> None:
        """Initialize shell runner with full isolation support.

        The runner automatically copies the entire project directory (current working directory)
        to a temporary workspace, excluding .venv, node_modules, __pycache__, .git, and other
        build artifacts.

        Args:
            predict_command: Command template for prediction (use relative paths)
            train_command: Optional command template for training (use relative paths).
                           Omit for stateless predict-only services.
            on_validate_train: Optional Python callback for domain-level train validation
            on_validate_predict: Optional Python callback for domain-level predict validation
        """
        self.train_command = train_command
        self.predict_command = predict_command
        self._on_validate_train = on_validate_train
        self._on_validate_predict = on_validate_predict

        # Project root is current working directory
        # Users run: fastapi dev main.py (from project dir)
        # Docker sets WORKDIR to project root
        self.project_root = Path.cwd()

        logger.info("shell_runner_initialized", project_root=str(self.project_root))

    @property
    def supports_train(self) -> bool:
        """True when a train command was supplied."""
        return self.train_command is not None

    async def on_validate_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Delegate to the provided validate-train callback, or return no diagnostics."""
        if self._on_validate_train is None:
            return []
        return await self._on_validate_train(config, data, geo)

    async def on_validate_predict(
        self,
        config: ConfigT,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Delegate to the provided validate-predict callback, or return no diagnostics."""
        if self._on_validate_predict is None:
            return []
        return await self._on_validate_predict(config, historic, future, geo)

    async def create_training_artifact(
        self,
        training_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact with workspace zip from training result."""
        # Validate training_result is workspace dict from on_train()
        if not isinstance(training_result, dict) or "workspace_dir" not in training_result:
            raise ValueError(
                "ShellModelRunner.create_training_artifact() requires workspace dict from on_train(). "
                f"Got: {type(training_result)}"
            )

        # Extract workspace info from training_result dict
        workspace_dir = Path(training_result["workspace_dir"])
        exit_code = training_result["exit_code"]
        stdout = training_result.get("stdout", "")
        stderr = training_result.get("stderr", "")

        # Determine status from exit code
        status: Literal["success", "failed"] = "success" if exit_code == 0 else "failed"

        # Zip workspace and create artifact
        workspace_content = zip_workspace(workspace_dir)
        return create_workspace_artifact(
            workspace_content=workspace_content,
            artifact_type="ml_training_workspace",
            config_id=config_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model by executing external training script (model file creation is optional)."""
        if self.train_command is None:
            raise RuntimeError("ShellModelRunner has no train_command; this runner is stateless")

        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_train_", dir=get_temp_dir()))

        try:
            # Copy entire project directory to temp workspace for full isolation
            prepare_workspace(self.project_root, temp_dir)
            # Write training input files
            write_training_inputs(temp_dir, config, data, geo)

            # Substitute variables in command (use relative paths)
            command = self.train_command.format(
                data_file="data.csv",
                geo_file="geo.json" if geo else "",
            )

            logger.info("executing_train_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess with cwd=temp_dir (scripts can now use relative imports!)
            result = await run_shell(command, cwd=str(temp_dir))
            stdout = result["stdout"]
            stderr = result["stderr"]
            exit_code = result["returncode"]

            if exit_code != 0:
                logger.error("train_script_failed", exit_code=exit_code, stderr=stderr)
            else:
                logger.info("train_script_completed", stdout=stdout, stderr=stderr)

            # Return workspace directory for artifact storage
            # Workspace preserved for both success and failure (manager will store artifact)
            return {
                "content": None,  # ShellRunner doesn't have in-memory model
                "workspace_dir": str(temp_dir),
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }

        except Exception:
            # Cleanup only on Python exception (not script failure)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def create_prediction_artifact(
        self,
        prediction_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact with workspace zip from prediction result."""
        # Validate prediction_result is workspace dict from on_predict()
        if not isinstance(prediction_result, dict) or "workspace_dir" not in prediction_result:
            raise ValueError(
                "ShellModelRunner.create_prediction_artifact() requires workspace dict from on_predict(). "
                f"Got: {type(prediction_result)}"
            )

        # Extract workspace info from prediction_result dict
        workspace_dir = Path(prediction_result["workspace_dir"])
        exit_code = prediction_result["exit_code"]
        stdout = prediction_result.get("stdout", "")
        stderr = prediction_result.get("stderr", "")

        # Determine status from exit code
        status: Literal["success", "failed"] = "success" if exit_code == 0 else "failed"

        # Zip workspace and create artifact
        workspace_content = zip_workspace(workspace_dir)
        return create_workspace_artifact(
            workspace_content=workspace_content,
            artifact_type="ml_prediction_workspace",
            config_id=config_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Make predictions by executing external prediction script."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_predict_", dir=get_temp_dir()))

        try:
            if isinstance(model, dict) and "workspace_dir" in model:
                # Train-backed: restore workspace from training artifact
                workspace_dir = Path(model["workspace_dir"])
                logger.info("predict_using_workspace", workspace_dir=str(workspace_dir))
                shutil.copytree(workspace_dir, temp_dir, dirs_exist_ok=True)
            elif model is None:
                # Stateless: build a fresh workspace from the project root
                logger.info("predict_stateless", project_root=str(self.project_root))
                prepare_workspace(self.project_root, temp_dir)
                # Write config.yml so predict scripts can still read it
                (temp_dir / "config.yml").write_text(yaml.safe_dump(config.model_dump(), indent=2))
            else:
                raise ValueError(
                    "ShellModelRunner.on_predict() expects either a workspace dict from on_train() "
                    f"or None for stateless mode. Got: {type(model)}"
                )

            # Write prediction input files (always fresh for each prediction)
            write_prediction_inputs(temp_dir, historic, future, geo)

            # Output file path
            output_file = temp_dir / "predictions.csv"

            # Execute prediction command (workspace may contain model files, config, etc.)
            command = self.predict_command.format(
                historic_file="historic.csv",
                future_file="future.csv",
                output_file="predictions.csv",
                geo_file="geo.json" if geo else "",
            )

            logger.info("executing_predict_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess with cwd=temp_dir (scripts can now use relative imports!)
            result = await run_shell(command, cwd=str(temp_dir))
            stdout = result["stdout"]
            stderr = result["stderr"]
            exit_code = result["returncode"]

            if exit_code != 0:
                logger.error("predict_script_failed", exit_code=exit_code, stderr=stderr)
            else:
                logger.info("predict_script_completed", stdout=stdout, stderr=stderr)

            # Load predictions from file
            if not output_file.exists():
                raise RuntimeError(f"Prediction script did not create output file at {output_file}")

            predictions = DataFrame.from_csv(output_file)

            # Return workspace directory for artifact storage (like on_train)
            # Workspace preserved for both success and failure (manager will store artifact)
            return {
                "content": predictions,  # DataFrame loaded from predictions.csv
                "workspace_dir": str(temp_dir),
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }

        except Exception:
            # Cleanup only on Python exception (not script failure)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
