"""Model runner implementations for ML train/predict operations."""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

import yaml
from geojson_pydantic import FeatureCollection
from servicekit.logging import get_logger

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from chapkit.utils import run_shell

ConfigT = TypeVar("ConfigT", bound=BaseConfig)

# Type aliases for ML runner functions
type TrainFunction[ConfigT] = Callable[[ConfigT, DataFrame, FeatureCollection | None], Awaitable[Any]]
type PredictFunction[ConfigT] = Callable[
    [ConfigT, Any, DataFrame, DataFrame, FeatureCollection | None], Awaitable[DataFrame]
]

logger = get_logger(__name__)


class BaseModelRunner(ABC, Generic[ConfigT]):
    """Abstract base class for model runners with lifecycle hooks."""

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

        Returns dict compatible with MLTrainingArtifactData structure.
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
            "type": "ml_training",
            "metadata": metadata.model_dump(),
            "content": training_result,  # Pickled model
            "content_type": "application/x-pickle",
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


class FunctionalModelRunner(BaseModelRunner[ConfigT]):
    """Functional model runner wrapping train and predict functions."""

    def __init__(
        self,
        on_train: TrainFunction[ConfigT],
        on_predict: PredictFunction[ConfigT],
    ) -> None:
        """Initialize functional runner with train and predict functions."""
        self._on_train = on_train
        self._on_predict = on_predict

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object."""
        return await self._on_train(config, data, geo)

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions using a trained model."""
        return await self._on_predict(config, model, historic, future, geo)


class ShellModelRunner(BaseModelRunner[ConfigT]):
    """Shell-based model runner that executes external scripts for train/predict operations."""

    def __init__(
        self,
        train_command: str,
        predict_command: str,
    ) -> None:
        """Initialize shell runner with full isolation support.

        The runner automatically copies the entire project directory (current working directory)
        to a temporary workspace, excluding .venv, node_modules, __pycache__, .git, and other
        build artifacts.

        Args:
            train_command: Command template for training (use relative paths)
            predict_command: Command template for prediction (use relative paths)
        """
        self.train_command = train_command
        self.predict_command = predict_command

        # Project root is current working directory
        # Users run: fastapi dev main.py (from project dir)
        # Docker sets WORKDIR to project root
        self.project_root = Path.cwd()

        logger.info("shell_runner_initialized", project_root=str(self.project_root))

    def _prepare_workspace(self, temp_dir: Path) -> None:
        """Prepare isolated workspace with full project directory copy.

        Copies the entire project directory to temp workspace, excluding build artifacts
        and virtual environments.

        Args:
            temp_dir: Temporary directory to copy project files into
        """
        shutil.copytree(
            self.project_root,
            temp_dir,
            ignore=shutil.ignore_patterns(
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
            ),
            dirs_exist_ok=True,
        )
        logger.info("copied_project_directory", src=str(self.project_root), dest=str(temp_dir))

    async def create_training_artifact(
        self,
        training_result: Any,
        config_id: str,
        started_at: datetime.datetime,
        completed_at: datetime.datetime,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Create artifact with workspace zip from training result."""
        import zipfile

        from chapkit.artifact.schemas import MLMetadata

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

        # Create workspace zip (compression level 9, stream to temp file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
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

            # Read zip into bytes
            workspace_content = zip_file_path.read_bytes()

            # Create metadata with exit_code, stdout, stderr
            metadata = MLMetadata(
                status=status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                config_id=config_id,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                duration_seconds=duration_seconds,
            )

            return {
                "type": "ml_training",
                "metadata": metadata.model_dump(),
                "content": workspace_content,
                "content_type": "application/zip",
                "content_size": len(workspace_content),
            }

        finally:
            # Cleanup temp zip file
            if zip_file_path.exists():
                zip_file_path.unlink()

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model by executing external training script (model file creation is optional)."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_train_"))

        try:
            # Copy entire project directory to temp workspace for full isolation
            self._prepare_workspace(temp_dir)

            # Write config to YAML file
            config_file = temp_dir / "config.yml"
            config_file.write_text(yaml.safe_dump(config.model_dump(), indent=2))

            # Write training data to CSV
            data_file = temp_dir / "data.csv"
            data.to_csv(data_file)

            # Write geo data if provided
            geo_file = temp_dir / "geo.json" if geo else None
            if geo:
                assert geo_file is not None  # For type checker
                geo_file.write_text(geo.model_dump_json(indent=2))

            # Substitute variables in command (use relative paths)
            command = self.train_command.format(
                data_file="data.csv",
                geo_file="geo.json" if geo_file else "",
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
                "workspace_dir": str(temp_dir),
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }

        except Exception:
            # Cleanup only on Python exception (not script failure)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions by executing external prediction script."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_predict_"))

        try:
            # Model must be workspace artifact from ShellModelRunner.on_train()
            if not (isinstance(model, dict) and "workspace_dir" in model):
                raise ValueError(
                    "ShellModelRunner.on_predict() requires workspace artifact from ShellModelRunner.on_train(). "
                    f"Got: {type(model)}"
                )

            # Extract and restore workspace from training artifact
            workspace_dir = Path(model["workspace_dir"])
            logger.info("predict_using_workspace", workspace_dir=str(workspace_dir))

            # Copy workspace contents to temp_dir (preserves all training artifacts)
            shutil.copytree(workspace_dir, temp_dir, dirs_exist_ok=True)

            # Write historic data (always fresh for each prediction)
            historic_file = temp_dir / "historic.csv"
            historic.to_csv(historic_file)

            # Write future data to CSV (always fresh for each prediction)
            future_file = temp_dir / "future.csv"
            future.to_csv(future_file)

            # Write geo data if provided (always fresh for each prediction)
            geo_file = temp_dir / "geo.json" if geo else None
            if geo:
                assert geo_file is not None  # For type checker
                geo_file.write_text(geo.model_dump_json(indent=2))

            # Output file path
            output_file = temp_dir / "predictions.csv"

            # Execute prediction command (workspace may contain model files, config, etc.)
            command = self.predict_command.format(
                historic_file="historic.csv",
                future_file="future.csv",
                output_file="predictions.csv",
                geo_file="geo.json" if geo_file else "",
            )

            logger.info("executing_predict_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess with cwd=temp_dir (scripts can now use relative imports!)
            result = await run_shell(command, cwd=str(temp_dir))
            stdout = result["stdout"]
            stderr = result["stderr"]

            if result["returncode"] != 0:
                logger.error("predict_script_failed", exit_code=result["returncode"], stderr=stderr)
                raise RuntimeError(f"Prediction script failed with exit code {result['returncode']}: {stderr}")

            logger.info("predict_script_completed", stdout=stdout, stderr=stderr)

            # Load predictions from file
            if not output_file.exists():
                raise RuntimeError(f"Prediction script did not create output file at {output_file}")

            predictions = DataFrame.from_csv(output_file)
            return predictions

        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)
