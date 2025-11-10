"""Model runner implementations for ML train/predict operations."""

from __future__ import annotations

import pickle
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

# Type alias for cleanup policy
type CleanupPolicy = Literal["never", "on_success", "always"]

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
        model_format: str = "pickle",
        cleanup_policy: CleanupPolicy = "always",
    ) -> None:
        """Initialize shell runner with command templates for train/predict operations.

        Args:
            train_command: Command template for training
            predict_command: Command template for prediction
            model_format: File extension for model files (default: "pickle")
            cleanup_policy: When to delete temp directories:
                - "never": Keep all temp directories (useful for debugging)
                - "on_success": Delete only if operation succeeds (keeps failed runs for inspection)
                - "always": Always delete temp directories (default, saves disk space)
        """
        self.train_command = train_command
        self.predict_command = predict_command
        self.model_format = model_format
        self.cleanup_policy = cleanup_policy

    def _detect_source_dir(self) -> Path:
        """Detect source directory by finding where scripts referenced in commands are located."""
        import re

        cwd = Path.cwd()

        # Extract script paths from train_command (look for .py, .R, .jl, .sh files)
        script_pattern = r"(?:^|\s)([^\s{]+\.(?:py|R|jl|sh))(?:\s|$)"
        match = re.search(script_pattern, self.train_command)

        if match:
            script_path = match.group(1).strip()
            # Try to find this script relative to cwd
            potential_script = cwd / script_path

            if potential_script.exists():
                # Found the script - determine the source directory
                # If script is "scripts/train.py", we want to copy from cwd
                # If script is "train.py", we want to copy from cwd
                return cwd

            # Script doesn't exist at that path - might be in a subdirectory
            # Search for the script in cwd
            for found_script in cwd.rglob(Path(script_path).name):
                # Found a matching script name - use its parent's parent if it's in a scripts dir
                # Or use cwd if it's at root
                if "scripts" in str(found_script.parent):
                    # Script is in a scripts/ directory - return parent of scripts/
                    return found_script.parent.parent
                return found_script.parent

        # Fallback to cwd if can't detect
        return cwd

    def _create_debug_archive(self, temp_dir: Path, source_dir: Path, operation: str) -> Path:
        """Create a zip archive of temp directory for debugging."""
        from datetime import datetime, timezone
        from ulid import ULID

        # Create debug directory in source_dir
        debug_dir = source_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ulid = str(ULID())
        zip_name = f"chapkit_{operation}_{timestamp}_{ulid}"
        zip_path = debug_dir / zip_name

        # Create zip archive
        shutil.make_archive(str(zip_path), "zip", str(temp_dir))

        return Path(f"{zip_path}.zip")

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model by executing external training script (model file creation is optional)."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_train_"))
        succeeded = False

        try:
            # Copy source directory into temp directory for isolated execution
            source_dir = self._detect_source_dir()
            logger.info("copying_project_to_temp", source=str(source_dir), temp_dir=str(temp_dir))

            shutil.copytree(
                source_dir,
                temp_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    "*.pyc",
                    ".pytest_cache",
                    ".venv",
                    "venv",
                    ".tox",
                    "node_modules",
                ),
            )

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

            # Model file path
            model_file = temp_dir / f"model.{self.model_format}"

            # Substitute variables in command (using relative paths within temp_dir)
            command = self.train_command.format(
                config_file="config.yml",
                data_file="data.csv",
                model_file=f"model.{self.model_format}",
                geo_file="geo.json" if geo_file else "",
            )

            logger.info("executing_train_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess
            result = await run_shell(command, cwd=str(temp_dir))
            stdout = result["stdout"]
            stderr = result["stderr"]

            if result["returncode"] != 0:
                logger.error("train_script_failed", exit_code=result["returncode"], stderr=stderr)
                raise RuntimeError(f"Training script failed with exit code {result['returncode']}: {stderr}")

            logger.info("train_script_completed", stdout=stdout[:500], stderr=stderr[:500])

            # Load trained model from file if it exists
            if model_file.exists():
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                succeeded = True
                return model
            else:
                # Return metadata placeholder when no model file is created
                logger.info("train_script_no_model_file", model_file=str(model_file))
                succeeded = True
                return {
                    "model_type": "no_file",
                    "stdout": stdout,
                    "stderr": stderr,
                    "temp_dir": str(temp_dir),
                }

        finally:
            # Cleanup temp files based on policy
            if self.cleanup_policy == "always":
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif self.cleanup_policy == "on_success" and succeeded:
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif self.cleanup_policy == "never" or (self.cleanup_policy == "on_success" and not succeeded):
                # Create debug archive before cleanup
                zip_path = self._create_debug_archive(temp_dir, source_dir, "train")
                logger.info(
                    "debug_archive_created",
                    zip_path=str(zip_path),
                    temp_dir=str(temp_dir),
                    reason=self.cleanup_policy,
                )
                # Clean up temp directory after zipping
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions by executing external prediction script (skips model file if placeholder)."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_predict_"))
        succeeded = False

        try:
            # Copy source directory into temp directory for isolated execution
            source_dir = self._detect_source_dir()
            logger.info("copying_project_to_temp", source=str(source_dir), temp_dir=str(temp_dir))

            shutil.copytree(
                source_dir,
                temp_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    "*.pyc",
                    ".pytest_cache",
                    ".venv",
                    "venv",
                    ".tox",
                    "node_modules",
                ),
            )

            # Write config to YAML file
            config_file = temp_dir / "config.yml"
            config_file.write_text(yaml.safe_dump(config.model_dump(), indent=2))

            # Write model to file only if it's not a placeholder
            is_placeholder = isinstance(model, dict) and model.get("model_type") == "no_file"
            if is_placeholder:
                logger.info("predict_script_no_model_file", reason="model is placeholder")
                model_file = None
            else:
                model_file = temp_dir / f"model.{self.model_format}"
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)

            # Write historic data
            historic_file = temp_dir / "historic.csv"
            historic.to_csv(historic_file)

            # Write future data to CSV
            future_file = temp_dir / "future.csv"
            future.to_csv(future_file)

            # Write geo data if provided
            geo_file = temp_dir / "geo.json" if geo else None
            if geo:
                assert geo_file is not None  # For type checker
                geo_file.write_text(geo.model_dump_json(indent=2))

            # Output file path
            output_file = temp_dir / "predictions.csv"

            # Substitute variables in command (using relative paths within temp_dir)
            command = self.predict_command.format(
                config_file="config.yml",
                model_file=f"model.{self.model_format}" if not is_placeholder else "",
                historic_file="historic.csv",
                future_file="future.csv",
                output_file="predictions.csv",
                geo_file="geo.json" if geo_file else "",
            )

            logger.info("executing_predict_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess
            result = await run_shell(command, cwd=str(temp_dir))
            stdout = result["stdout"]
            stderr = result["stderr"]

            if result["returncode"] != 0:
                logger.error("predict_script_failed", exit_code=result["returncode"], stderr=stderr)
                raise RuntimeError(f"Prediction script failed with exit code {result['returncode']}: {stderr}")

            logger.info("predict_script_completed", stdout=stdout[:500], stderr=stderr[:500])

            # Load predictions from file
            if not output_file.exists():
                raise RuntimeError(f"Prediction script did not create output file at {output_file}")

            predictions = DataFrame.from_csv(output_file)
            succeeded = True
            return predictions

        finally:
            # Cleanup temp files based on policy
            if self.cleanup_policy == "always":
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif self.cleanup_policy == "on_success" and succeeded:
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif self.cleanup_policy == "never" or (self.cleanup_policy == "on_success" and not succeeded):
                # Create debug archive before cleanup
                zip_path = self._create_debug_archive(temp_dir, source_dir, "predict")
                logger.info(
                    "debug_archive_created",
                    zip_path=str(zip_path),
                    temp_dir=str(temp_dir),
                    reason=self.cleanup_policy,
                )
                # Clean up temp directory after zipping
                shutil.rmtree(temp_dir, ignore_errors=True)
