"""Model runner implementations for ML train/predict operations."""

from __future__ import annotations

import asyncio
import json
import pickle
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, TypeVar

import pandas as pd
from geojson_pydantic import FeatureCollection
from servicekit.logging import get_logger

from chapkit.config.schemas import BaseConfig

ConfigT = TypeVar("ConfigT", bound=BaseConfig)

# Type aliases for ML runner functions
type TrainFunction[ConfigT] = Callable[[ConfigT, pd.DataFrame, FeatureCollection | None], Awaitable[Any]]
type PredictFunction[ConfigT] = Callable[
    [ConfigT, Any, pd.DataFrame, pd.DataFrame, FeatureCollection | None], Awaitable[pd.DataFrame]
]

logger = get_logger(__name__)


class BaseModelRunner(ABC):
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
        config: BaseConfig,
        data: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object (must be pickleable)."""
        ...

    @abstractmethod
    async def on_predict(
        self,
        config: BaseConfig,
        model: Any,
        historic: pd.DataFrame,
        future: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> pd.DataFrame:
        """Make predictions using a trained model and return predictions as DataFrame."""
        ...


class FunctionalModelRunner(BaseModelRunner, Generic[ConfigT]):
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
        config: BaseConfig,
        data: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object."""
        return await self._on_train(config, data, geo)  # type: ignore[arg-type]

    async def on_predict(
        self,
        config: BaseConfig,
        model: Any,
        historic: pd.DataFrame,
        future: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> pd.DataFrame:
        """Make predictions using a trained model."""
        return await self._on_predict(config, model, historic, future, geo)  # type: ignore[arg-type]


class ShellModelRunner(BaseModelRunner):
    """Shell-based model runner that executes external scripts for train/predict operations."""

    def __init__(
        self,
        train_command: str,
        predict_command: str,
        model_format: str = "pickle",
    ) -> None:
        """Initialize shell runner with command templates for train/predict operations."""
        self.train_command = train_command
        self.predict_command = predict_command
        self.model_format = model_format

    async def on_train(
        self,
        config: BaseConfig,
        data: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model by executing external training script."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_train_"))

        try:
            # Write config to JSON file
            config_file = temp_dir / "config.json"
            config_file.write_text(json.dumps(config.model_dump(), indent=2))

            # Write training data to CSV
            data_file = temp_dir / "data.csv"
            data.to_csv(data_file, index=False)

            # Write geo data if provided
            geo_file = temp_dir / "geo.json" if geo else None
            if geo:
                assert geo_file is not None  # For type checker
                geo_file.write_text(geo.model_dump_json(indent=2))

            # Model file path
            model_file = temp_dir / f"model.{self.model_format}"

            # Substitute variables in command
            command = self.train_command.format(
                config_file=str(config_file),
                data_file=str(data_file),
                model_file=str(model_file),
                geo_file=str(geo_file) if geo_file else "",
            )

            logger.info("executing_train_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(temp_dir),
            )

            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""

            if process.returncode != 0:
                logger.error("train_script_failed", exit_code=process.returncode, stderr=stderr)
                raise RuntimeError(f"Training script failed with exit code {process.returncode}: {stderr}")

            logger.info("train_script_completed", stdout=stdout[:500], stderr=stderr[:500])

            # Load trained model from file
            if not model_file.exists():
                raise RuntimeError(f"Training script did not create model file at {model_file}")

            with open(model_file, "rb") as f:
                model = pickle.load(f)

            return model

        finally:
            # Cleanup temp files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    async def on_predict(
        self,
        config: BaseConfig,
        model: Any,
        historic: pd.DataFrame,
        future: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> pd.DataFrame:
        """Make predictions by executing external prediction script."""
        temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_ml_predict_"))

        try:
            # Write config to JSON file
            config_file = temp_dir / "config.json"
            config_file.write_text(json.dumps(config.model_dump(), indent=2))

            # Write model to file
            model_file = temp_dir / f"model.{self.model_format}"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            # Write historic data
            historic_file = temp_dir / "historic.csv"
            historic.to_csv(historic_file, index=False)

            # Write future data to CSV
            future_file = temp_dir / "future.csv"
            future.to_csv(future_file, index=False)

            # Write geo data if provided
            geo_file = temp_dir / "geo.json" if geo else None
            if geo:
                assert geo_file is not None  # For type checker
                geo_file.write_text(geo.model_dump_json(indent=2))

            # Output file path
            output_file = temp_dir / "predictions.csv"

            # Substitute variables in command
            command = self.predict_command.format(
                config_file=str(config_file),
                model_file=str(model_file),
                historic_file=str(historic_file),
                future_file=str(future_file),
                output_file=str(output_file),
                geo_file=str(geo_file) if geo_file else "",
            )

            logger.info("executing_predict_script", command=command, temp_dir=str(temp_dir))

            # Execute subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(temp_dir),
            )

            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""

            if process.returncode != 0:
                logger.error("predict_script_failed", exit_code=process.returncode, stderr=stderr)
                raise RuntimeError(f"Prediction script failed with exit code {process.returncode}: {stderr}")

            logger.info("predict_script_completed", stdout=stdout[:500], stderr=stderr[:500])

            # Load predictions from file
            if not output_file.exists():
                raise RuntimeError(f"Prediction script did not create output file at {output_file}")

            predictions = pd.read_csv(output_file)
            return predictions

        finally:
            # Cleanup temp files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
