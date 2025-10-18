"""Tests for class-based ModelRunner implementations."""

from typing import Any

import pandas as pd
import pytest
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.ml import BaseModelRunner


class MockConfig(BaseConfig):
    """Mock config for testing."""

    threshold: float = 0.5


class SimpleRunner(BaseModelRunner):
    """Simple test runner for basic functionality."""

    def __init__(self) -> None:
        """Initialize runner."""
        self.trained_models: list[str] = []
        self.predictions: list[int] = []
        self.init_called: bool = False
        self.cleanup_called: bool = False

    async def on_init(self) -> None:
        """Track initialization."""
        self.init_called = True

    async def on_cleanup(self) -> None:
        """Track cleanup."""
        self.cleanup_called = True

    async def on_train(
        self,
        config: BaseConfig,
        data: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Simple training implementation."""
        model_id = f"model_{len(self.trained_models)}"
        self.trained_models.append(model_id)
        return {"model_id": model_id, "sample_count": len(data)}

    async def on_predict(
        self,
        config: BaseConfig,
        model: Any,
        historic: pd.DataFrame | None,
        future: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> pd.DataFrame:
        """Simple prediction implementation."""
        pred_count = len(future)
        self.predictions.append(pred_count)
        future["prediction"] = 1.0
        return future


class StatefulRunner(BaseModelRunner):
    """Runner with shared state between train and predict."""

    def __init__(self) -> None:
        """Initialize with shared state."""
        self.feature_names: list[str] = []
        self.normalization_params: dict[str, float] = {}

    async def on_train(
        self,
        config: BaseConfig,
        data: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train and store feature info."""
        self.feature_names = list(data.columns)
        self.normalization_params = {col: float(data[col].mean()) for col in data.columns if col != "target"}

        return {"features": self.feature_names, "params": self.normalization_params}

    async def on_predict(
        self,
        config: BaseConfig,
        model: Any,
        historic: pd.DataFrame | None,
        future: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> pd.DataFrame:
        """Predict using stored state."""
        # Use stored feature names and normalization
        for col in self.feature_names:
            if col in future.columns and col in self.normalization_params:
                future[f"{col}_normalized"] = future[col] - self.normalization_params[col]

        future["prediction"] = 1.0
        return future


def test_base_model_runner_is_abstract() -> None:
    """Test that BaseModelRunner cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseModelRunner()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_simple_runner_train() -> None:
    """Test basic training with simple runner."""
    runner = SimpleRunner()
    config = MockConfig()
    data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

    model = await runner.on_train(config, data)

    assert isinstance(model, dict)
    assert model["model_id"] == "model_0"
    assert model["sample_count"] == 3
    assert len(runner.trained_models) == 1


@pytest.mark.asyncio
async def test_simple_runner_predict() -> None:
    """Test basic prediction with simple runner."""
    runner = SimpleRunner()
    config = MockConfig()
    model = {"model_id": "model_0"}
    future = pd.DataFrame({"feature1": [4, 5, 6]})

    result = await runner.on_predict(config, model, None, future)

    assert "prediction" in result.columns
    assert len(result) == 3
    assert result["prediction"].iloc[0] == 1.0
    assert len(runner.predictions) == 1


@pytest.mark.asyncio
async def test_lifecycle_hooks_called() -> None:
    """Test that lifecycle hooks are called during train/predict."""
    runner = SimpleRunner()
    config = MockConfig()
    data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

    # Train calls lifecycle hooks
    runner.init_called = False
    runner.cleanup_called = False
    await runner.on_init()
    model = await runner.on_train(config, data)
    await runner.on_cleanup()

    assert runner.init_called
    assert runner.cleanup_called

    # Predict calls lifecycle hooks
    runner.init_called = False
    runner.cleanup_called = False
    await runner.on_init()
    future = pd.DataFrame({"feature1": [4, 5, 6]})
    await runner.on_predict(config, model, None, future)
    await runner.on_cleanup()

    assert runner.init_called
    assert runner.cleanup_called


@pytest.mark.asyncio
async def test_stateful_runner_shares_state() -> None:
    """Test that runner can share state between train and predict."""
    runner = StatefulRunner()
    config = MockConfig()

    # Train stores state
    train_data = pd.DataFrame({"temp": [20.0, 25.0, 30.0], "humidity": [60.0, 70.0, 80.0], "target": [0, 1, 0]})
    model = await runner.on_train(config, train_data)

    # Check state was stored
    assert len(runner.feature_names) == 3
    assert "temp" in runner.normalization_params
    assert "humidity" in runner.normalization_params
    assert runner.normalization_params["temp"] == 25.0
    assert runner.normalization_params["humidity"] == 70.0

    # Predict uses stored state
    predict_data = pd.DataFrame({"temp": [22.0, 28.0], "humidity": [65.0, 75.0]})
    result = await runner.on_predict(config, model, None, predict_data)

    # Check normalized columns were added using stored params
    assert "temp_normalized" in result.columns
    assert "humidity_normalized" in result.columns
    assert result["temp_normalized"].iloc[0] == -3.0  # 22 - 25
    assert result["humidity_normalized"].iloc[0] == -5.0  # 65 - 70


@pytest.mark.asyncio
async def test_multiple_train_predict_cycles() -> None:
    """Test multiple train and predict cycles."""
    runner = SimpleRunner()
    config = MockConfig()

    # First train-predict cycle
    data1 = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
    model1 = await runner.on_train(config, data1)
    future1 = pd.DataFrame({"feature1": [4, 5]})
    await runner.on_predict(config, model1, None, future1)

    # Second train-predict cycle
    data2 = pd.DataFrame({"feature1": [10, 20, 30, 40], "target": [1, 0, 1, 0]})
    model2 = await runner.on_train(config, data2)
    future2 = pd.DataFrame({"feature1": [50, 60, 70]})
    await runner.on_predict(config, model2, None, future2)

    # Verify both cycles tracked
    assert len(runner.trained_models) == 2
    assert len(runner.predictions) == 2
    assert runner.trained_models[0] == "model_0"
    assert runner.trained_models[1] == "model_1"


@pytest.mark.asyncio
async def test_runner_with_geo_data() -> None:
    """Test runner can handle optional geospatial data."""
    runner = SimpleRunner()
    config = MockConfig()
    data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

    # Mock GeoJSON FeatureCollection
    geo: FeatureCollection = FeatureCollection(type="FeatureCollection", features=[])

    # Should work with geo data
    model = await runner.on_train(config, data, geo)
    future = pd.DataFrame({"feature1": [4, 5]})
    result = await runner.on_predict(config, model, None, future, geo)

    assert model is not None
    assert len(result) == 2


@pytest.mark.asyncio
async def test_runner_with_historic_data() -> None:
    """Test runner can handle optional historic data."""
    runner = SimpleRunner()
    config = MockConfig()
    model = {"model_id": "test_model"}

    # With historic data
    historic = pd.DataFrame({"feature1": [1, 2, 3]})
    future = pd.DataFrame({"feature1": [4, 5]})

    result = await runner.on_predict(config, model, historic, future)

    assert len(result) == 2
    assert "prediction" in result.columns


@pytest.mark.asyncio
async def test_default_lifecycle_hooks_do_nothing() -> None:
    """Test that default lifecycle hooks don't raise errors."""

    class MinimalRunner(BaseModelRunner):
        """Runner without overriding lifecycle hooks."""

        async def on_train(
            self,
            config: BaseConfig,
            data: pd.DataFrame,
            geo: FeatureCollection | None = None,
        ) -> Any:
            """Minimal train."""
            return "model"

        async def on_predict(
            self,
            config: BaseConfig,
            model: Any,
            historic: pd.DataFrame | None,
            future: pd.DataFrame,
            geo: FeatureCollection | None = None,
        ) -> pd.DataFrame:
            """Minimal predict."""
            return future

    runner = MinimalRunner()

    # Default hooks should not raise
    await runner.on_init()
    await runner.on_cleanup()

    # Should work without explicitly calling hooks
    config = MockConfig()
    data = pd.DataFrame({"col1": [1, 2]})
    model = await runner.on_train(config, data)
    future = pd.DataFrame({"col1": [3, 4]})
    result = await runner.on_predict(config, model, None, future)

    assert model == "model"
    assert len(result) == 2
