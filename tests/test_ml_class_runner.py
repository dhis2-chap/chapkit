"""Tests for class-based ModelRunner implementations."""

from typing import Any

import pytest
from geojson_pydantic import FeatureCollection
from servicekit.data import DataFrame

from chapkit import BaseConfig
from chapkit.ml import BaseModelRunner


class MockConfig(BaseConfig):
    """Mock config for testing."""

    threshold: float = 0.5


class SimpleRunner(BaseModelRunner[MockConfig]):
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
        config: MockConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Simple training implementation."""
        model_id = f"model_{len(self.trained_models)}"
        self.trained_models.append(model_id)
        return {"model_id": model_id, "sample_count": len(data.data)}

    async def on_predict(
        self,
        config: MockConfig,
        model: Any,
        historic: DataFrame | None,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Simple prediction implementation."""
        pred_count = len(future.data)
        self.predictions.append(pred_count)
        return future.add_column("prediction", [1.0] * len(future.data))


class StatefulRunner(BaseModelRunner[MockConfig]):
    """Runner with shared state between train and predict."""

    def __init__(self) -> None:
        """Initialize with shared state."""
        self.feature_names: list[str] = []
        self.normalization_params: dict[str, float] = {}

    async def on_train(
        self,
        config: MockConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train and store feature info."""
        self.feature_names = data.columns.copy()
        self.normalization_params = {}

        for col in data.columns:
            if col != "target":
                col_idx = data.columns.index(col)
                col_data = [row[col_idx] for row in data.data]
                self.normalization_params[col] = sum(col_data) / len(col_data)

        return {"features": self.feature_names, "params": self.normalization_params}

    async def on_predict(
        self,
        config: MockConfig,
        model: Any,
        historic: DataFrame | None,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Predict using stored state."""
        result = future

        # Use stored feature names and normalization
        for col in self.feature_names:
            if col in future.columns and col in self.normalization_params:
                col_idx = future.columns.index(col)
                normalized_values = [row[col_idx] - self.normalization_params[col] for row in future.data]
                result = result.add_column(f"{col}_normalized", normalized_values)

        result = result.add_column("prediction", [1.0] * len(result.data))
        return result


def test_base_model_runner_is_abstract() -> None:
    """Test that BaseModelRunner cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseModelRunner()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_simple_runner_train() -> None:
    """Test basic training with simple runner."""
    runner = SimpleRunner()
    config = MockConfig()
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1], [3, 0]])

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
    future = DataFrame(columns=["feature1"], data=[[4], [5], [6]])

    result = await runner.on_predict(config, model, None, future)

    assert "prediction" in result.columns
    assert len(result.data) == 3
    pred_idx = result.columns.index("prediction")
    assert result.data[0][pred_idx] == 1.0
    assert len(runner.predictions) == 1


@pytest.mark.asyncio
async def test_lifecycle_hooks_called() -> None:
    """Test that lifecycle hooks are called during train/predict."""
    runner = SimpleRunner()
    config = MockConfig()
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1], [3, 0]])

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
    future = DataFrame(columns=["feature1"], data=[[4], [5], [6]])
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
    train_data = DataFrame(
        columns=["temp", "humidity", "target"], data=[[20.0, 60.0, 0], [25.0, 70.0, 1], [30.0, 80.0, 0]]
    )
    model = await runner.on_train(config, train_data)

    # Check state was stored
    assert len(runner.feature_names) == 3
    assert "temp" in runner.normalization_params
    assert "humidity" in runner.normalization_params
    assert runner.normalization_params["temp"] == 25.0
    assert runner.normalization_params["humidity"] == 70.0

    # Predict uses stored state
    predict_data = DataFrame(columns=["temp", "humidity"], data=[[22.0, 65.0], [28.0, 75.0]])
    result = await runner.on_predict(config, model, None, predict_data)

    # Check normalized columns were added using stored params
    assert "temp_normalized" in result.columns
    assert "humidity_normalized" in result.columns
    temp_norm_idx = result.columns.index("temp_normalized")
    humidity_norm_idx = result.columns.index("humidity_normalized")
    assert result.data[0][temp_norm_idx] == -3.0  # 22 - 25
    assert result.data[0][humidity_norm_idx] == -5.0  # 65 - 70


@pytest.mark.asyncio
async def test_multiple_train_predict_cycles() -> None:
    """Test multiple train and predict cycles."""
    runner = SimpleRunner()
    config = MockConfig()

    # First train-predict cycle
    data1 = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1], [3, 0]])
    model1 = await runner.on_train(config, data1)
    future1 = DataFrame(columns=["feature1"], data=[[4], [5]])
    await runner.on_predict(config, model1, None, future1)

    # Second train-predict cycle
    data2 = DataFrame(columns=["feature1", "target"], data=[[10, 1], [20, 0], [30, 1], [40, 0]])
    model2 = await runner.on_train(config, data2)
    future2 = DataFrame(columns=["feature1"], data=[[50], [60], [70]])
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
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1], [3, 0]])

    # Mock GeoJSON FeatureCollection
    geo: FeatureCollection = FeatureCollection(type="FeatureCollection", features=[])

    # Should work with geo data
    model = await runner.on_train(config, data, geo)
    future = DataFrame(columns=["feature1"], data=[[4], [5]])
    result = await runner.on_predict(config, model, None, future, geo)

    assert model is not None
    assert len(result.data) == 2


@pytest.mark.asyncio
async def test_runner_with_historic_data() -> None:
    """Test runner can handle optional historic data."""
    runner = SimpleRunner()
    config = MockConfig()
    model = {"model_id": "test_model"}

    # With historic data
    historic = DataFrame(columns=["feature1"], data=[[1], [2], [3]])
    future = DataFrame(columns=["feature1"], data=[[4], [5]])

    result = await runner.on_predict(config, model, historic, future)

    assert len(result.data) == 2
    assert "prediction" in result.columns


@pytest.mark.asyncio
async def test_default_lifecycle_hooks_do_nothing() -> None:
    """Test that default lifecycle hooks don't raise errors."""

    class MinimalRunner(BaseModelRunner[MockConfig]):
        """Runner without overriding lifecycle hooks."""

        async def on_train(
            self,
            config: MockConfig,
            data: DataFrame,
            geo: FeatureCollection | None = None,
        ) -> Any:
            """Minimal train."""
            return "model"

        async def on_predict(
            self,
            config: MockConfig,
            model: Any,
            historic: DataFrame | None,
            future: DataFrame,
            geo: FeatureCollection | None = None,
        ) -> DataFrame:
            """Minimal predict."""
            return future

    runner = MinimalRunner()

    # Default hooks should not raise
    await runner.on_init()
    await runner.on_cleanup()

    # Should work without explicitly calling hooks
    config = MockConfig()
    data = DataFrame(columns=["col1"], data=[[1], [2]])
    model = await runner.on_train(config, data)
    future = DataFrame(columns=["col1"], data=[[3], [4]])
    result = await runner.on_predict(config, model, None, future)

    assert model == "model"
    assert len(result.data) == 2
