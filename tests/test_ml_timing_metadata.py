"""Tests for ML timing metadata capture."""

import asyncio
import datetime
from collections.abc import AsyncIterator

import pandas as pd
import pytest
from geojson_pydantic import FeatureCollection
from servicekit import SqliteDatabaseBuilder
from servicekit.data import DataFrame
from ulid import ULID

from chapkit.artifact import ArtifactManager, ArtifactRepository
from chapkit.config import BaseConfig, ConfigIn, ConfigManager, ConfigRepository
from chapkit.ml import FunctionalModelRunner, MLManager, PredictRequest, TrainRequest
from chapkit.scheduler import ChapkitJobScheduler


class SimpleConfig(BaseConfig):
    """Simple test config."""

    value: int = 42


async def simple_train(
    config: BaseConfig,
    data: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> dict[str, int]:
    """Simple training function that takes measurable time."""
    await asyncio.sleep(0.1)  # Simulate training time
    return {"trained": True, "samples": len(data)}


async def simple_predict(
    config: BaseConfig,
    model: dict[str, int],
    historic: pd.DataFrame | None,
    future: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> pd.DataFrame:
    """Simple prediction function that takes measurable time."""
    await asyncio.sleep(0.05)  # Simulate prediction time
    future["sample_0"] = 1.0
    return future


class MockModel:
    """Mock model class for testing dict-wrapped models."""

    def __init__(self, value: int = 42) -> None:
        """Initialize mock model."""
        self.value = value


async def dict_wrapped_train(
    config: BaseConfig,
    data: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> dict[str, object]:
    """Training function that returns dict with 'model' key (ml_class.py pattern)."""
    await asyncio.sleep(0.1)
    return {"model": MockModel(42), "metadata": "test", "sample_count": len(data)}


@pytest.fixture
async def ml_manager() -> AsyncIterator[MLManager]:
    """Create ML manager for testing."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    scheduler = ChapkitJobScheduler()

    runner = FunctionalModelRunner(on_train=simple_train, on_predict=simple_predict)

    manager = MLManager(runner, scheduler, database, SimpleConfig)
    yield manager


@pytest.fixture
async def setup_data(ml_manager: MLManager) -> tuple[ULID, pd.DataFrame, pd.DataFrame]:
    """Set up test data for ML operations."""
    # Create config
    async with ml_manager.database.session() as session:
        config_repo = ConfigRepository(session)
        config_manager = ConfigManager[SimpleConfig](config_repo, SimpleConfig)
        config = await config_manager.save(ConfigIn(name="test", data=SimpleConfig(value=42)))
        await config_repo.commit()

    # Create training data
    train_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [7, 8, 9]})

    # Create prediction data
    predict_df = pd.DataFrame({"feature1": [10, 11], "feature2": [12, 13]})

    return config.id, train_df, predict_df


async def test_training_timing_metadata_captured(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that training timing metadata is captured correctly."""
    config_id, train_df, _ = setup_data

    # Submit training job
    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )

    response = await ml_manager.execute_train(train_request)

    # Wait for job to complete
    await asyncio.sleep(0.5)

    # Retrieve trained model artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    assert artifact.data["ml_type"] == "trained_model"

    # Verify timing metadata exists
    assert "started_at" in artifact.data
    assert "completed_at" in artifact.data
    assert "duration_seconds" in artifact.data

    # Verify timing values are reasonable
    started_at = datetime.datetime.fromisoformat(artifact.data["started_at"])
    completed_at = datetime.datetime.fromisoformat(artifact.data["completed_at"])
    duration = artifact.data["duration_seconds"]

    assert isinstance(started_at, datetime.datetime)
    assert isinstance(completed_at, datetime.datetime)
    assert isinstance(duration, (int, float))

    # Verify timing makes sense (completed after started)
    assert completed_at > started_at

    # Verify duration roughly matches sleep time (0.1s + overhead)
    assert 0.05 < duration < 1.0

    # Verify duration matches calculated difference
    calculated_duration = (completed_at - started_at).total_seconds()
    assert abs(duration - calculated_duration) < 0.01


async def test_prediction_timing_metadata_captured(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that prediction timing metadata is captured correctly."""
    config_id, train_df, predict_df = setup_data

    # Train model first
    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    train_response = await ml_manager.execute_train(train_request)

    # Wait for training to complete
    await asyncio.sleep(0.5)

    # Submit prediction job
    predict_request = PredictRequest(
        model_artifact_id=ULID.from_str(train_response.model_artifact_id),
        historic=DataFrame.from_pandas(pd.DataFrame({"feature1": [], "feature2": []})),
        future=DataFrame.from_pandas(predict_df),
    )
    predict_response = await ml_manager.execute_predict(predict_request)

    # Wait for prediction to complete
    await asyncio.sleep(0.5)

    # Retrieve prediction artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(predict_response.prediction_artifact_id))

    assert artifact is not None
    assert artifact.data["ml_type"] == "prediction"

    # Verify timing metadata exists
    assert "started_at" in artifact.data
    assert "completed_at" in artifact.data
    assert "duration_seconds" in artifact.data

    # Verify timing values are reasonable
    started_at = datetime.datetime.fromisoformat(artifact.data["started_at"])
    completed_at = datetime.datetime.fromisoformat(artifact.data["completed_at"])
    duration = artifact.data["duration_seconds"]

    assert isinstance(started_at, datetime.datetime)
    assert isinstance(completed_at, datetime.datetime)
    assert isinstance(duration, (int, float))

    # Verify timing makes sense
    assert completed_at > started_at

    # Verify duration roughly matches sleep time (0.05s + overhead)
    assert 0.01 < duration < 1.0

    # Verify duration matches calculated difference
    calculated_duration = (completed_at - started_at).total_seconds()
    assert abs(duration - calculated_duration) < 0.01


async def test_timing_metadata_iso_format(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that timestamps are in ISO format."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    # Verify ISO format can be parsed
    started_str = artifact.data["started_at"]
    completed_str = artifact.data["completed_at"]

    # These should not raise exceptions
    started = datetime.datetime.fromisoformat(started_str)
    completed = datetime.datetime.fromisoformat(completed_str)

    # Verify timezone info exists (UTC)
    assert started.tzinfo is not None
    assert completed.tzinfo is not None


async def test_timing_duration_rounded_to_two_decimals(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that duration is rounded to 2 decimal places."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    duration = artifact.data["duration_seconds"]

    # Verify precision (should have at most 2 decimal places)
    duration_str = str(duration)
    if "." in duration_str:
        decimal_places = len(duration_str.split(".")[1])
        assert decimal_places <= 2


async def test_original_metadata_preserved(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that original metadata fields are still present."""
    config_id, train_df, predict_df = setup_data

    # Train model
    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    train_response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    # Check training artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        train_artifact = await artifact_manager.find_by_id(ULID.from_str(train_response.model_artifact_id))

    # Original fields should still exist
    assert train_artifact is not None
    assert train_artifact.data["ml_type"] == "trained_model"
    assert train_artifact.data["config_id"] == str(config_id)
    assert "model" in train_artifact.data

    # Predict
    predict_request = PredictRequest(
        model_artifact_id=ULID.from_str(train_response.model_artifact_id),
        historic=DataFrame.from_pandas(pd.DataFrame({"feature1": [], "feature2": []})),
        future=DataFrame.from_pandas(predict_df),
    )
    predict_response = await ml_manager.execute_predict(predict_request)
    await asyncio.sleep(0.5)

    # Check prediction artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        predict_artifact = await artifact_manager.find_by_id(ULID.from_str(predict_response.prediction_artifact_id))

    # Original fields should still exist
    assert predict_artifact is not None
    assert predict_artifact.data["ml_type"] == "prediction"
    assert predict_artifact.data["model_artifact_id"] == str(train_response.model_artifact_id)
    assert predict_artifact.data["config_id"] == str(config_id)
    assert "predictions" in predict_artifact.data


async def test_model_type_captured_in_training_artifact(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that model_type field is captured in training artifact."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    assert "model_type" in artifact.data

    # Verify it's a string with module and class name
    model_type = artifact.data["model_type"]
    assert isinstance(model_type, str)
    assert "." in model_type  # Should have module.class format
    assert "dict" in model_type  # simple_train returns a dict


async def test_model_size_bytes_captured_in_training_artifact(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that model_size_bytes field is captured in training artifact."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    assert "model_size_bytes" in artifact.data

    # Verify it's a positive integer
    model_size_bytes = artifact.data["model_size_bytes"]
    assert isinstance(model_size_bytes, int)
    assert model_size_bytes > 0


async def test_model_metrics_are_optional_fields(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that model_type and model_size_bytes are optional and can handle None."""
    from chapkit.ml.schemas import TrainedModelArtifactData

    # Create artifact data with None values for optional fields
    artifact_data = TrainedModelArtifactData(
        ml_type="trained_model",
        config_id="01K72P5N5KCRM6MD3BRE4P0001",
        model={"test": "model"},
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        model_type=None,
        model_size_bytes=None,
    )

    # Verify schema accepts None values
    assert artifact_data.model_type is None
    assert artifact_data.model_size_bytes is None

    # Verify schema can omit these fields entirely
    artifact_data_minimal = TrainedModelArtifactData(
        ml_type="trained_model",
        config_id="01K72P5N5KCRM6MD3BRE4P0001",
        model={"test": "model"},
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )

    assert artifact_data_minimal.model_type is None
    assert artifact_data_minimal.model_size_bytes is None


async def test_model_type_extracts_from_dict_wrapped_models(
    setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame],
) -> None:
    """Test that model_type extracts inner model type from dict-wrapped models."""
    # Create manager with dict-wrapped training function
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    scheduler = ChapkitJobScheduler()
    runner = FunctionalModelRunner(on_train=dict_wrapped_train, on_predict=simple_predict)
    manager = MLManager(runner, scheduler, database, SimpleConfig)

    config_id, train_df, _ = setup_data

    # Create config in the new manager's database
    async with manager.database.session() as session:
        config_repo = ConfigRepository(session)
        config_manager = ConfigManager[SimpleConfig](config_repo, SimpleConfig)
        await config_manager.save(ConfigIn(id=config_id, name="test", data=SimpleConfig(value=42)))
        await config_repo.commit()

    # Train model
    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    # Retrieve artifact
    async with manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    assert "model_type" in artifact.data

    # Verify model_type extracted the inner MockModel, not builtins.dict
    model_type = artifact.data["model_type"]
    assert "MockModel" in model_type
    assert "dict" not in model_type


async def test_model_size_bytes_varies_with_complexity() -> None:
    """Test that model_size_bytes varies with model complexity."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    # Create config
    async with database.session() as session:
        config_repo = ConfigRepository(session)
        config_manager = ConfigManager[SimpleConfig](config_repo, SimpleConfig)
        config = await config_manager.save(ConfigIn(name="test", data=SimpleConfig(value=42)))
        await config_repo.commit()

    train_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [7, 8, 9]})

    # Train simple model (small dict)
    scheduler1 = ChapkitJobScheduler()
    runner1 = FunctionalModelRunner(on_train=simple_train, on_predict=simple_predict)
    manager1 = MLManager(runner1, scheduler1, database, SimpleConfig)

    train_request1 = TrainRequest(
        config_id=config.id,
        data=DataFrame.from_pandas(train_df),
    )
    response1 = await manager1.execute_train(train_request1)
    await asyncio.sleep(0.5)

    # Train complex model (dict with nested model object)
    scheduler2 = ChapkitJobScheduler()
    runner2 = FunctionalModelRunner(on_train=dict_wrapped_train, on_predict=simple_predict)
    manager2 = MLManager(runner2, scheduler2, database, SimpleConfig)

    train_request2 = TrainRequest(
        config_id=config.id,
        data=DataFrame.from_pandas(train_df),
    )
    response2 = await manager2.execute_train(train_request2)
    await asyncio.sleep(0.5)

    # Retrieve both artifacts
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact1 = await artifact_manager.find_by_id(ULID.from_str(response1.model_artifact_id))
        artifact2 = await artifact_manager.find_by_id(ULID.from_str(response2.model_artifact_id))

    assert artifact1 is not None
    assert artifact2 is not None

    size1 = artifact1.data["model_size_bytes"]
    size2 = artifact2.data["model_size_bytes"]

    # Both should be positive integers
    assert isinstance(size1, int) and size1 > 0
    assert isinstance(size2, int) and size2 > 0

    # Sizes should be different (we can't guarantee which is larger, just that they differ)
    # This verifies that the size calculation is meaningful
    assert size1 != size2


async def test_model_metrics_present_alongside_timing_metadata(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that model metrics and timing metadata coexist in artifact."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(0.5)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.model_artifact_id))

    assert artifact is not None
    data = artifact.data

    # Verify all timing metadata fields exist
    assert "started_at" in data
    assert "completed_at" in data
    assert "duration_seconds" in data

    # Verify all model metric fields exist
    assert "model_type" in data
    assert "model_size_bytes" in data

    # Verify all core fields exist
    assert "ml_type" in data
    assert "config_id" in data
    assert "model" in data

    # Verify types are correct
    assert isinstance(data["model_type"], str)
    assert isinstance(data["model_size_bytes"], int)
    assert isinstance(data["duration_seconds"], (int, float))
