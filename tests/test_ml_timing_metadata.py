"""Tests for ML timing metadata capture."""

import asyncio
import datetime
from collections.abc import AsyncIterator

import pandas as pd
import pytest
from geojson_pydantic import FeatureCollection
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit.artifact import ArtifactManager, ArtifactRepository
from chapkit.config import BaseConfig, ConfigIn, ConfigManager, ConfigRepository
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner, MLManager, PredictRequest, TrainRequest
from chapkit.scheduler import InMemoryChapkitScheduler


class SimpleConfig(BaseConfig):
    """Simple test config."""

    value: int = 42


async def simple_train(
    config: BaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> dict[str, int]:
    """Simple training function that takes measurable time."""
    await asyncio.sleep(0.1)  # Simulate training time
    # Convert to pandas for processing (users would use their preferred library)
    df = data.to_pandas()
    return {"trained": True, "samples": len(df)}


async def simple_predict(
    config: BaseConfig,
    model: dict[str, int],
    historic: DataFrame | None,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Simple prediction function that takes measurable time."""
    await asyncio.sleep(0.05)  # Simulate prediction time
    # Convert to pandas for processing
    future_df = future.to_pandas()
    future_df["sample_0"] = 1.0
    return DataFrame.from_pandas(future_df)


class MockModel:
    """Mock model class for testing dict-wrapped models."""

    def __init__(self, value: int = 42) -> None:
        """Initialize mock model."""
        self.value = value


async def dict_wrapped_train(
    config: BaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> dict[str, object]:
    """Training function that returns dict with 'model' key (ml_class.py pattern)."""
    await asyncio.sleep(0.1)
    df = data.to_pandas()
    return {"model": MockModel(42), "metadata": "test", "sample_count": len(df)}


@pytest.fixture
async def ml_manager() -> AsyncIterator[MLManager]:
    """Create ML manager for testing."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    scheduler = InMemoryChapkitScheduler()

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

    # Wait for job to complete (workspace creation takes longer)
    await asyncio.sleep(2.0)

    # Retrieve trained model artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.artifact_id))

    assert artifact is not None
    assert artifact.data["type"] == "ml_training_workspace"

    # Verify timing metadata exists
    assert "started_at" in artifact.data["metadata"]
    assert "completed_at" in artifact.data["metadata"]
    assert "duration_seconds" in artifact.data["metadata"]

    # Verify timing values are reasonable
    started_at = datetime.datetime.fromisoformat(artifact.data["metadata"]["started_at"])
    completed_at = datetime.datetime.fromisoformat(artifact.data["metadata"]["completed_at"])
    duration = artifact.data["metadata"]["duration_seconds"]

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

    # Wait for training to complete (workspace creation takes longer)
    await asyncio.sleep(2.0)

    # Submit prediction job
    predict_request = PredictRequest(
        artifact_id=ULID.from_str(train_response.artifact_id),
        historic=DataFrame.from_pandas(pd.DataFrame({"feature1": [], "feature2": []})),
        future=DataFrame.from_pandas(predict_df),
    )
    predict_response = await ml_manager.execute_predict(predict_request)

    # Wait for prediction to complete (workspace creation takes longer)
    await asyncio.sleep(2.0)

    # Retrieve prediction artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(predict_response.artifact_id))

    assert artifact is not None
    assert artifact.data["type"] == "ml_prediction"

    # Verify timing metadata exists
    assert "started_at" in artifact.data["metadata"]
    assert "completed_at" in artifact.data["metadata"]
    assert "duration_seconds" in artifact.data["metadata"]

    # Verify timing values are reasonable
    started_at = datetime.datetime.fromisoformat(artifact.data["metadata"]["started_at"])
    completed_at = datetime.datetime.fromisoformat(artifact.data["metadata"]["completed_at"])
    duration = artifact.data["metadata"]["duration_seconds"]

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
    await asyncio.sleep(2.0)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.artifact_id))

    assert artifact is not None
    # Verify ISO format can be parsed
    started_str = artifact.data["metadata"]["started_at"]
    completed_str = artifact.data["metadata"]["completed_at"]

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
    await asyncio.sleep(2.0)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.artifact_id))

    assert artifact is not None
    duration = artifact.data["metadata"]["duration_seconds"]

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
    await asyncio.sleep(2.0)

    # Check training artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        train_artifact = await artifact_manager.find_by_id(ULID.from_str(train_response.artifact_id))

    # Original fields should still exist
    assert train_artifact is not None
    assert train_artifact.data["type"] == "ml_training_workspace"
    assert train_artifact.data["metadata"]["config_id"] == str(config_id)
    assert "content" in train_artifact.data

    # Predict
    predict_request = PredictRequest(
        artifact_id=ULID.from_str(train_response.artifact_id),
        historic=DataFrame.from_pandas(pd.DataFrame({"feature1": [], "feature2": []})),
        future=DataFrame.from_pandas(predict_df),
    )
    predict_response = await ml_manager.execute_predict(predict_request)
    await asyncio.sleep(2.0)

    # Check prediction artifact
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        predict_artifact = await artifact_manager.find_by_id(ULID.from_str(predict_response.artifact_id))

    # Original fields should still exist
    assert predict_artifact is not None
    assert predict_artifact.data["type"] == "ml_prediction"
    # artifact_id is now in parent_id, not in data
    assert predict_artifact.parent_id == ULID.from_str(train_response.artifact_id)
    assert predict_artifact.data["metadata"]["config_id"] == str(config_id)
    assert "content" in predict_artifact.data


async def test_content_type_set_in_training_artifact(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that content_type field is set in training artifact."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(2.0)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.artifact_id))

    assert artifact is not None
    assert "content_type" in artifact.data
    # Workspace is always enabled, so content_type is always application/zip
    assert artifact.data["content_type"] == "application/zip"


async def test_typed_metadata_validation(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that artifact data follows typed schema structure."""
    from chapkit.artifact.schemas import MLMetadata, MLTrainingWorkspaceArtifactData

    # Create artifact data with typed structure
    metadata = MLMetadata(
        status="success",
        config_id="01K72P5N5KCRM6MD3BRE4P0001",
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:00:01+00:00",
        duration_seconds=1.0,
    )

    # Create typed artifact data
    artifact_data = MLTrainingWorkspaceArtifactData(
        type="ml_training_workspace",
        metadata=metadata,
        content={"test": "model"},
        content_type="application/x-pickle",
        content_size=None,
    )

    # Verify typed structure
    assert artifact_data.type == "ml_training_workspace"
    assert artifact_data.metadata.status == "success"
    assert artifact_data.metadata.config_id == "01K72P5N5KCRM6MD3BRE4P0001"
    assert artifact_data.content == {"test": "model"}
    assert artifact_data.content_type == "application/x-pickle"
    assert artifact_data.content_size is None


async def test_typed_structure_present_in_artifact(
    ml_manager: MLManager, setup_data: tuple[ULID, pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that artifact follows typed data structure."""
    config_id, train_df, _ = setup_data

    train_request = TrainRequest(
        config_id=config_id,
        data=DataFrame.from_pandas(train_df),
    )
    response = await ml_manager.execute_train(train_request)
    await asyncio.sleep(2.0)

    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        artifact = await artifact_manager.find_by_id(ULID.from_str(response.artifact_id))

    assert artifact is not None
    data = artifact.data

    # Verify typed structure exists
    assert "type" in data
    assert "metadata" in data
    assert "content" in data
    assert "content_type" in data

    # Verify type is correct
    assert data["type"] == "ml_training_workspace"

    # Verify metadata fields exist
    assert "started_at" in data["metadata"]
    assert "completed_at" in data["metadata"]
    assert "duration_seconds" in data["metadata"]
    assert "config_id" in data["metadata"]
    assert "status" in data["metadata"]

    # Verify types are correct
    assert isinstance(data["metadata"]["duration_seconds"], (int, float))
    assert data["metadata"]["status"] == "success"


async def test_predict_with_wrong_artifact_type_raises_error(ml_manager: MLManager, setup_data: tuple) -> None:
    """Test that predicting with a non-training artifact raises ValueError."""
    from chapkit.artifact import ArtifactIn

    _, _, predict_df = setup_data

    # Create an artifact with wrong ml_type (not a training artifact)
    wrong_artifact_id = ULID()
    async with ml_manager.database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)

        # Create artifact with ml_type="ml_prediction" instead of "ml_training_workspace"
        await artifact_manager.save(
            ArtifactIn(
                id=wrong_artifact_id,
                data={"ml_type": "ml_prediction", "some_data": "test"},
                parent_id=None,
                level=0,
            )
        )

    # Try to predict using this wrong artifact
    predict_request = PredictRequest(
        artifact_id=wrong_artifact_id,
        historic=DataFrame.from_pandas(pd.DataFrame({"feature1": [], "feature2": []})),
        future=DataFrame.from_pandas(predict_df),
    )

    response = await ml_manager.execute_predict(predict_request)
    job_id = ULID.from_str(response.job_id)

    # Wait for job to fail
    await asyncio.sleep(2.0)

    # Check that job failed with the right error
    record = await ml_manager.scheduler.get_record(job_id)
    assert record.status == "failed"
    assert "is not a training artifact" in str(record.error)
