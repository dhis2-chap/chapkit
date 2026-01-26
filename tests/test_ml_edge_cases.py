"""Edge case tests for ML module to improve coverage."""

from __future__ import annotations

import datetime
import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from geojson_pydantic import Feature, FeatureCollection, Point

from chapkit import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml import BaseModelRunner, FunctionalModelRunner, ShellModelRunner
from chapkit.ml.runner import (
    create_workspace_artifact,
    write_prediction_inputs,
    write_training_inputs,
    zip_workspace,
)


class MockConfig(BaseConfig):
    """Mock config for testing."""

    prediction_periods: int = 3
    threshold: float = 0.5


# --- Tests for GeoJSON handling (lines 83, 96) ---


def test_write_training_inputs_with_geo() -> None:
    """Test writing training inputs with GeoJSON data."""
    workspace_dir = Path(tempfile.mkdtemp(prefix="chapkit_test_"))

    try:
        config = MockConfig()
        data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1]])
        geo = FeatureCollection(
            type="FeatureCollection",
            features=[
                Feature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(0.0, 0.0)),  # type: ignore[arg-type]
                    properties={"name": "test"},
                )
            ],
        )

        write_training_inputs(workspace_dir, config, data, geo)

        # Verify geo.json was written
        geo_file = workspace_dir / "geo.json"
        assert geo_file.exists()
        geo_content = geo_file.read_text()
        assert "FeatureCollection" in geo_content
        assert "test" in geo_content

        # Verify other files also exist
        assert (workspace_dir / "config.yml").exists()
        assert (workspace_dir / "data.csv").exists()

    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


def test_write_prediction_inputs_with_geo() -> None:
    """Test writing prediction inputs with GeoJSON data."""
    workspace_dir = Path(tempfile.mkdtemp(prefix="chapkit_test_"))

    try:
        historic = DataFrame(columns=["feature1"], data=[[1], [2]])
        future = DataFrame(columns=["feature1"], data=[[3], [4]])
        geo = FeatureCollection(
            type="FeatureCollection",
            features=[
                Feature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),  # type: ignore[arg-type]
                    properties={"location": "office"},
                )
            ],
        )

        write_prediction_inputs(workspace_dir, historic, future, geo)

        # Verify geo.json was written
        geo_file = workspace_dir / "geo.json"
        assert geo_file.exists()
        geo_content = geo_file.read_text()
        assert "FeatureCollection" in geo_content
        assert "office" in geo_content

        # Verify other files also exist
        assert (workspace_dir / "historic.csv").exists()
        assert (workspace_dir / "future.csv").exists()

    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


# --- Tests for zip validation (line 116) ---


def test_zip_workspace_with_corrupted_file() -> None:
    """Test zip_workspace detects corrupted files via testzip()."""
    workspace_dir = Path(tempfile.mkdtemp(prefix="chapkit_test_"))

    try:
        # Create some files
        (workspace_dir / "file1.txt").write_text("content1")
        (workspace_dir / "file2.txt").write_text("content2")

        # Normal zip should work
        zip_bytes = zip_workspace(workspace_dir)
        assert len(zip_bytes) > 0

        # Verify it's a valid zip
        import io

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            assert zf.testzip() is None
            assert "file1.txt" in zf.namelist()
            assert "file2.txt" in zf.namelist()

    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


def test_zip_workspace_with_mock_corruption() -> None:
    """Test that zip validation error is raised for corrupted zips."""
    workspace_dir = Path(tempfile.mkdtemp(prefix="chapkit_test_"))

    try:
        (workspace_dir / "file.txt").write_text("content")

        # Mock testzip to return a bad file (simulating corruption)
        with patch.object(zipfile.ZipFile, "testzip", return_value="file.txt"):
            with pytest.raises(ValueError, match="Corrupted file in workspace zip"):
                zip_workspace(workspace_dir)

    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)


# --- Tests for FunctionalModelRunner artifact validation ---


@pytest.mark.asyncio
async def test_functional_runner_create_training_artifact_requires_workspace() -> None:
    """Test create_training_artifact raises error when workspace_dir is missing."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    now = datetime.datetime.now(datetime.UTC)

    # Pass result without workspace_dir
    with pytest.raises(ValueError, match="requires workspace dict from on_train"):
        await runner.create_training_artifact(
            training_result={"content": "model", "workspace_dir": None},
            config_id="test-config-id",
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
        )


@pytest.mark.asyncio
async def test_functional_runner_create_prediction_artifact_requires_workspace() -> None:
    """Test create_prediction_artifact raises error when workspace_dir is missing."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    now = datetime.datetime.now(datetime.UTC)
    predictions = DataFrame(columns=["x"], data=[[1]])

    # Pass result without workspace_dir
    with pytest.raises(ValueError, match="requires workspace dict from on_predict"):
        await runner.create_prediction_artifact(
            prediction_result={"content": predictions, "workspace_dir": None},
            config_id="test-config-id",
            started_at=now,
            completed_at=now,
            duration_seconds=0.5,
        )


# --- Tests for BaseModelRunner.create_prediction_artifact default ---


@pytest.mark.asyncio
async def test_base_model_runner_create_prediction_artifact() -> None:
    """Test BaseModelRunner.create_prediction_artifact default implementation."""

    class MinimalRunner(BaseModelRunner[MockConfig]):
        """Minimal runner to test base class methods."""

        async def on_train(
            self,
            config: MockConfig,
            data: DataFrame,
            geo: Any = None,
        ) -> Any:
            return "model"

        async def on_predict(
            self,
            config: MockConfig,
            model: Any,
            historic: DataFrame,
            future: DataFrame,
            geo: Any = None,
        ) -> DataFrame:
            return future

    runner = MinimalRunner()
    now = datetime.datetime.now(datetime.UTC)
    predictions = DataFrame(columns=["x", "y"], data=[[1, 2]])

    # Call base class create_prediction_artifact directly
    artifact = await runner.create_prediction_artifact(
        prediction_result=predictions,
        config_id="base-test-config",
        started_at=now,
        completed_at=now,
        duration_seconds=0.1,
    )

    assert artifact["type"] == "ml_prediction"
    assert artifact["content"] == predictions
    assert artifact["content_type"] == "application/vnd.chapkit.dataframe+json"
    assert artifact["metadata"]["config_id"] == "base-test-config"


@pytest.mark.asyncio
async def test_base_model_runner_create_training_artifact() -> None:
    """Test BaseModelRunner.create_training_artifact default implementation."""

    class MinimalRunner(BaseModelRunner[MockConfig]):
        """Minimal runner to test base class methods."""

        async def on_train(
            self,
            config: MockConfig,
            data: DataFrame,
            geo: Any = None,
        ) -> Any:
            return "model"

        async def on_predict(
            self,
            config: MockConfig,
            model: Any,
            historic: DataFrame,
            future: DataFrame,
            geo: Any = None,
        ) -> DataFrame:
            return future

    runner = MinimalRunner()
    now = datetime.datetime.now(datetime.UTC)

    artifact = await runner.create_training_artifact(
        training_result={"model_data": "test"},
        config_id="base-test-config",
        started_at=now,
        completed_at=now,
        duration_seconds=2.5,
    )

    assert artifact["type"] == "ml_training_workspace"
    assert artifact["content"] == {"model_data": "test"}
    assert artifact["content_type"] == "application/x-pickle"
    assert artifact["metadata"]["config_id"] == "base-test-config"
    assert artifact["metadata"]["duration_seconds"] == 2.5


# --- Tests for create_workspace_artifact helper ---


def test_create_workspace_artifact_with_all_fields() -> None:
    """Test create_workspace_artifact with all optional fields."""
    now = datetime.datetime.now(datetime.UTC)

    artifact = create_workspace_artifact(
        workspace_content=b"zip_content_here",
        artifact_type="ml_training_workspace",
        config_id="test-config",
        started_at=now,
        completed_at=now,
        duration_seconds=10.5,
        status="failed",
        exit_code=1,
        stdout="output",
        stderr="error output",
    )

    assert artifact["type"] == "ml_training_workspace"
    assert artifact["content"] == b"zip_content_here"
    assert artifact["content_type"] == "application/zip"
    assert artifact["content_size"] == len(b"zip_content_here")
    assert artifact["metadata"]["status"] == "failed"
    assert artifact["metadata"]["exit_code"] == 1
    assert artifact["metadata"]["stdout"] == "output"
    assert artifact["metadata"]["stderr"] == "error output"


# --- Tests for ShellModelRunner edge cases ---


@pytest.mark.asyncio
async def test_shell_runner_create_training_artifact_validation() -> None:
    """Test ShellModelRunner.create_training_artifact validates input type."""
    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'test'",
        predict_command="echo 'test' > {output_file}",
    )

    now = datetime.datetime.now(datetime.UTC)

    # Should raise ValueError for non-dict input
    with pytest.raises(ValueError, match="requires workspace dict"):
        await runner.create_training_artifact(
            training_result="not_a_dict",
            config_id="test",
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
        )


@pytest.mark.asyncio
async def test_shell_runner_create_prediction_artifact_validation() -> None:
    """Test ShellModelRunner.create_prediction_artifact validates input type."""
    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'test'",
        predict_command="echo 'test' > {output_file}",
    )

    now = datetime.datetime.now(datetime.UTC)

    # Should raise ValueError for non-dict input
    with pytest.raises(ValueError, match="requires workspace dict"):
        await runner.create_prediction_artifact(
            prediction_result="not_a_dict",
            config_id="test",
            started_at=now,
            completed_at=now,
            duration_seconds=1.0,
        )


@pytest.mark.asyncio
async def test_shell_runner_on_predict_validates_model_type() -> None:
    """Test ShellModelRunner.on_predict validates model is workspace dict."""
    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'test'",
        predict_command="echo 'test' > {output_file}",
    )

    config = MockConfig()
    historic = DataFrame(columns=["x"], data=[])
    future = DataFrame(columns=["x"], data=[[1]])

    # Should raise ValueError for non-workspace model
    with pytest.raises(ValueError, match="requires workspace artifact"):
        await runner.on_predict(config, "not_a_workspace", historic, future)


# --- Tests for FunctionalModelRunner with GeoJSON ---


@pytest.mark.asyncio
async def test_functional_runner_train_with_geo() -> None:
    """Test FunctionalModelRunner training with GeoJSON creates geo.json in workspace."""
    geo_received = []

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        geo_received.append(geo)
        return "model_with_geo"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future.add_column("prediction", [1.0] * len(future.data))

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1]])
    geo = FeatureCollection(
        type="FeatureCollection",
        features=[
            Feature(
                type="Feature",
                geometry=Point(type="Point", coordinates=(10.0, 20.0)),  # type: ignore[arg-type]
                properties={"name": "training_location"},
            )
        ],
    )

    result = await runner.on_train(config, data, geo)

    assert result["workspace_dir"] is not None
    workspace_dir = Path(result["workspace_dir"])

    # Verify geo.json was written
    geo_file = workspace_dir / "geo.json"
    assert geo_file.exists()
    assert "training_location" in geo_file.read_text()

    # Verify geo was passed to train function
    assert len(geo_received) == 1
    assert geo_received[0] is geo

    # Cleanup
    shutil.rmtree(workspace_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_functional_runner_predict_with_geo() -> None:
    """Test FunctionalModelRunner prediction with GeoJSON creates geo.json in workspace."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future.add_column("prediction", [1.0] * len(future.data))

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    config = MockConfig()
    model = "model"
    historic = DataFrame(columns=["x"], data=[[1]])
    future = DataFrame(columns=["x"], data=[[2], [3]])
    geo = FeatureCollection(
        type="FeatureCollection",
        features=[
            Feature(
                type="Feature",
                geometry=Point(type="Point", coordinates=(30.0, 40.0)),  # type: ignore[arg-type]
                properties={"name": "prediction_location"},
            )
        ],
    )

    result = await runner.on_predict(config, model, historic, future, geo)

    assert result["workspace_dir"] is not None
    workspace_dir = Path(result["workspace_dir"])

    # Verify geo.json was written
    geo_file = workspace_dir / "geo.json"
    assert geo_file.exists()
    assert "prediction_location" in geo_file.read_text()

    # Cleanup
    shutil.rmtree(workspace_dir, ignore_errors=True)


# --- Tests for corrupted pickle handling in MLManager ---


def _create_workspace_zip_with_corrupted_pickle() -> bytes:
    """Create a workspace ZIP with a corrupted model.pickle file."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add config.yml
        zf.writestr("config.yml", "threshold: 0.5\n")
        # Add corrupted model.pickle (invalid pickle data)
        zf.writestr("model.pickle", b"this is not valid pickle data")
    return zip_buffer.getvalue()


def _create_workspace_zip_with_empty_pickle() -> bytes:
    """Create a workspace ZIP with an empty model.pickle file (causes EOFError)."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.yml", "threshold: 0.5\n")
        zf.writestr("model.pickle", b"")  # Empty file causes EOFError
    return zip_buffer.getvalue()


@pytest.mark.asyncio
async def test_mlmanager_predict_with_corrupted_pickle() -> None:
    """Test MLManager._predict_task raises ValueError for corrupted pickle files."""
    from ulid import ULID

    from chapkit.ml.manager import MLManager
    from chapkit.ml.schemas import PredictRequest

    # Create mock runner (FunctionalModelRunner, not ShellModelRunner)
    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    # Create mock database and scheduler
    mock_database = MagicMock()
    mock_scheduler = MagicMock()

    manager: MLManager[MockConfig] = MLManager(
        runner=runner,
        scheduler=mock_scheduler,
        database=mock_database,
        config_schema=MockConfig,
    )

    # Create training artifact with corrupted pickle
    training_artifact_id = ULID()
    config_id = ULID()
    corrupted_workspace = _create_workspace_zip_with_corrupted_pickle()

    training_artifact = MagicMock()
    training_artifact.data = {
        "type": "ml_training_workspace",
        "metadata": {
            "status": "success",
            "config_id": str(config_id),
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
            "duration_seconds": 60.0,
        },
        "content": corrupted_workspace,
        "content_type": "application/zip",
    }

    # Mock artifact manager to return our training artifact
    mock_artifact_manager = AsyncMock()
    mock_artifact_manager.find_by_id = AsyncMock(return_value=training_artifact)

    # Mock session context manager
    mock_session = MagicMock()
    mock_database.session = MagicMock(
        return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
    )

    # Patch ArtifactManager and ArtifactRepository
    with (
        patch("chapkit.ml.manager.ArtifactRepository"),
        patch("chapkit.ml.manager.ArtifactManager", return_value=mock_artifact_manager),
    ):
        request = PredictRequest(
            artifact_id=training_artifact_id,
            historic=DataFrame(columns=["x"], data=[]),
            future=DataFrame(columns=["x"], data=[[1.0]]),
        )
        prediction_artifact_id = ULID()

        # Should raise ValueError due to corrupted pickle
        with pytest.raises(ValueError, match="corrupted or incompatible pickle file"):
            await manager._predict_task(request, prediction_artifact_id)


@pytest.mark.asyncio
async def test_mlmanager_predict_with_empty_pickle() -> None:
    """Test MLManager._predict_task raises ValueError for empty pickle files (EOFError)."""
    from ulid import ULID

    from chapkit.ml.manager import MLManager
    from chapkit.ml.schemas import PredictRequest

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    mock_database = MagicMock()
    mock_scheduler = MagicMock()

    manager: MLManager[MockConfig] = MLManager(
        runner=runner,
        scheduler=mock_scheduler,
        database=mock_database,
        config_schema=MockConfig,
    )

    training_artifact_id = ULID()
    config_id = ULID()
    empty_pickle_workspace = _create_workspace_zip_with_empty_pickle()

    training_artifact = MagicMock()
    training_artifact.data = {
        "type": "ml_training_workspace",
        "metadata": {
            "status": "success",
            "config_id": str(config_id),
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
            "duration_seconds": 60.0,
        },
        "content": empty_pickle_workspace,
        "content_type": "application/zip",
    }

    mock_artifact_manager = AsyncMock()
    mock_artifact_manager.find_by_id = AsyncMock(return_value=training_artifact)

    mock_session = MagicMock()
    mock_database.session = MagicMock(
        return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
    )

    with (
        patch("chapkit.ml.manager.ArtifactRepository"),
        patch("chapkit.ml.manager.ArtifactManager", return_value=mock_artifact_manager),
    ):
        request = PredictRequest(
            artifact_id=training_artifact_id,
            historic=DataFrame(columns=["x"], data=[]),
            future=DataFrame(columns=["x"], data=[[1.0]]),
        )
        prediction_artifact_id = ULID()

        # Should raise ValueError due to empty pickle (EOFError)
        with pytest.raises(ValueError, match="corrupted or incompatible pickle file"):
            await manager._predict_task(request, prediction_artifact_id)


@pytest.mark.asyncio
async def test_mlmanager_predict_with_failed_training_artifact() -> None:
    """Test that prediction is blocked when training artifact has status='failed'."""
    from ulid import ULID

    from chapkit.ml.manager import MLManager
    from chapkit.ml.schemas import PredictRequest

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    mock_database = MagicMock()
    mock_scheduler = MagicMock()

    manager: MLManager[MockConfig] = MLManager(
        runner=runner,
        scheduler=mock_scheduler,
        database=mock_database,
        config_schema=MockConfig,
    )

    # Create training artifact with status="failed"
    training_artifact_id = ULID()
    config_id = ULID()

    training_artifact = MagicMock()
    training_artifact.data = {
        "type": "ml_training_workspace",
        "metadata": {
            "status": "failed",  # Training failed
            "exit_code": 1,
            "config_id": str(config_id),
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
            "duration_seconds": 60.0,
            "stdout": "Training started...",
            "stderr": "Error: training failed",
        },
        "content": b"workspace zip content",
        "content_type": "application/zip",
    }

    mock_artifact_manager = AsyncMock()
    mock_artifact_manager.find_by_id = AsyncMock(return_value=training_artifact)

    mock_session = MagicMock()
    mock_database.session = MagicMock(
        return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
    )

    with (
        patch("chapkit.ml.manager.ArtifactRepository"),
        patch("chapkit.ml.manager.ArtifactManager", return_value=mock_artifact_manager),
    ):
        request = PredictRequest(
            artifact_id=training_artifact_id,
            historic=DataFrame(columns=["x"], data=[]),
            future=DataFrame(columns=["x"], data=[[1.0]]),
        )
        prediction_artifact_id = ULID()

        # Should raise ValueError because training failed
        with pytest.raises(ValueError, match="Cannot predict using failed training artifact"):
            await manager._predict_task(request, prediction_artifact_id)


# --- Tests for workspace cleanup ---


@pytest.mark.asyncio
async def test_mlmanager_train_cleans_up_workspace_on_success() -> None:
    """Test that training cleans up workspace directory after successful completion."""

    from ulid import ULID

    from chapkit.ml.manager import MLManager
    from chapkit.ml.schemas import TrainRequest

    # Track workspace directory created
    created_workspace_dir: Path | None = None

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "trained_model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    mock_database = MagicMock()
    mock_scheduler = MagicMock()

    manager: MLManager[MockConfig] = MLManager(
        runner=runner,
        scheduler=mock_scheduler,
        database=mock_database,
        config_schema=MockConfig,
    )

    config_id = ULID()
    mock_config = MagicMock()
    mock_config.data = MockConfig()

    mock_config_manager = AsyncMock()
    mock_config_manager.find_by_id = AsyncMock(return_value=mock_config)

    mock_artifact_manager = AsyncMock()
    mock_config_repo = AsyncMock()

    mock_session = MagicMock()
    mock_database.session = MagicMock(
        return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
    )

    # Capture workspace directory from runner
    original_on_train = runner.on_train

    async def capturing_on_train(config: Any, data: Any, geo: Any = None) -> Any:
        nonlocal created_workspace_dir
        result = await original_on_train(config, data, geo)
        if isinstance(result, dict) and result.get("workspace_dir"):
            created_workspace_dir = Path(result["workspace_dir"])
        return result

    with (
        patch("chapkit.ml.manager.ConfigRepository", return_value=mock_config_repo),
        patch("chapkit.ml.manager.ConfigManager", return_value=mock_config_manager),
        patch("chapkit.ml.manager.ArtifactRepository"),
        patch("chapkit.ml.manager.ArtifactManager", return_value=mock_artifact_manager),
        patch.object(runner, "on_train", side_effect=capturing_on_train),
    ):
        request = TrainRequest(
            config_id=config_id,
            data=DataFrame(columns=["x", "y"], data=[[1, 2]]),
        )
        artifact_id = ULID()

        await manager._train_task(request, artifact_id)

        # Verify workspace was created and then cleaned up
        assert created_workspace_dir is not None
        assert not created_workspace_dir.exists(), "Workspace directory should be cleaned up"


@pytest.mark.asyncio
async def test_mlmanager_train_cleans_up_workspace_on_error() -> None:
    """Test that training cleans up workspace directory even when error occurs."""
    from ulid import ULID

    from chapkit.ml.manager import MLManager
    from chapkit.ml.schemas import TrainRequest

    created_workspace_dir: Path | None = None

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "trained_model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
    )

    mock_database = MagicMock()
    mock_scheduler = MagicMock()

    manager: MLManager[MockConfig] = MLManager(
        runner=runner,
        scheduler=mock_scheduler,
        database=mock_database,
        config_schema=MockConfig,
    )

    config_id = ULID()
    mock_config = MagicMock()
    mock_config.data = MockConfig()

    mock_config_manager = AsyncMock()
    mock_config_manager.find_by_id = AsyncMock(return_value=mock_config)

    mock_session = MagicMock()
    mock_database.session = MagicMock(
        return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
    )

    original_on_train = runner.on_train

    async def capturing_on_train(config: Any, data: Any, geo: Any = None) -> Any:
        nonlocal created_workspace_dir
        result = await original_on_train(config, data, geo)
        if isinstance(result, dict) and result.get("workspace_dir"):
            created_workspace_dir = Path(result["workspace_dir"])
        return result

    async def failing_create_artifact(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # Call original to ensure workspace is created, then fail
        raise RuntimeError("Artifact creation error")

    with (
        patch("chapkit.ml.manager.ConfigRepository"),
        patch("chapkit.ml.manager.ConfigManager", return_value=mock_config_manager),
        patch.object(runner, "on_train", side_effect=capturing_on_train),
        patch.object(runner, "create_training_artifact", side_effect=failing_create_artifact),
    ):
        request = TrainRequest(
            config_id=config_id,
            data=DataFrame(columns=["x", "y"], data=[[1, 2]]),
        )
        artifact_id = ULID()

        with pytest.raises(RuntimeError, match="Artifact creation error"):
            await manager._train_task(request, artifact_id)

        # Verify workspace was cleaned up even after error
        assert created_workspace_dir is not None
        assert not created_workspace_dir.exists(), "Workspace should be cleaned up after error"
