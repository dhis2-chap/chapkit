"""Edge case tests for ML module to improve coverage."""

from __future__ import annotations

import datetime
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


# --- Tests for FunctionalModelRunner with workspace disabled ---


@pytest.mark.asyncio
async def test_functional_runner_workspace_disabled_train() -> None:
    """Test FunctionalModelRunner with enable_workspace=False for training."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "simple_model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future.add_column("prediction", [1.0] * len(future.data))

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
        enable_workspace=False,
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1]])

    result = await runner.on_train(config, data)

    # With workspace disabled, result should still be dict but with None workspace_dir
    assert isinstance(result, dict)
    assert result["content"] == "simple_model"
    assert result["workspace_dir"] is None


@pytest.mark.asyncio
async def test_functional_runner_workspace_disabled_predict() -> None:
    """Test FunctionalModelRunner with enable_workspace=False for prediction."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "simple_model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future.add_column("prediction", [1.0] * len(future.data))

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
        enable_workspace=False,
    )

    config = MockConfig()
    model = "simple_model"
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[3], [4]])

    result = await runner.on_predict(config, model, historic, future)

    # With workspace disabled, result should still be dict but with None workspace_dir
    assert isinstance(result, dict)
    assert result["workspace_dir"] is None
    assert "prediction" in result["content"].columns


@pytest.mark.asyncio
async def test_functional_runner_create_training_artifact_no_workspace() -> None:
    """Test create_training_artifact falls back to pickle when workspace disabled."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "simple_model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
        enable_workspace=False,
    )

    now = datetime.datetime.now(datetime.UTC)
    training_result = {"content": "my_model", "workspace_dir": None}

    artifact = await runner.create_training_artifact(
        training_result=training_result,
        config_id="test-config-id",
        started_at=now,
        completed_at=now,
        duration_seconds=1.5,
    )

    assert artifact["type"] == "ml_training_workspace"
    assert artifact["content"] == "my_model"
    assert artifact["content_type"] == "application/x-pickle"


@pytest.mark.asyncio
async def test_functional_runner_create_prediction_artifact_no_workspace() -> None:
    """Test create_prediction_artifact falls back to DataFrame when workspace disabled."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
        enable_workspace=False,
    )

    now = datetime.datetime.now(datetime.UTC)
    predictions = DataFrame(columns=["feature1", "prediction"], data=[[1, 0.5]])
    prediction_result = {"content": predictions, "workspace_dir": None}

    artifact = await runner.create_prediction_artifact(
        prediction_result=prediction_result,
        config_id="test-config-id",
        started_at=now,
        completed_at=now,
        duration_seconds=0.5,
    )

    assert artifact["type"] == "ml_prediction"
    assert artifact["content"] == predictions
    assert artifact["content_type"] == "application/vnd.chapkit.dataframe+json"


# --- Tests for legacy fallback in create_training/prediction_artifact ---


@pytest.mark.asyncio
async def test_functional_runner_create_training_artifact_legacy_format() -> None:
    """Test create_training_artifact handles legacy non-dict training result."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "legacy_model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
        enable_workspace=False,
    )

    now = datetime.datetime.now(datetime.UTC)

    # Pass a non-dict legacy result
    artifact = await runner.create_training_artifact(
        training_result="legacy_model_object",
        config_id="test-config-id",
        started_at=now,
        completed_at=now,
        duration_seconds=1.0,
    )

    # Should fall back to base class implementation
    assert artifact["type"] == "ml_training_workspace"
    assert artifact["content"] == "legacy_model_object"
    assert artifact["content_type"] == "application/x-pickle"


@pytest.mark.asyncio
async def test_functional_runner_create_prediction_artifact_legacy_format() -> None:
    """Test create_prediction_artifact handles legacy non-dict prediction result."""

    async def train_fn(config: MockConfig, data: DataFrame, geo: Any = None) -> str:
        return "model"

    async def predict_fn(
        config: MockConfig, model: Any, historic: DataFrame, future: DataFrame, geo: Any = None
    ) -> DataFrame:
        return future

    runner: FunctionalModelRunner[MockConfig] = FunctionalModelRunner(
        on_train=train_fn,
        on_predict=predict_fn,
        enable_workspace=False,
    )

    now = datetime.datetime.now(datetime.UTC)
    legacy_predictions = DataFrame(columns=["col1", "pred"], data=[[1, 0.9]])

    # Pass a non-dict legacy result (just DataFrame)
    artifact = await runner.create_prediction_artifact(
        prediction_result=legacy_predictions,
        config_id="test-config-id",
        started_at=now,
        completed_at=now,
        duration_seconds=0.3,
    )

    # Should fall back to base class implementation
    assert artifact["type"] == "ml_prediction"
    assert artifact["content"] == legacy_predictions
    assert artifact["content_type"] == "application/vnd.chapkit.dataframe+json"


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
        enable_workspace=True,
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
        enable_workspace=True,
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
