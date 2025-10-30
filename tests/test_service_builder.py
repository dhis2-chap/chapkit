"""Tests for ServiceBuilder validation."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from geojson_pydantic import FeatureCollection
from pydantic import Field
from servicekit.api.service_builder import ServiceInfo
from servicekit.data import DataFrame

from chapkit import ArtifactHierarchy, BaseConfig, get_alembic_dir
from chapkit.api import MLServiceBuilder, ServiceBuilder
from chapkit.ml import ModelRunnerProtocol


class DummyConfig(BaseConfig):
    """Dummy config for testing."""

    test_value: str = Field(default="test")


class DummyRunner(ModelRunnerProtocol[DummyConfig]):
    """Dummy ML runner for testing."""

    async def on_train(
        self,
        config: DummyConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model."""
        return {"status": "trained"}

    async def on_predict(
        self,
        config: DummyConfig,
        model: Any,
        historic: DataFrame | None,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions."""
        return DataFrame(columns=["predictions"], data=[])


def test_ml_without_config_raises_error() -> None:
    """Test that ML without config raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    with pytest.raises(ValueError, match="ML operations require config"):
        builder.with_artifacts(hierarchy=hierarchy).with_ml(runner=runner).build()


def test_ml_without_artifacts_raises_error() -> None:
    """Test that ML without artifacts raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(display_name="Test"))
    runner = DummyRunner()

    with pytest.raises(ValueError, match="ML operations require artifacts"):
        builder.with_config(DummyConfig).with_ml(runner=runner).build()


def test_ml_without_jobs_raises_error() -> None:
    """Test that ML without job scheduler raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    with pytest.raises(ValueError, match="ML operations require job scheduler"):
        builder.with_config(DummyConfig).with_artifacts(hierarchy=hierarchy).with_ml(runner=runner).build()


def test_artifacts_config_linking_without_config_raises_error() -> None:
    """Test that artifact config-linking without config raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")

    with pytest.raises(ValueError, match="Artifact config-linking requires a config schema"):
        builder.with_artifacts(hierarchy=hierarchy, enable_config_linking=True).build()


def test_valid_ml_service_builds_successfully() -> None:
    """Test that a properly configured ML service builds without errors."""
    builder = ServiceBuilder(info=ServiceInfo(display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    # This should build successfully with all dependencies
    app = (
        builder.with_config(DummyConfig).with_artifacts(hierarchy=hierarchy).with_jobs().with_ml(runner=runner).build()
    )

    assert app is not None


def test_get_alembic_dir_returns_valid_path() -> None:
    """Test that get_alembic_dir returns a valid path to bundled migrations."""
    alembic_dir = get_alembic_dir()

    assert isinstance(alembic_dir, Path)
    assert alembic_dir.exists()
    assert alembic_dir.is_dir()
    assert (alembic_dir / "env.py").exists()
    assert (alembic_dir / "versions").exists()


def test_ml_service_builder_with_file_database() -> None:
    """Test MLServiceBuilder with file-based database configures migrations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        hierarchy = ArtifactHierarchy(name="test")
        runner = DummyRunner()

        # MLServiceBuilder should auto-configure database with migrations
        app = MLServiceBuilder(
            info=ServiceInfo(display_name="Test"),
            config_schema=DummyConfig,
            hierarchy=hierarchy,
            runner=runner,
            database_url=f"sqlite+aiosqlite:///{db_path}",
        ).build()

        assert app is not None


def test_ml_service_builder_with_memory_database() -> None:
    """Test MLServiceBuilder with in-memory database doesn't use migrations."""
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    # MLServiceBuilder should use in-memory database without migrations
    app = MLServiceBuilder(
        info=ServiceInfo(display_name="Test"),
        config_schema=DummyConfig,
        hierarchy=hierarchy,
        runner=runner,
        database_url="sqlite+aiosqlite:///:memory:",
    ).build()

    assert app is not None
