"""Tests for ServiceBuilder validation."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from geojson_pydantic import FeatureCollection
from pydantic import Field
from servicekit.api.service_builder import ServiceInfo

from chapkit import ArtifactHierarchy, BaseConfig, get_alembic_dir
from chapkit.api import MLServiceBuilder, ServiceBuilder
from chapkit.data import DataFrame
from chapkit.ml import BaseModelRunner


class DummyConfig(BaseConfig):
    """Dummy config for testing."""

    prediction_periods: int = 3
    test_value: str = Field(default="test")


class DummyRunner(BaseModelRunner[DummyConfig]):
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
    builder = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    with pytest.raises(ValueError, match="ML operations require config"):
        builder.with_artifacts(hierarchy=hierarchy).with_ml(runner=runner).build()


def test_ml_without_artifacts_raises_error() -> None:
    """Test that ML without artifacts raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    runner = DummyRunner()

    with pytest.raises(ValueError, match="ML operations require artifacts"):
        builder.with_config(DummyConfig).with_ml(runner=runner).build()


def test_ml_without_jobs_raises_error() -> None:
    """Test that ML without job scheduler raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    with pytest.raises(ValueError, match="ML operations require job scheduler"):
        builder.with_config(DummyConfig).with_artifacts(hierarchy=hierarchy).with_ml(runner=runner).build()


def test_artifacts_config_linking_without_config_raises_error() -> None:
    """Test that artifact config-linking without config raises ValueError."""
    builder = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")

    with pytest.raises(ValueError, match="Artifact config-linking requires a config schema"):
        builder.with_artifacts(hierarchy=hierarchy, enable_config_linking=True).build()


def test_valid_ml_service_builds_successfully() -> None:
    """Test that a properly configured ML service builds without errors."""
    builder = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    # This should build successfully with all dependencies
    app = (
        builder.with_config(DummyConfig).with_artifacts(hierarchy=hierarchy).with_jobs().with_ml(runner=runner).build()
    )

    assert app is not None


async def test_ml_with_wrong_scheduler_type_raises_error() -> None:
    """Test that ML operations require ChapkitScheduler, not just any scheduler."""
    from servicekit import InMemoryScheduler, SqliteDatabaseBuilder

    builder = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    hierarchy = ArtifactHierarchy(name="test")
    runner = DummyRunner()

    # Build ML service to get the dependency
    builder.with_config(DummyConfig).with_artifacts(hierarchy=hierarchy).with_jobs().with_ml(runner=runner)

    # Get the ML manager dependency function
    ml_dependency = builder._build_ml_dependency()

    # Resolve the dependency with a non-ChapkitScheduler (use plain InMemoryScheduler)
    wrong_scheduler = InMemoryScheduler()
    database = SqliteDatabaseBuilder.in_memory().build()

    # Try to get ML manager - should fail because scheduler is wrong type
    with pytest.raises(RuntimeError, match="Scheduler must be ChapkitScheduler"):
        await ml_dependency(scheduler_base=wrong_scheduler, database=database)


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
            info=ServiceInfo(id="test", display_name="Test"),
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
        info=ServiceInfo(id="test", display_name="Test"),
        config_schema=DummyConfig,
        hierarchy=hierarchy,
        runner=runner,
        database_url="sqlite+aiosqlite:///:memory:",
    ).build()

    assert app is not None


async def test_ml_dependency_requires_runner_and_config_schema() -> None:
    """The ML dependency refuses to build a manager when the builder was not fully configured."""
    from servicekit import InMemoryScheduler, SqliteDatabaseBuilder

    database = SqliteDatabaseBuilder.in_memory().build()
    scheduler = InMemoryScheduler()

    without_runner = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test"))
    with pytest.raises(RuntimeError, match="ML runner not configured"):
        await without_runner._build_ml_dependency()(scheduler_base=scheduler, database=database)

    without_config = ServiceBuilder(info=ServiceInfo(id="test", display_name="Test")).with_ml(runner=DummyRunner())
    with pytest.raises(RuntimeError, match="Config schema not configured"):
        await without_config._build_ml_dependency()(scheduler_base=scheduler, database=database)
