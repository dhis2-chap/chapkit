"""Tests for ServiceBuilder validation."""

from typing import Any

import pandas as pd
import pytest
from geojson_pydantic import FeatureCollection
from pydantic import Field
from servicekit.api.service_builder import ServiceInfo

from chapkit import ArtifactHierarchy, BaseConfig
from chapkit.api import ServiceBuilder
from chapkit.ml import ModelRunnerProtocol


class DummyConfig(BaseConfig):
    """Dummy config for testing."""

    test_value: str = Field(default="test")


class DummyRunner(ModelRunnerProtocol):
    """Dummy ML runner for testing."""

    async def on_train(
        self,
        config: BaseConfig,
        data: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model."""
        return {"status": "trained"}

    async def on_predict(
        self,
        config: BaseConfig,
        model: Any,
        historic: pd.DataFrame | None,
        future: pd.DataFrame,
        geo: FeatureCollection | None = None,
    ) -> pd.DataFrame:
        """Make predictions."""
        return pd.DataFrame({"predictions": []})


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
