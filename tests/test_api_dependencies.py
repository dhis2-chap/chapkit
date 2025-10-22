"""Tests for API dependency injection functions."""

import pytest
from servicekit import SqliteDatabaseBuilder

from chapkit.api.dependencies import get_artifact_manager, get_config_manager, get_ml_manager
from chapkit.artifact import ArtifactManager
from chapkit.config import BaseConfig, ConfigManager


async def test_get_config_manager() -> None:
    """Test get_config_manager dependency injection."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        manager = await get_config_manager(session)
        assert isinstance(manager, ConfigManager)

    await db.dispose()


async def test_get_artifact_manager() -> None:
    """Test get_artifact_manager dependency injection."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        manager = await get_artifact_manager(session)
        assert isinstance(manager, ArtifactManager)

    await db.dispose()


async def test_get_ml_manager_raises_without_override() -> None:
    """Test get_ml_manager raises RuntimeError when not configured."""
    with pytest.raises(
        RuntimeError, match="ML manager dependency not configured. Use ServiceBuilder.with_ml\\(\\) to enable ML operations."
    ):
        await get_ml_manager()
