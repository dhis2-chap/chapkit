"""Tests for API dependency injection functions."""

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit.api.dependencies import get_artifact_manager, get_config_manager, get_ml_manager, get_task_manager
from chapkit.artifact import ArtifactManager
from chapkit.config import ConfigManager
from chapkit.task import TaskManager


@pytest.mark.asyncio
async def test_get_config_manager() -> None:
    """Test get_config_manager returns a properly configured ConfigManager."""
    mock_session = Mock(spec=AsyncSession)

    manager = await get_config_manager(mock_session)

    assert isinstance(manager, ConfigManager)
    assert manager.repository is not None


@pytest.mark.asyncio
async def test_get_artifact_manager() -> None:
    """Test get_artifact_manager returns a properly configured ArtifactManager."""
    mock_session = Mock(spec=AsyncSession)

    manager = await get_artifact_manager(mock_session)

    assert isinstance(manager, ArtifactManager)
    assert manager.repository is not None
    assert manager.config_repo is not None


@pytest.mark.asyncio
async def test_get_task_manager_with_scheduler_and_database() -> None:
    """Test get_task_manager returns TaskManager with scheduler and database."""
    mock_session = Mock(spec=AsyncSession)
    mock_artifact_manager = Mock(spec=ArtifactManager)
    mock_scheduler = Mock()
    mock_database = Mock()

    with (
        patch("chapkit.api.dependencies.get_scheduler", return_value=mock_scheduler),
        patch("chapkit.api.dependencies.get_database", return_value=mock_database),
    ):
        manager = await get_task_manager(mock_session, mock_artifact_manager)

        assert isinstance(manager, TaskManager)
        assert manager.repository is not None
        assert manager.scheduler is mock_scheduler
        assert manager.database is mock_database
        assert manager.artifact_manager is mock_artifact_manager


@pytest.mark.asyncio
async def test_get_task_manager_without_scheduler() -> None:
    """Test get_task_manager handles missing scheduler gracefully."""
    mock_session = Mock(spec=AsyncSession)
    mock_artifact_manager = Mock(spec=ArtifactManager)
    mock_database = Mock()

    with (
        patch("chapkit.api.dependencies.get_scheduler", side_effect=RuntimeError("No scheduler")),
        patch("chapkit.api.dependencies.get_database", return_value=mock_database),
    ):
        manager = await get_task_manager(mock_session, mock_artifact_manager)

        assert isinstance(manager, TaskManager)
        assert manager.scheduler is None
        assert manager.database is mock_database


@pytest.mark.asyncio
async def test_get_task_manager_without_database() -> None:
    """Test get_task_manager handles missing database gracefully."""
    mock_session = Mock(spec=AsyncSession)
    mock_artifact_manager = Mock(spec=ArtifactManager)
    mock_scheduler = Mock()

    with (
        patch("chapkit.api.dependencies.get_scheduler", return_value=mock_scheduler),
        patch("chapkit.api.dependencies.get_database", side_effect=RuntimeError("No database")),
    ):
        manager = await get_task_manager(mock_session, mock_artifact_manager)

        assert isinstance(manager, TaskManager)
        assert manager.scheduler is mock_scheduler
        assert manager.database is None


@pytest.mark.asyncio
async def test_get_task_manager_without_scheduler_or_database() -> None:
    """Test get_task_manager handles missing scheduler and database."""
    mock_session = Mock(spec=AsyncSession)
    mock_artifact_manager = Mock(spec=ArtifactManager)

    with (
        patch("chapkit.api.dependencies.get_scheduler", side_effect=RuntimeError("No scheduler")),
        patch("chapkit.api.dependencies.get_database", side_effect=RuntimeError("No database")),
    ):
        manager = await get_task_manager(mock_session, mock_artifact_manager)

        assert isinstance(manager, TaskManager)
        assert manager.scheduler is None
        assert manager.database is None


@pytest.mark.asyncio
async def test_get_ml_manager_raises_error() -> None:
    """Test get_ml_manager raises RuntimeError as it's a placeholder."""
    with pytest.raises(RuntimeError, match="ML manager dependency not configured"):
        await get_ml_manager()
