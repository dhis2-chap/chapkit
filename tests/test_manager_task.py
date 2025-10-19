"""Tests for TaskManager error handling and edge cases."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from servicekit import Database
from ulid import ULID

from chapkit.artifact import ArtifactManager
from chapkit.scheduler import ChapkitJobScheduler
from chapkit.task import Task, TaskManager, TaskRegistry, TaskRepository


@pytest.mark.asyncio
async def test_execute_task_without_scheduler() -> None:
    """Test execute_task raises error when scheduler not configured."""
    mock_repo = Mock(spec=TaskRepository)
    manager = TaskManager(repo=mock_repo, scheduler=None, database=None, artifact_manager=None)

    with pytest.raises(ValueError, match="Task execution requires a scheduler"):
        await manager.execute_task(ULID())


@pytest.mark.asyncio
async def test_execute_task_without_artifact_manager() -> None:
    """Test execute_task raises error when artifact manager not configured."""
    mock_repo = Mock(spec=TaskRepository)
    mock_scheduler = Mock(spec=ChapkitJobScheduler)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=None,
    )

    with pytest.raises(ValueError, match="Task execution requires artifacts"):
        await manager.execute_task(ULID())


@pytest.mark.asyncio
async def test_execute_task_not_found() -> None:
    """Test execute_task raises error for non-existent task."""
    mock_repo = Mock(spec=TaskRepository)
    mock_repo.find_by_id = AsyncMock(return_value=None)

    mock_scheduler = Mock(spec=ChapkitJobScheduler)
    mock_artifact_manager = Mock(spec=ArtifactManager)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=mock_artifact_manager,
    )

    task_id = ULID()
    with pytest.raises(ValueError, match=f"Task {task_id} not found"):
        await manager.execute_task(task_id)

    mock_repo.find_by_id.assert_called_once_with(task_id)


@pytest.mark.asyncio
async def test_execute_task_submits_to_scheduler() -> None:
    """Test execute_task successfully submits job to scheduler."""
    task_id = ULID()
    job_id = ULID()

    mock_task = Mock(spec=Task)
    mock_task.id = task_id
    mock_task.command = "echo test"

    mock_repo = Mock(spec=TaskRepository)
    mock_repo.find_by_id = AsyncMock(return_value=mock_task)

    mock_scheduler = Mock(spec=ChapkitJobScheduler)
    mock_scheduler.add_job = AsyncMock(return_value=job_id)

    mock_artifact_manager = Mock(spec=ArtifactManager)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=mock_artifact_manager,
    )

    result = await manager.execute_task(task_id)

    assert result == job_id
    mock_repo.find_by_id.assert_called_once_with(task_id)
    mock_scheduler.add_job.assert_called_once()


@pytest.mark.asyncio
async def test_execute_command_without_database() -> None:
    """Test _execute_command raises error when database not configured."""
    mock_repo = Mock(spec=TaskRepository)
    manager = TaskManager(repo=mock_repo, scheduler=None, database=None, artifact_manager=None)

    with pytest.raises(RuntimeError, match="Database instance required"):
        await manager._execute_command(ULID())


@pytest.mark.asyncio
async def test_execute_command_without_artifact_manager() -> None:
    """Test _execute_command raises error when artifact manager not configured."""
    mock_repo = Mock(spec=TaskRepository)
    mock_database = Mock(spec=Database)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=None,
        database=mock_database,
        artifact_manager=None,
    )

    with pytest.raises(RuntimeError, match="ArtifactManager instance required"):
        await manager._execute_command(ULID())


@pytest.mark.asyncio
async def test_execute_command_task_not_found() -> None:
    """Test _execute_command raises error for non-existent task."""
    task_id = ULID()

    # Mock session context manager
    mock_session = Mock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    # Mock task repo that returns None
    mock_task_repo = Mock(spec=TaskRepository)
    mock_task_repo.find_by_id = AsyncMock(return_value=None)

    # Mock database
    mock_database = Mock(spec=Database)
    mock_database.session = Mock(return_value=mock_session)

    # Patch TaskRepository to return our mock
    with patch(
        "chapkit.task.manager.TaskRepository",
        return_value=mock_task_repo,
    ):
        mock_repo = Mock(spec=TaskRepository)
        mock_artifact_manager = Mock(spec=ArtifactManager)

        manager = TaskManager(
            repo=mock_repo,
            scheduler=None,
            database=mock_database,
            artifact_manager=mock_artifact_manager,
        )

        with pytest.raises(ValueError, match=f"Task {task_id} not found"):
            await manager._execute_command(task_id)


@pytest.mark.asyncio
async def test_execute_task_routes_to_python() -> None:
    """Test execute_task routes Python tasks to _execute_python."""
    task_id = ULID()
    job_id = ULID()

    mock_task = Mock(spec=Task)
    mock_task.id = task_id
    mock_task.command = "test_func"
    mock_task.task_type = "python"

    mock_repo = Mock(spec=TaskRepository)
    mock_repo.find_by_id = AsyncMock(return_value=mock_task)

    mock_scheduler = Mock(spec=ChapkitJobScheduler)
    mock_scheduler.add_job = AsyncMock(return_value=job_id)

    mock_artifact_manager = Mock(spec=ArtifactManager)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=mock_artifact_manager,
    )

    result = await manager.execute_task(task_id)

    assert result == job_id
    # Verify _execute_python was passed to scheduler
    call_args = mock_scheduler.add_job.call_args
    assert call_args[0][0].__name__ == "_execute_python"


@pytest.mark.asyncio
async def test_execute_task_routes_to_shell() -> None:
    """Test execute_task routes shell tasks to _execute_command."""
    task_id = ULID()
    job_id = ULID()

    mock_task = Mock(spec=Task)
    mock_task.id = task_id
    mock_task.command = "echo test"
    mock_task.task_type = "shell"

    mock_repo = Mock(spec=TaskRepository)
    mock_repo.find_by_id = AsyncMock(return_value=mock_task)

    mock_scheduler = Mock(spec=ChapkitJobScheduler)
    mock_scheduler.add_job = AsyncMock(return_value=job_id)

    mock_artifact_manager = Mock(spec=ArtifactManager)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=mock_artifact_manager,
    )

    result = await manager.execute_task(task_id)

    assert result == job_id
    # Verify _execute_command was passed to scheduler
    call_args = mock_scheduler.add_job.call_args
    assert call_args[0][0].__name__ == "_execute_command"


@pytest.mark.asyncio
async def test_execute_python_without_database() -> None:
    """Test _execute_python raises error when database not configured."""
    mock_repo = Mock(spec=TaskRepository)
    manager = TaskManager(repo=mock_repo, scheduler=None, database=None, artifact_manager=None)

    with pytest.raises(RuntimeError, match="Database instance required"):
        await manager._execute_python(ULID())


@pytest.mark.asyncio
async def test_execute_python_without_artifact_manager() -> None:
    """Test _execute_python raises error when artifact manager not configured."""
    mock_repo = Mock(spec=TaskRepository)
    mock_database = Mock(spec=Database)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=None,
        database=mock_database,
        artifact_manager=None,
    )

    with pytest.raises(RuntimeError, match="ArtifactManager instance required"):
        await manager._execute_python(ULID())


@pytest.mark.asyncio
async def test_execute_python_task_not_found() -> None:
    """Test _execute_python raises error for non-existent task."""
    task_id = ULID()

    # Mock session context manager
    mock_session = Mock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    # Mock task repo that returns None
    mock_task_repo = Mock(spec=TaskRepository)
    mock_task_repo.find_by_id = AsyncMock(return_value=None)

    # Mock database
    mock_database = Mock(spec=Database)
    mock_database.session = Mock(return_value=mock_session)

    # Patch TaskRepository to return our mock
    with patch(
        "chapkit.task.manager.TaskRepository",
        return_value=mock_task_repo,
    ):
        mock_repo = Mock(spec=TaskRepository)
        mock_artifact_manager = Mock(spec=ArtifactManager)

        manager = TaskManager(
            repo=mock_repo,
            scheduler=None,
            database=mock_database,
            artifact_manager=mock_artifact_manager,
        )

        with pytest.raises(ValueError, match=f"Task {task_id} not found"):
            await manager._execute_python(task_id)


@pytest.mark.asyncio
async def test_execute_python_function_not_in_registry() -> None:
    """Test _execute_python raises error when function not in registry."""
    task_id = ULID()

    # Clear registry
    TaskRegistry.clear()

    # Mock task
    mock_task = Mock(spec=Task)
    mock_task.id = task_id
    mock_task.command = "missing_function"
    mock_task.task_type = "python"
    mock_task.parameters = {}
    mock_task.created_at = datetime.now(timezone.utc)
    mock_task.updated_at = datetime.now(timezone.utc)

    # Mock session context manager
    mock_session = Mock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    # Mock task repo
    mock_task_repo = Mock(spec=TaskRepository)
    mock_task_repo.find_by_id = AsyncMock(return_value=mock_task)

    # Mock database
    mock_database = Mock(spec=Database)
    mock_database.session = Mock(return_value=mock_session)

    # Patch TaskRepository
    with patch(
        "chapkit.task.manager.TaskRepository",
        return_value=mock_task_repo,
    ):
        mock_repo = Mock(spec=TaskRepository)
        mock_artifact_manager = Mock(spec=ArtifactManager)

        manager = TaskManager(
            repo=mock_repo,
            scheduler=None,
            database=mock_database,
            artifact_manager=mock_artifact_manager,
        )

        with pytest.raises(ValueError, match="Python function 'missing_function' not found in registry"):
            await manager._execute_python(task_id)


@pytest.mark.asyncio
async def test_execute_task_disabled() -> None:
    """Test execute_task raises error when task is disabled."""
    task_id = ULID()

    mock_task = Mock(spec=Task)
    mock_task.id = task_id
    mock_task.command = "echo test"
    mock_task.enabled = False  # Disabled task

    mock_repo = Mock(spec=TaskRepository)
    mock_repo.find_by_id = AsyncMock(return_value=mock_task)

    mock_scheduler = Mock(spec=ChapkitJobScheduler)
    mock_artifact_manager = Mock(spec=ArtifactManager)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=mock_artifact_manager,
    )

    with pytest.raises(ValueError, match=f"Cannot execute disabled task {task_id}"):
        await manager.execute_task(task_id)

    mock_repo.find_by_id.assert_called_once_with(task_id)


@pytest.mark.asyncio
async def test_execute_task_enabled() -> None:
    """Test execute_task successfully executes enabled task."""
    task_id = ULID()
    job_id = ULID()

    mock_task = Mock(spec=Task)
    mock_task.id = task_id
    mock_task.command = "echo test"
    mock_task.enabled = True  # Enabled task

    mock_repo = Mock(spec=TaskRepository)
    mock_repo.find_by_id = AsyncMock(return_value=mock_task)

    mock_scheduler = Mock(spec=ChapkitJobScheduler)
    mock_scheduler.add_job = AsyncMock(return_value=job_id)

    mock_artifact_manager = Mock(spec=ArtifactManager)

    manager = TaskManager(
        repo=mock_repo,
        scheduler=mock_scheduler,
        database=None,
        artifact_manager=mock_artifact_manager,
    )

    result = await manager.execute_task(task_id)

    assert result == job_id
    mock_repo.find_by_id.assert_called_once_with(task_id)
    mock_scheduler.add_job.assert_called_once()
