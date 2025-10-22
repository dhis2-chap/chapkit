"""Integration tests for shell command execution in TaskManager."""

import pytest
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit.artifact import ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitJobScheduler
from chapkit.task import Task, TaskManager, TaskRepository


async def test_execute_command_captures_stdout() -> None:
    """Test _execute_command captures stdout from shell command."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()

    async with db.session() as session:
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)
        scheduler = ChapkitJobScheduler()
        artifact_manager = ArtifactManager(artifact_repo)

        manager = TaskManager(
            repo=task_repo,
            scheduler=scheduler,
            database=db,
            artifact_manager=artifact_manager,
        )

        # Create a task with shell command that produces stdout
        task = Task(command="echo 'Hello from shell'")
        await task_repo.save(task)
        await task_repo.commit()

        # Execute shell command
        artifact_id = await manager._execute_command(task.id)

        # Verify artifact was created with stdout
        artifact = await artifact_manager.find_by_id(artifact_id)
        assert artifact is not None
        assert isinstance(artifact.data, dict)
        assert "stdout" in artifact.data
        assert "Hello from shell" in artifact.data["stdout"]
        assert artifact.data["exit_code"] == 0

    await db.dispose()


async def test_execute_command_captures_stderr() -> None:
    """Test _execute_command captures stderr from shell command."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()

    async with db.session() as session:
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)
        scheduler = ChapkitJobScheduler()
        artifact_manager = ArtifactManager(artifact_repo)

        manager = TaskManager(
            repo=task_repo,
            scheduler=scheduler,
            database=db,
            artifact_manager=artifact_manager,
        )

        # Create a task that writes to stderr
        task = Task(command="echo 'Error message' >&2")
        await task_repo.save(task)
        await task_repo.commit()

        # Execute shell command
        artifact_id = await manager._execute_command(task.id)

        # Verify artifact was created with stderr
        artifact = await artifact_manager.find_by_id(artifact_id)
        assert artifact is not None
        assert isinstance(artifact.data, dict)
        assert "stderr" in artifact.data
        assert "Error message" in artifact.data["stderr"]

    await db.dispose()


async def test_execute_command_captures_exit_code() -> None:
    """Test _execute_command captures non-zero exit codes."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()

    async with db.session() as session:
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)
        scheduler = ChapkitJobScheduler()
        artifact_manager = ArtifactManager(artifact_repo)

        manager = TaskManager(
            repo=task_repo,
            scheduler=scheduler,
            database=db,
            artifact_manager=artifact_manager,
        )

        # Create a task that exits with error code
        task = Task(command="exit 42")
        await task_repo.save(task)
        await task_repo.commit()

        # Execute shell command
        artifact_id = await manager._execute_command(task.id)

        # Verify artifact captured exit code
        artifact = await artifact_manager.find_by_id(artifact_id)
        assert artifact is not None
        assert isinstance(artifact.data, dict)
        assert artifact.data["exit_code"] == 42

    await db.dispose()


async def test_execute_command_includes_task_snapshot() -> None:
    """Test _execute_command includes task metadata in artifact."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()

    async with db.session() as session:
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)
        scheduler = ChapkitJobScheduler()
        artifact_manager = ArtifactManager(artifact_repo)

        manager = TaskManager(
            repo=task_repo,
            scheduler=scheduler,
            database=db,
            artifact_manager=artifact_manager,
        )

        # Create a task
        task_id = ULID()
        task = Task(id=task_id, command="echo 'test'")
        await task_repo.save(task)
        await task_repo.commit()

        # Execute shell command
        artifact_id = await manager._execute_command(task_id)

        # Verify artifact includes task snapshot
        artifact = await artifact_manager.find_by_id(artifact_id)
        assert artifact is not None
        assert isinstance(artifact.data, dict)
        assert "task" in artifact.data
        task_data = artifact.data["task"]
        assert isinstance(task_data, dict)
        assert task_data["id"] == str(task_id)
        assert task_data["command"] == "echo 'test'"
        assert "created_at" in task_data
        assert "updated_at" in task_data

    await db.dispose()


async def test_execute_command_raises_on_missing_task() -> None:
    """Test _execute_command raises ValueError when task not found."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()

    async with db.session() as session:
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)
        scheduler = ChapkitJobScheduler()
        artifact_manager = ArtifactManager(artifact_repo)

        manager = TaskManager(
            repo=task_repo,
            scheduler=scheduler,
            database=db,
            artifact_manager=artifact_manager,
        )

        # Try to execute non-existent task
        missing_id = ULID()
        with pytest.raises(ValueError, match=f"Task {missing_id} not found"):
            await manager._execute_command(missing_id)

    await db.dispose()
