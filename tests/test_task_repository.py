"""Tests for TaskRepository enabled filtering."""

import pytest
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit.task import TaskIn, TaskManager, TaskRepository


@pytest.mark.asyncio
async def test_find_by_enabled_true() -> None:
    """Test finding only enabled tasks."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create enabled and disabled tasks
        await task_manager.save(TaskIn(id=ULID(), command="enabled1", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="enabled2", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="disabled1", task_type="shell", enabled=False))

        # Find only enabled tasks
        enabled_tasks = await task_repo.find_by_enabled(True)

        assert len(enabled_tasks) == 2
        assert all(task.enabled for task in enabled_tasks)
        assert {task.command for task in enabled_tasks} == {"enabled1", "enabled2"}


@pytest.mark.asyncio
async def test_find_by_enabled_false() -> None:
    """Test finding only disabled tasks."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create enabled and disabled tasks
        await task_manager.save(TaskIn(id=ULID(), command="enabled1", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="disabled1", task_type="shell", enabled=False))
        await task_manager.save(TaskIn(id=ULID(), command="disabled2", task_type="shell", enabled=False))

        # Find only disabled tasks
        disabled_tasks = await task_repo.find_by_enabled(False)

        assert len(disabled_tasks) == 2
        assert all(not task.enabled for task in disabled_tasks)
        assert {task.command for task in disabled_tasks} == {"disabled1", "disabled2"}


@pytest.mark.asyncio
async def test_find_all_with_enabled_filter_true() -> None:
    """Test find_all with enabled=True filter."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create mixed tasks
        await task_manager.save(TaskIn(id=ULID(), command="enabled1", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="disabled1", task_type="shell", enabled=False))

        # Filter for enabled only
        enabled_tasks = await task_repo.find_all(enabled=True)

        assert len(enabled_tasks) == 1
        assert enabled_tasks[0].command == "enabled1"
        assert enabled_tasks[0].enabled is True


@pytest.mark.asyncio
async def test_find_all_with_enabled_filter_false() -> None:
    """Test find_all with enabled=False filter."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create mixed tasks
        await task_manager.save(TaskIn(id=ULID(), command="enabled1", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="disabled1", task_type="shell", enabled=False))

        # Filter for disabled only
        disabled_tasks = await task_repo.find_all(enabled=False)

        assert len(disabled_tasks) == 1
        assert disabled_tasks[0].command == "disabled1"
        assert disabled_tasks[0].enabled is False


@pytest.mark.asyncio
async def test_find_all_without_filter() -> None:
    """Test find_all returns all tasks when enabled=None."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create mixed tasks
        await task_manager.save(TaskIn(id=ULID(), command="enabled1", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="disabled1", task_type="shell", enabled=False))
        await task_manager.save(TaskIn(id=ULID(), command="enabled2", task_type="shell", enabled=True))

        # Get all tasks (no filter)
        all_tasks = await task_repo.find_all(enabled=None)

        assert len(all_tasks) == 3
        commands = {task.command for task in all_tasks}
        assert commands == {"enabled1", "disabled1", "enabled2"}


@pytest.mark.asyncio
async def test_find_by_enabled_empty() -> None:
    """Test find_by_enabled returns empty list when no matches."""
    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create only enabled tasks
        await task_manager.save(TaskIn(id=ULID(), command="enabled1", task_type="shell", enabled=True))

        # Find disabled tasks (should be empty)
        disabled_tasks = await task_repo.find_by_enabled(False)

        assert len(disabled_tasks) == 0
