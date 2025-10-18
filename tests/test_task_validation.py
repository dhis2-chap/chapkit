"""Tests for validate_and_disable_orphaned_tasks utility."""

import pytest
from fastapi import FastAPI
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import TaskIn, TaskManager, TaskRegistry, TaskRepository, validate_and_disable_orphaned_tasks


@pytest.mark.asyncio
async def test_validate_no_database() -> None:
    """Test validation returns 0 when no database configured."""
    app = FastAPI()
    # No database in app.state

    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 0


@pytest.mark.asyncio
async def test_validate_no_orphaned_tasks() -> None:
    """Test validation returns 0 when all Python tasks are valid."""
    # Register a function
    TaskRegistry.clear()
    TaskRegistry.register_function("valid_func", lambda: {"result": "ok"})

    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create valid Python task
        await task_manager.save(TaskIn(id=ULID(), command="valid_func", task_type="python", enabled=True))

        # Create shell task (should be ignored)
        await task_manager.save(TaskIn(id=ULID(), command="echo test", task_type="shell", enabled=True))

    # Setup app with database
    app = FastAPI()
    app.state.database = database

    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 0

    # Verify tasks are still enabled
    async with database.session() as session:
        task_repo = TaskRepository(session)
        all_tasks = await task_repo.find_all()
        assert len(all_tasks) == 2
        assert all(task.enabled for task in all_tasks)

    TaskRegistry.clear()


@pytest.mark.asyncio
async def test_validate_disables_orphaned_tasks() -> None:
    """Test validation disables orphaned Python tasks."""
    TaskRegistry.clear()
    # Don't register "missing_func"

    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create orphaned Python task
        orphaned_task_id = ULID()
        await task_manager.save(TaskIn(id=orphaned_task_id, command="missing_func", task_type="python", enabled=True))

        # Create shell task (should not be affected)
        shell_task_id = ULID()
        await task_manager.save(TaskIn(id=shell_task_id, command="echo test", task_type="shell", enabled=True))

    # Setup app with database
    app = FastAPI()
    app.state.database = database

    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 1

    # Verify orphaned task is disabled
    async with database.session() as session:
        task_repo = TaskRepository(session)

        orphaned_task = await task_repo.find_by_id(orphaned_task_id)
        assert orphaned_task is not None
        assert orphaned_task.enabled is False

        shell_task = await task_repo.find_by_id(shell_task_id)
        assert shell_task is not None
        assert shell_task.enabled is True  # Still enabled

    TaskRegistry.clear()


@pytest.mark.asyncio
async def test_validate_multiple_orphaned_tasks() -> None:
    """Test validation disables multiple orphaned Python tasks."""
    TaskRegistry.clear()
    # Register only one function
    TaskRegistry.register_function("valid_func", lambda: {"result": "ok"})

    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create valid task
        valid_task_id = ULID()
        await task_manager.save(TaskIn(id=valid_task_id, command="valid_func", task_type="python", enabled=True))

        # Create orphaned tasks
        orphaned1_id = ULID()
        await task_manager.save(TaskIn(id=orphaned1_id, command="missing_func1", task_type="python", enabled=True))

        orphaned2_id = ULID()
        await task_manager.save(TaskIn(id=orphaned2_id, command="missing_func2", task_type="python", enabled=True))

    # Setup app with database
    app = FastAPI()
    app.state.database = database

    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 2

    # Verify correct tasks are disabled
    async with database.session() as session:
        task_repo = TaskRepository(session)

        valid_task = await task_repo.find_by_id(valid_task_id)
        assert valid_task is not None
        assert valid_task.enabled is True

        orphaned1 = await task_repo.find_by_id(orphaned1_id)
        assert orphaned1 is not None
        assert orphaned1.enabled is False

        orphaned2 = await task_repo.find_by_id(orphaned2_id)
        assert orphaned2 is not None
        assert orphaned2.enabled is False

    TaskRegistry.clear()


@pytest.mark.asyncio
async def test_validate_already_disabled_orphaned_task() -> None:
    """Test validation handles already disabled orphaned tasks."""
    TaskRegistry.clear()
    # Don't register "missing_func"

    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create orphaned task that's already disabled
        orphaned_task_id = ULID()
        await task_manager.save(TaskIn(id=orphaned_task_id, command="missing_func", task_type="python", enabled=False))

    # Setup app with database
    app = FastAPI()
    app.state.database = database

    # Run validation - should still disable it (idempotent)
    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 1

    # Verify task is still disabled
    async with database.session() as session:
        task_repo = TaskRepository(session)
        orphaned_task = await task_repo.find_by_id(orphaned_task_id)
        assert orphaned_task is not None
        assert orphaned_task.enabled is False

    TaskRegistry.clear()


@pytest.mark.asyncio
async def test_validate_no_tasks() -> None:
    """Test validation returns 0 when there are no tasks."""
    TaskRegistry.clear()

    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    # Setup app with empty database
    app = FastAPI()
    app.state.database = database

    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 0

    TaskRegistry.clear()


@pytest.mark.asyncio
async def test_validate_only_shell_tasks() -> None:
    """Test validation ignores shell tasks."""
    TaskRegistry.clear()

    database = SqliteDatabaseBuilder().in_memory().build()
    await database.init()

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Create only shell tasks
        await task_manager.save(TaskIn(id=ULID(), command="echo test1", task_type="shell", enabled=True))
        await task_manager.save(TaskIn(id=ULID(), command="echo test2", task_type="shell", enabled=True))

    # Setup app with database
    app = FastAPI()
    app.state.database = database

    disabled_count = await validate_and_disable_orphaned_tasks(app)

    assert disabled_count == 0

    # Verify all tasks still enabled
    async with database.session() as session:
        task_repo = TaskRepository(session)
        all_tasks = await task_repo.find_all()
        assert len(all_tasks) == 2
        assert all(task.enabled for task in all_tasks)

    TaskRegistry.clear()
