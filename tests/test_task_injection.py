"""Tests for type-based dependency injection in Python tasks."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from servicekit import Database, SqliteDatabaseBuilder
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit.artifact import ArtifactManager
from chapkit.scheduler import ChapkitScheduler
from chapkit.task import TaskExecutor
from chapkit.task.registry import TaskRegistry


@pytest.fixture
async def database() -> Database:
    """Create in-memory database for testing."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()
    return db


@pytest.fixture
def task_executor(database: Database) -> TaskExecutor:
    """Create task executor with database dependency."""
    return TaskExecutor(database)


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    """Clear registry before each test."""
    TaskRegistry.clear()


@pytest.mark.asyncio
async def test_inject_async_session(database: Database, task_executor: TaskExecutor) -> None:
    """Test AsyncSession injection into Python task."""

    @TaskRegistry.register("test_session_injection")
    async def task_with_session(session: AsyncSession) -> dict[str, Any]:
        """Task that uses injected session."""
        assert session is not None
        assert isinstance(session, AsyncSession)
        return {"session_injected": True}

    # Execute and get result directly
    result = await task_executor.execute("test_session_injection", {})
    assert result["session_injected"] is True


@pytest.mark.asyncio
async def test_inject_database(database: Database, task_executor: TaskExecutor) -> None:
    """Test Database injection into Python task."""

    @TaskRegistry.register("test_database_injection")
    async def task_with_database(db: Database) -> dict[str, Any]:
        """Task that uses injected database."""
        assert db is not None
        assert isinstance(db, Database)
        return {"database_injected": True}

    # Execute and get result directly
    result = await task_executor.execute("test_database_injection", {})
    assert result["database_injected"] is True


@pytest.mark.asyncio
async def test_inject_with_user_parameters(database: Database, task_executor: TaskExecutor) -> None:
    """Test mixing injected types with user parameters."""

    @TaskRegistry.register("test_mixed_params")
    async def task_with_mixed(
        name: str,  # From user parameters
        count: int,  # From user parameters
        session: AsyncSession,  # Injected
    ) -> dict[str, Any]:
        """Task that mixes user and injected parameters."""
        assert name == "test"
        assert count == 42
        assert session is not None
        return {"name": name, "count": count, "has_session": True}

    # Execute with user parameters
    result = await task_executor.execute("test_mixed_params", {"name": "test", "count": 42})
    assert result["name"] == "test"
    assert result["count"] == 42
    assert result["has_session"] is True


@pytest.mark.asyncio
async def test_optional_injection(database: Database, task_executor: TaskExecutor) -> None:
    """Test Optional type injection."""

    @TaskRegistry.register("test_optional_injection")
    async def task_with_optional(session: AsyncSession | None = None) -> dict[str, Any]:
        """Task with optional injected parameter."""
        return {"session_provided": session is not None}

    # Execute
    result = await task_executor.execute("test_optional_injection", {})
    assert result["session_provided"] is True


@pytest.mark.asyncio
async def test_missing_required_user_parameter(database: Database, task_executor: TaskExecutor) -> None:
    """Test error when required user parameter is missing."""

    @TaskRegistry.register("test_missing_param")
    async def task_with_required(name: str) -> dict[str, str]:
        """Task with required user parameter."""
        return {"name": name}

    # Execute without required parameter - should raise ValueError
    with pytest.raises(ValueError, match="Missing required parameter"):
        await task_executor.execute("test_missing_param", {})


@pytest.mark.asyncio
async def test_sync_function_injection(database: Database, task_executor: TaskExecutor) -> None:
    """Test dependency injection works with synchronous functions."""

    @TaskRegistry.register("test_sync_injection")
    def sync_task_with_database(db: Database) -> dict[str, bool]:
        """Synchronous task with database injection."""
        assert db is not None
        return {"sync_database_injected": True}

    # Execute synchronous function
    result = await task_executor.execute("test_sync_injection", {})
    assert result["sync_database_injected"] is True


@pytest.mark.asyncio
async def test_execute_nonexistent_task(database: Database, task_executor: TaskExecutor) -> None:
    """Test error when executing non-existent task."""
    with pytest.raises(ValueError, match="not found in registry"):
        await task_executor.execute("nonexistent_task", {})


@pytest.mark.asyncio
async def test_task_without_type_annotations(database: Database, task_executor: TaskExecutor) -> None:
    """Test task function without type annotations."""

    @TaskRegistry.register("test_no_annotations")
    async def task_no_annotations(value: Any) -> dict[str, Any]:
        """Task without strict type annotations."""
        return {"value": value}

    # Execute with user parameter
    result = await task_executor.execute("test_no_annotations", {"value": "test"})
    assert result["value"] == "test"


@pytest.mark.asyncio
async def test_inject_scheduler(database: Database) -> None:
    """Test ChapkitScheduler injection into Python task."""
    mock_scheduler = Mock(spec=ChapkitScheduler)
    executor = TaskExecutor(database, scheduler=mock_scheduler)

    @TaskRegistry.register("test_scheduler_injection")
    async def task_with_scheduler(scheduler: ChapkitScheduler) -> dict[str, Any]:
        """Task that uses injected scheduler."""
        assert scheduler is not None
        assert scheduler is mock_scheduler
        return {"scheduler_injected": True}

    # Execute and get result directly
    result = await executor.execute("test_scheduler_injection", {})
    assert result["scheduler_injected"] is True


@pytest.mark.asyncio
async def test_inject_artifact_manager(database: Database) -> None:
    """Test ArtifactManager injection into Python task."""
    mock_artifact_manager = Mock(spec=ArtifactManager)
    executor = TaskExecutor(database, artifact_manager=mock_artifact_manager)

    @TaskRegistry.register("test_artifact_manager_injection")
    async def task_with_artifact_manager(artifact_manager: ArtifactManager) -> dict[str, Any]:
        """Task that uses injected artifact manager."""
        assert artifact_manager is not None
        assert artifact_manager is mock_artifact_manager
        return {"artifact_manager_injected": True}

    # Execute and get result directly
    result = await executor.execute("test_artifact_manager_injection", {})
    assert result["artifact_manager_injected"] is True


@pytest.mark.asyncio
async def test_inject_all_optional_dependencies(database: Database) -> None:
    """Test injection with all optional dependencies."""
    mock_scheduler = Mock(spec=ChapkitScheduler)
    mock_artifact_manager = Mock(spec=ArtifactManager)
    executor = TaskExecutor(database, scheduler=mock_scheduler, artifact_manager=mock_artifact_manager)

    @TaskRegistry.register("test_all_dependencies")
    async def task_with_all(
        scheduler: ChapkitScheduler,
        artifact_manager: ArtifactManager,
        database: Database,
    ) -> dict[str, Any]:
        """Task that uses all injectable dependencies."""
        assert scheduler is mock_scheduler
        assert artifact_manager is mock_artifact_manager
        assert database is not None
        return {"all_injected": True}

    # Execute and get result directly
    result = await executor.execute("test_all_dependencies", {})
    assert result["all_injected"] is True


@pytest.mark.asyncio
async def test_optional_scheduler_not_provided(database: Database, task_executor: TaskExecutor) -> None:
    """Test optional scheduler injection when not provided."""

    @TaskRegistry.register("test_optional_scheduler")
    async def task_with_optional_scheduler(
        scheduler: ChapkitScheduler | None = None,
    ) -> dict[str, Any]:
        """Task with optional scheduler parameter."""
        return {"scheduler_provided": scheduler is not None}

    # Execute without scheduler - should default to None
    result = await task_executor.execute("test_optional_scheduler", {})
    assert result["scheduler_provided"] is False


@pytest.mark.asyncio
async def test_varargs_parameters(database: Database, task_executor: TaskExecutor) -> None:
    """Test that *args and **kwargs parameters are skipped."""

    @TaskRegistry.register("test_varargs")
    async def task_with_varargs(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Task with *args and **kwargs."""
        return {"name": name, "args": args, "kwargs": kwargs}

    # Execute with parameters
    result = await task_executor.execute("test_varargs", {"name": "test"})
    assert result["name"] == "test"
    assert result["args"] == ()
    assert result["kwargs"] == {}


@pytest.mark.asyncio
async def test_user_parameter_with_default(database: Database, task_executor: TaskExecutor) -> None:
    """Test user parameter with default value."""

    @TaskRegistry.register("test_default_param")
    async def task_with_default(name: str = "default_name") -> dict[str, str]:
        """Task with default parameter value."""
        return {"name": name}

    # Execute without parameter - should use default
    result = await task_executor.execute("test_default_param", {})
    assert result["name"] == "default_name"

    # Execute with parameter - should override default
    result = await task_executor.execute("test_default_param", {"name": "custom"})
    assert result["name"] == "custom"
