"""Tests for type-based dependency injection in Python tasks."""

from __future__ import annotations

from typing import Any

import pytest
from servicekit import Database, SqliteDatabaseBuilder
from sqlalchemy.ext.asyncio import AsyncSession

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
