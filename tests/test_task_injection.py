"""Tests for type-based dependency injection in Python tasks."""

from __future__ import annotations

from typing import Any

import pytest
from servicekit import Database, SqliteDatabaseBuilder
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit.artifact import ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitJobScheduler
from chapkit.task import TaskExecutor
from chapkit.task.registry import TaskRegistry


@pytest.fixture
async def database() -> Database:
    """Create in-memory database for testing."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()
    return db


@pytest.fixture
async def task_executor(database: Database) -> TaskExecutor:
    """Create task executor with all dependencies."""
    scheduler = ChapkitJobScheduler()
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        return TaskExecutor(scheduler, database, artifact_manager)


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

    # Execute task by name
    job_id = await task_executor.execute("test_session_injection", {})

    # Wait for completion
    await task_executor.scheduler.wait(job_id)

    # Verify result
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"
    assert job_record.artifact_id is not None

    # Check artifact
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["session_injected"] is True


@pytest.mark.asyncio
async def test_inject_database(database: Database, task_executor: TaskExecutor) -> None:
    """Test Database injection into Python task."""

    @TaskRegistry.register("test_database_injection")
    async def task_with_database(db: Database) -> dict[str, Any]:
        """Task that uses injected database."""
        assert db is not None
        assert isinstance(db, Database)
        return {"database_injected": True}

    # Execute
    job_id = await task_executor.execute("test_database_injection", {})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["database_injected"] is True


@pytest.mark.asyncio
async def test_inject_artifact_manager(database: Database, task_executor: TaskExecutor) -> None:
    """Test ArtifactManager injection into Python task."""

    @TaskRegistry.register("test_artifact_injection")
    async def task_with_artifacts(artifact_manager: ArtifactManager) -> dict[str, Any]:
        """Task that uses injected artifact manager."""
        assert artifact_manager is not None
        assert isinstance(artifact_manager, ArtifactManager)
        return {"artifact_manager_injected": True}

    # Execute
    job_id = await task_executor.execute("test_artifact_injection", {})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["artifact_manager_injected"] is True


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
    job_id = await task_executor.execute("test_mixed_params", {"name": "test", "count": 42})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        result = artifact.data["result"]
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
    job_id = await task_executor.execute("test_optional_injection", {})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["session_provided"] is True


@pytest.mark.asyncio
async def test_missing_required_user_parameter(database: Database, task_executor: TaskExecutor) -> None:
    """Test error when required user parameter is missing."""

    @TaskRegistry.register("test_missing_param")
    async def task_with_required(name: str, session: AsyncSession) -> dict[str, Any]:
        """Task with required user parameter."""
        return {"name": name}

    # Execute WITHOUT required parameter - should capture error
    job_id = await task_executor.execute("test_missing_param", {})

    # Wait for completion
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"  # Job completes but captures error

    # Check error in artifact
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is not None
        assert "Missing required parameter 'name'" in artifact.data["error"]["message"]


@pytest.mark.asyncio
async def test_sync_function_injection(database: Database, task_executor: TaskExecutor) -> None:
    """Test injection works with sync functions too."""

    @TaskRegistry.register("test_sync_injection")
    def sync_task_with_injection(value: int, database: Database) -> dict[str, Any]:
        """Sync task with injection."""
        assert database is not None
        return {"value": value * 2, "has_database": True}

    # Execute
    job_id = await task_executor.execute("test_sync_injection", {"value": 21})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["value"] == 42
        assert artifact.data["result"]["has_database"] is True


@pytest.mark.asyncio
async def test_inject_scheduler(database: Database, task_executor: TaskExecutor) -> None:
    """Test ChapkitJobScheduler injection into Python task."""

    @TaskRegistry.register("test_scheduler_injection")
    async def task_with_scheduler(scheduler: ChapkitJobScheduler) -> dict[str, Any]:
        """Task that uses injected scheduler."""
        assert scheduler is not None
        assert isinstance(scheduler, ChapkitJobScheduler)
        return {"scheduler_injected": True}

    # Execute
    job_id = await task_executor.execute("test_scheduler_injection", {})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["scheduler_injected"] is True


@pytest.mark.asyncio
async def test_execute_nonexistent_task(database: Database, task_executor: TaskExecutor) -> None:
    """Test that executing a non-existent task raises ValueError."""
    with pytest.raises(ValueError, match="not found in registry"):
        await task_executor.execute("nonexistent_task", {})


@pytest.mark.asyncio
async def test_task_without_type_annotations(database: Database, task_executor: TaskExecutor) -> None:
    """Test task with parameters that have no type annotations."""

    @TaskRegistry.register("test_no_annotations")
    async def task_no_annotations(value):  # pyright: ignore[reportMissingParameterType]
        """Task without type annotation."""
        return {"value": value}

    # Execute with parameter
    job_id = await task_executor.execute("test_no_annotations", {"value": 42})

    # Wait and verify
    await task_executor.scheduler.wait(job_id)
    job_record = await task_executor.scheduler.get_record(job_id)
    assert job_record is not None
    assert job_record.status == "completed"

    # Check result
    assert job_record.artifact_id is not None
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_mgr = ArtifactManager(artifact_repo)
        artifact = await artifact_mgr.find_by_id(job_record.artifact_id)
        assert artifact is not None
        assert artifact.data["error"] is None
        assert artifact.data["result"]["value"] == 42
