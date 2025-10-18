"""Tests for type-based dependency injection in Python tasks."""

from __future__ import annotations

from typing import Any

import pytest
from servicekit import AIOJobScheduler, Database, SqliteDatabaseBuilder
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit import ArtifactManager, TaskIn, TaskManager, TaskRepository
from chapkit.artifact import ArtifactRepository
from chapkit.task.registry import TaskRegistry


@pytest.fixture
async def database() -> Database:
    """Create in-memory database for testing."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()
    return db


@pytest.fixture
async def task_manager(database: Database) -> TaskManager:
    """Create task manager with all dependencies."""
    async with database.session() as session:
        task_repo = TaskRepository(session)
        scheduler = AIOJobScheduler()
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)

        return TaskManager(
            repo=task_repo,
            scheduler=scheduler,
            database=database,
            artifact_manager=artifact_manager,
        )


@pytest.mark.asyncio
async def test_inject_async_session(database: Database, task_manager: TaskManager) -> None:
    """Test AsyncSession injection into Python task."""

    # Register task that uses AsyncSession
    @TaskRegistry.register("test_session_injection")
    async def task_with_session(session: AsyncSession) -> dict[str, Any]:
        """Task that uses injected session."""
        assert session is not None
        assert isinstance(session, AsyncSession)
        return {"session_injected": True}

    try:
        # Create task
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_session_injection",
                    task_type="python",
                    parameters={},  # No user parameters
                )
            )

        # Execute with full manager (has scheduler)
        job_id = await task_manager.execute_task(task.id)

        # Wait for completion
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)

        # Verify result
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()


@pytest.mark.asyncio
async def test_inject_database(database: Database, task_manager: TaskManager) -> None:
    """Test Database injection into Python task."""

    @TaskRegistry.register("test_database_injection")
    async def task_with_database(db: Database) -> dict[str, Any]:
        """Task that uses injected database."""
        assert db is not None
        assert isinstance(db, Database)
        return {"database_injected": True}

    try:
        # Create task
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_database_injection",
                    task_type="python",
                    parameters={},
                )
            )

        # Execute
        job_id = await task_manager.execute_task(task.id)

        # Wait and verify
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()


@pytest.mark.asyncio
async def test_inject_artifact_manager(database: Database, task_manager: TaskManager) -> None:
    """Test ArtifactManager injection into Python task."""

    @TaskRegistry.register("test_artifact_injection")
    async def task_with_artifacts(artifact_manager: ArtifactManager) -> dict[str, Any]:
        """Task that uses injected artifact manager."""
        assert artifact_manager is not None
        assert isinstance(artifact_manager, ArtifactManager)
        return {"artifact_manager_injected": True}

    try:
        # Create task
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_artifact_injection",
                    task_type="python",
                    parameters={},
                )
            )

        # Execute
        job_id = await task_manager.execute_task(task.id)

        # Wait and verify
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()


@pytest.mark.asyncio
async def test_inject_with_user_parameters(database: Database, task_manager: TaskManager) -> None:
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

    try:
        # Create task with user parameters
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_mixed_params",
                    task_type="python",
                    parameters={"name": "test", "count": 42},
                )
            )

        # Execute
        job_id = await task_manager.execute_task(task.id)

        # Wait and verify
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()


@pytest.mark.asyncio
async def test_optional_injection(database: Database, task_manager: TaskManager) -> None:
    """Test Optional type injection."""

    @TaskRegistry.register("test_optional_injection")
    async def task_with_optional(session: AsyncSession | None = None) -> dict[str, Any]:
        """Task with optional injected parameter."""
        # Verify session was injected (not None)
        return {"session_provided": session is not None}

    try:
        # Create task
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_optional_injection",
                    task_type="python",
                    parameters={},
                )
            )

        # Execute
        job_id = await task_manager.execute_task(task.id)

        # Wait and verify
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()


@pytest.mark.asyncio
async def test_missing_required_user_parameter(database: Database, task_manager: TaskManager) -> None:
    """Test error when required user parameter is missing."""

    @TaskRegistry.register("test_missing_param")
    async def task_with_required(name: str, session: AsyncSession) -> dict[str, Any]:
        """Task with required user parameter."""
        return {"name": name}

    try:
        # Create task WITHOUT required parameter
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_missing_param",
                    task_type="python",
                    parameters={},  # Missing 'name'
                )
            )

        # Execute - should capture error
        job_id = await task_manager.execute_task(task.id)

        # Wait for completion
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()


@pytest.mark.asyncio
async def test_sync_function_injection(database: Database, task_manager: TaskManager) -> None:
    """Test injection works with sync functions too."""

    @TaskRegistry.register("test_sync_injection")
    def sync_task_with_injection(value: int, database: Database) -> dict[str, Any]:
        """Sync task with injection."""
        assert database is not None
        return {"value": value * 2, "has_database": True}

    try:
        # Create task
        async with database.session() as session:
            task_repo = TaskRepository(session)
            task_mgr = TaskManager(task_repo, None, database, None)
            task = await task_mgr.save(
                TaskIn(
                    command="test_sync_injection",
                    task_type="python",
                    parameters={"value": 21},
                )
            )

        # Execute
        job_id = await task_manager.execute_task(task.id)

        # Wait and verify
        scheduler = task_manager.scheduler
        assert scheduler is not None
        await scheduler.wait(job_id)
        job_record = await scheduler.get_record(job_id)
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
    finally:
        TaskRegistry.clear()
