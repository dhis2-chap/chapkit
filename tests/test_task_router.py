"""Tests for TaskRouter error handling."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ulid import ULID

from chapkit import TaskIn, TaskManager, TaskOut
from chapkit.task import TaskRouter


def test_execute_task_value_error_returns_400() -> None:
    """Test that ValueError from execute_task returns 400 Bad Request."""
    # Create mock manager that raises ValueError
    mock_manager = Mock(spec=TaskManager)
    mock_manager.execute_task = AsyncMock(side_effect=ValueError("Task not found"))

    def manager_factory() -> TaskManager:
        return mock_manager

    # Create app with router
    app = FastAPI()
    router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        entity_in_type=TaskIn,
        entity_out_type=TaskOut,
        manager_factory=manager_factory,
    )
    app.include_router(router)

    client = TestClient(app)
    task_id = ULID()

    response = client.post(f"/api/v1/tasks/{task_id}/$execute")

    assert response.status_code == 400
    assert "Task not found" in response.json()["detail"]


def test_execute_task_runtime_error_returns_409() -> None:
    """Test that RuntimeError from execute_task returns 409 Conflict."""
    # Create mock manager that raises RuntimeError
    mock_manager = Mock(spec=TaskManager)
    mock_manager.execute_task = AsyncMock(side_effect=RuntimeError("Database instance required for task execution"))

    def manager_factory() -> TaskManager:
        return mock_manager

    # Create app with router
    app = FastAPI()
    router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        entity_in_type=TaskIn,
        entity_out_type=TaskOut,
        manager_factory=manager_factory,
    )
    app.include_router(router)

    client = TestClient(app)
    task_id = ULID()

    response = client.post(f"/api/v1/tasks/{task_id}/$execute")

    assert response.status_code == 409
    assert "Database instance required" in response.json()["detail"]


def test_execute_task_with_valid_ulid() -> None:
    """Test execute_task endpoint with valid ULID."""
    mock_manager = Mock(spec=TaskManager)
    job_id = ULID()
    mock_manager.execute_task = AsyncMock(return_value=job_id)

    def manager_factory() -> TaskManager:
        return mock_manager

    app = FastAPI()
    router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        entity_in_type=TaskIn,
        entity_out_type=TaskOut,
        manager_factory=manager_factory,
    )
    app.include_router(router)

    client = TestClient(app)
    task_id = ULID()

    response = client.post(f"/api/v1/tasks/{task_id}/$execute")

    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == str(job_id)
    assert "submitted for execution" in data["message"]


def test_list_tasks_with_enabled_filter_true() -> None:
    """Test GET /tasks?enabled=true returns only enabled tasks."""
    # Create mock manager
    mock_manager = Mock(spec=TaskManager)

    now = datetime.now(timezone.utc)

    enabled_task1 = TaskOut(
        id=ULID(),
        command="echo enabled1",
        task_type="shell",
        parameters=None,
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    enabled_task2 = TaskOut(
        id=ULID(),
        command="echo enabled2",
        task_type="shell",
        parameters=None,
        enabled=True,
        created_at=now,
        updated_at=now,
    )

    mock_manager.find_all = AsyncMock(return_value=[enabled_task1, enabled_task2])

    def manager_factory() -> TaskManager:
        return mock_manager

    # Create app with router
    app = FastAPI()
    router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        entity_in_type=TaskIn,
        entity_out_type=TaskOut,
        manager_factory=manager_factory,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get("/api/v1/tasks?enabled=true")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(task["enabled"] for task in data)
    # Verify find_all was called with enabled=True
    mock_manager.find_all.assert_called_once()
    call_kwargs = mock_manager.find_all.call_args.kwargs
    assert call_kwargs.get("enabled") is True


def test_list_tasks_with_enabled_filter_false() -> None:
    """Test GET /tasks?enabled=false returns only disabled tasks."""
    # Create mock manager
    mock_manager = Mock(spec=TaskManager)

    now = datetime.now(timezone.utc)

    disabled_task1 = TaskOut(
        id=ULID(),
        command="echo disabled1",
        task_type="shell",
        parameters=None,
        enabled=False,
        created_at=now,
        updated_at=now,
    )
    disabled_task2 = TaskOut(
        id=ULID(),
        command="echo disabled2",
        task_type="shell",
        parameters=None,
        enabled=False,
        created_at=now,
        updated_at=now,
    )

    mock_manager.find_all = AsyncMock(return_value=[disabled_task1, disabled_task2])

    def manager_factory() -> TaskManager:
        return mock_manager

    # Create app with router
    app = FastAPI()
    router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        entity_in_type=TaskIn,
        entity_out_type=TaskOut,
        manager_factory=manager_factory,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get("/api/v1/tasks?enabled=false")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(not task["enabled"] for task in data)
    # Verify find_all was called with enabled=False
    mock_manager.find_all.assert_called_once()
    call_kwargs = mock_manager.find_all.call_args.kwargs
    assert call_kwargs.get("enabled") is False


def test_list_tasks_without_enabled_filter() -> None:
    """Test GET /tasks returns all tasks when enabled parameter not provided."""
    # Create mock manager
    mock_manager = Mock(spec=TaskManager)

    now = datetime.now(timezone.utc)

    task1 = TaskOut(
        id=ULID(),
        command="echo enabled",
        task_type="shell",
        parameters=None,
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    task2 = TaskOut(
        id=ULID(),
        command="echo disabled",
        task_type="shell",
        parameters=None,
        enabled=False,
        created_at=now,
        updated_at=now,
    )

    mock_manager.find_all = AsyncMock(return_value=[task1, task2])

    def manager_factory() -> TaskManager:
        return mock_manager

    # Create app with router
    app = FastAPI()
    router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        entity_in_type=TaskIn,
        entity_out_type=TaskOut,
        manager_factory=manager_factory,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get("/api/v1/tasks")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    # Verify find_all was called with enabled=None
    mock_manager.find_all.assert_called_once()
    call_kwargs = mock_manager.find_all.call_args.kwargs
    assert call_kwargs.get("enabled") is None
