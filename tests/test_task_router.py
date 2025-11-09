"""Tests for TaskRouter with registry-based API."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chapkit.task import TaskExecutor, TaskRouter
from chapkit.task.registry import TaskRegistry


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    """Clear registry before each test."""
    TaskRegistry.clear()


def test_list_all_tasks() -> None:
    """Test GET /tasks lists all registered tasks."""

    # Register test tasks
    @TaskRegistry.register("task1", tags=["demo"])
    def task1() -> dict:
        """First task."""
        return {}

    @TaskRegistry.register("task2", tags=["test"])
    def task2() -> dict:
        """Second task."""
        return {}

    # Create app with router
    mock_executor = Mock(spec=TaskExecutor)

    def executor_factory() -> TaskExecutor:
        return mock_executor

    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.get("/api/v1/tasks")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "task1"
    assert data[0]["tags"] == ["demo"]
    assert data[0]["docstring"] == "First task."
    assert data[1]["name"] == "task2"
    assert data[1]["tags"] == ["test"]


def test_get_task_by_name() -> None:
    """Test GET /tasks/{name} returns task metadata."""

    @TaskRegistry.register("test_task", tags=["demo"])
    def test_task(a: int, b: str = "default") -> dict:
        """Test task with parameters."""
        return {}

    # Create app with router
    mock_executor = Mock(spec=TaskExecutor)

    def executor_factory() -> TaskExecutor:
        return mock_executor

    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.get("/api/v1/tasks/test_task")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_task"
    assert data["docstring"] == "Test task with parameters."
    assert data["tags"] == ["demo"]
    assert len(data["parameters"]) == 2


def test_get_task_by_name_not_found() -> None:
    """Test GET /tasks/{name} returns 404 for non-existent task."""
    # Create app with router
    mock_executor = Mock(spec=TaskExecutor)

    def executor_factory() -> TaskExecutor:
        return mock_executor

    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.get("/api/v1/tasks/non_existent")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_execute_task_by_name() -> None:
    """Test POST /tasks/{name}/$execute executes task and returns result."""

    @TaskRegistry.register("test_task")
    def test_task(value: int) -> dict:
        """Test task."""
        return {"result": value * 2}

    # Create mock executor
    mock_executor = Mock(spec=TaskExecutor)
    mock_executor.execute = AsyncMock(return_value={"result": 42})

    def executor_factory() -> TaskExecutor:
        return mock_executor

    # Create app with router
    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.post(
        "/api/v1/tasks/test_task/$execute",
        json={"params": {"value": 21}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task_name"] == "test_task"
    assert data["params"] == {"value": 21}
    assert data["result"] == {"result": 42}
    assert data["error"] is None

    # Verify executor.execute was called with correct params
    mock_executor.execute.assert_called_once_with("test_task", {"value": 21})


def test_execute_task_without_params() -> None:
    """Test POST /tasks/{name}/$execute works without params."""

    @TaskRegistry.register("test_task")
    def test_task() -> dict:
        """Test task with no parameters."""
        return {"result": "success"}

    # Create mock executor
    mock_executor = Mock(spec=TaskExecutor)
    mock_executor.execute = AsyncMock(return_value={"result": "success"})

    def executor_factory() -> TaskExecutor:
        return mock_executor

    # Create app with router
    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.post("/api/v1/tasks/test_task/$execute", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["task_name"] == "test_task"
    assert data["result"] == {"result": "success"}
    mock_executor.execute.assert_called_once_with("test_task", {})


def test_execute_task_not_found() -> None:
    """Test POST /tasks/{name}/$execute returns 404 for non-existent task."""
    # Create mock executor that raises ValueError
    mock_executor = Mock(spec=TaskExecutor)
    mock_executor.execute = AsyncMock(side_effect=ValueError("Task 'non_existent' not found in registry"))

    def executor_factory() -> TaskExecutor:
        return mock_executor

    # Create app with router
    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.post("/api/v1/tasks/non_existent/$execute", json={})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_execute_task_internal_error() -> None:
    """Test POST /tasks/{name}/$execute returns 200 with error in response."""

    @TaskRegistry.register("test_task")
    def test_task() -> dict:
        """Test task."""
        return {}

    # Create mock executor that raises generic exception
    mock_executor = Mock(spec=TaskExecutor)
    mock_executor.execute = AsyncMock(side_effect=Exception("Something went wrong"))

    def executor_factory() -> TaskExecutor:
        return mock_executor

    # Create app with router
    app = FastAPI()
    task_router = TaskRouter.create(
        prefix="/api/v1/tasks",
        tags=["Tasks"],
        executor_factory=executor_factory,
    )
    app.include_router(task_router.router)

    client = TestClient(app)
    response = client.post("/api/v1/tasks/test_task/$execute", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["task_name"] == "test_task"
    assert data["result"] is None
    assert data["error"] is not None
    assert data["error"]["type"] == "Exception"
    assert data["error"]["message"] == "Something went wrong"
