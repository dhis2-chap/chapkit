"""Example demonstrating registry-based task execution system."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from servicekit import Database
from servicekit.api.dependencies import get_database

from chapkit import run_shell
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.task import TaskExecutor, TaskRegistry, TaskRouter

# ==================== Task Functions ====================

# Clear registry on module reload (for development hot-reload)
TaskRegistry.clear()


@TaskRegistry.register("greet_user", tags=["demo", "simple"])
async def greet_user(name: str = "World") -> dict[str, str]:
    """Simple task that returns a greeting."""
    return {"message": f"Hello, {name}!"}


@TaskRegistry.register("process_data", tags=["demo", "injection"])
async def process_data(database: Database) -> dict[str, object]:
    """Task with dependency injection - database injected automatically."""
    # Example: You could query the database here
    return {
        "status": "processed",
        "records": 42,
        "database_url": str(database.url),
    }


@TaskRegistry.register("count_files", tags=["demo", "filesystem"])
async def count_files(directory: str = ".") -> dict[str, object]:
    """Task that counts files in a directory."""
    import os

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return {"count": len(files), "directory": directory}


@TaskRegistry.register("multiply_numbers", tags=["demo", "math"])
def multiply_numbers(a: int, b: int) -> dict[str, int]:
    """Synchronous task that multiplies two numbers."""
    return {"result": a * b, "a": a, "b": b}


@TaskRegistry.register("run_command", tags=["demo", "subprocess"])
async def run_command(command: str) -> dict[str, object]:
    """Run a shell command using the run_shell utility."""
    return await run_shell(command)


# ==================== Service Setup ====================


# Build service
info = ServiceInfo(
    display_name="Task Execution Service",
    version="1.0.0",
    summary="Registry-based task execution with dependency injection",
    description="Example service demonstrating task execution without database persistence",
)


# Define task executor dependency
def get_task_executor(database: Database = Depends(get_database)) -> TaskExecutor:
    """Provide task executor for dependency injection."""
    return TaskExecutor(database)


# Create task router
task_router = TaskRouter.create(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
    executor_factory=get_task_executor,
)

app: FastAPI = (
    ServiceBuilder(info=info)
    .with_landing_page()
    .with_logging()
    .with_health()
    .with_system()
    .include_router(task_router.router)  # Add task router before build
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
