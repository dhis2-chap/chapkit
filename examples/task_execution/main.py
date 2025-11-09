"""Example demonstrating registry-based task execution system."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from servicekit import Database
from servicekit.api.dependencies import get_database, get_scheduler

from chapkit import run_shell
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.artifact import ArtifactHierarchy, ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitJobScheduler
from chapkit.task import TaskExecutor, TaskRegistry, TaskRouter

# Define artifact hierarchy for task results
TASK_HIERARCHY = ArtifactHierarchy(
    name="task_results",
    level_labels={0: "task_run", 1: "output"},
)


# ==================== Task Functions ====================

# Clear registry on module reload (for development hot-reload)
TaskRegistry.clear()


@TaskRegistry.register("greet_user", tags=["demo", "simple"])
async def greet_user(name: str = "World") -> dict[str, str]:
    """Simple task that returns a greeting."""
    return {"message": f"Hello, {name}!"}


@TaskRegistry.register("process_data", tags=["demo", "injection"])
async def process_data(database: Database, artifact_manager: ArtifactManager) -> dict[str, object]:
    """Task with dependency injection - database and artifact manager injected automatically."""
    # Example: Store processing result in artifact
    artifact = await artifact_manager.save(
        ArtifactIn(
            data={
                "status": "processed",
                "records": 42,
                "timestamp": "2025-10-18T12:00:00Z",
            }
        )
    )
    return {
        "status": "complete",
        "artifact_id": str(artifact.id),
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
async def get_task_executor(
    scheduler: ChapkitJobScheduler = Depends(get_scheduler),
    database: Database = Depends(get_database),
) -> TaskExecutor:
    """Provide task executor for dependency injection."""
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo, hierarchy=TASK_HIERARCHY)
        return TaskExecutor(scheduler, database, artifact_manager)


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
    .with_jobs(max_concurrency=5)  # Required for task execution
    .include_router(task_router.router)  # Add task router before build
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
