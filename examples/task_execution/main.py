"""Example demonstrating task execution system with Python and shell tasks."""

from __future__ import annotations

import sys

from fastapi import Depends, FastAPI
from servicekit import Database
from servicekit.api.dependencies import get_database, get_scheduler, get_session
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.artifact import ArtifactHierarchy, ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitJobScheduler
from chapkit.task import TaskIn, TaskManager, TaskOut, TaskRegistry, TaskRepository, TaskRouter

# Define artifact hierarchy for task results
TASK_HIERARCHY = ArtifactHierarchy(
    name="task_results",
    level_labels={0: "task_run", 1: "output"},
)


# ==================== Python Task Functions ====================


@TaskRegistry.register("greet_user")
async def greet_user(name: str = "World") -> dict[str, str]:
    """Simple task that returns a greeting."""
    return {"message": f"Hello, {name}!"}


@TaskRegistry.register("process_data")
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


@TaskRegistry.register("count_files")
async def count_files(directory: str = ".") -> dict[str, object]:
    """Task that counts files in a directory."""
    import os

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return {"count": len(files), "directory": directory}


# ==================== Service Setup ====================


async def seed_demo_tasks(app: FastAPI) -> None:
    """Startup hook that seeds the database with example tasks."""
    database: Database | None = getattr(app.state, "database", None)
    scheduler: ChapkitJobScheduler | None = getattr(app.state, "scheduler", None)
    if database is None or scheduler is None:
        return

    async with database.session() as session:
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo, hierarchy=TASK_HIERARCHY)

        task_manager = TaskManager(task_repo, scheduler, database, artifact_manager)

        # Clear existing tasks
        await task_manager.delete_all()

        # Seed Python function tasks
        await task_manager.save(
            TaskIn(
                command="greet_user",
                enabled=True,
            )
        )

        await task_manager.save(
            TaskIn(
                command="process_data",
                enabled=True,
            )
        )

        # Seed shell command tasks
        await task_manager.save(
            TaskIn(
                command=f"{sys.executable} --version",
                enabled=True,
            )
        )

        await task_manager.save(
            TaskIn(
                command="echo 'Task system ready!'",
                enabled=True,
            )
        )

        print("✓ Seeded example tasks (Python functions and shell commands)")


# Build service
info = ServiceInfo(
    display_name="Task Execution Service",
    version="1.0.0",
    summary="Task execution with Python and shell commands",
    description="Example service demonstrating task execution with dependency injection",
)


# Define task manager dependency
async def get_task_manager(
    session: AsyncSession = Depends(get_session),
    scheduler: ChapkitJobScheduler = Depends(get_scheduler),
    database: Database = Depends(get_database),
) -> TaskManager:
    """Provide task manager for dependency injection."""
    task_repo = TaskRepository(session)
    artifact_repo = ArtifactRepository(session)
    artifact_manager = ArtifactManager(artifact_repo, hierarchy=TASK_HIERARCHY)
    return TaskManager(task_repo, scheduler, database, artifact_manager)


# Create task router
task_router = TaskRouter.create(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
    manager_factory=get_task_manager,
    entity_in_type=TaskIn,
    entity_out_type=TaskOut,
)

app: FastAPI = (
    ServiceBuilder(info=info)
    .with_landing_page()
    .with_logging()
    .with_health()
    .with_system()
    .with_jobs(max_concurrency=5)  # Required for task execution
    .include_router(task_router)  # Add task router before build
    .on_startup(seed_demo_tasks)
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
