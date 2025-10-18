"""FastAPI service demonstrating task execution with artifact-based result storage."""

from __future__ import annotations

from fastapi import FastAPI
from servicekit import Database
from ulid import ULID

from chapkit import ArtifactHierarchy, TaskIn, TaskManager, TaskRepository
from chapkit.api import ServiceBuilder, ServiceInfo


async def seed_example_tasks(app: FastAPI) -> None:
    """Seed example task templates with stable ULIDs."""
    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        task_repo = TaskRepository(session)
        # Note: artifact_manager not needed for seeding task templates
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Check if tasks already exist
        existing_tasks = await task_manager.find_all()
        if len(existing_tasks) > 0:
            return  # Skip seeding if tasks already exist

        # Example 1: Simple directory listing
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000TASK1"),
                command="ls -la /tmp",
            )
        )

        # Example 2: Echo command with output
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000TASK2"),
                command='echo "Hello from task execution!"',
            )
        )

        # Example 3: Date command
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000TASK3"),
                command="date",
            )
        )

        # Example 4: Python one-liner
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000TASK4"),
                command='python3 -c "print(\\"Python task execution works!\\")"',
            )
        )

        # Example 5: Command that will fail (demonstrates error capture)
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000TASK5"),
                command="ls /nonexistent/directory",
            )
        )


info = ServiceInfo(
    display_name="Task Execution Service",
    summary="Example service demonstrating task execution with artifact-based result storage",
    version="1.0.0",
    description="""This service demonstrates chapkit's clean task execution architecture.""",
)

# Simple hierarchy for task execution artifacts
TASK_HIERARCHY = ArtifactHierarchy(
    name="task_executions",
    level_labels={0: "execution"},
)

app = (
    ServiceBuilder(info=info)
    .with_health()
    .with_artifacts(hierarchy=TASK_HIERARCHY)  # Required for task execution results
    .with_jobs(max_concurrency=3)  # Limit concurrent task execution
    .with_tasks()
    .on_startup(seed_example_tasks)
    .build()
)

if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("task_execution_api:app")
