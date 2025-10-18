"""FastAPI service demonstrating read-only task API with pre-seeded tasks."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from servicekit import Database
from servicekit.api.crud import CrudPermissions
from ulid import ULID

from chapkit import (
    ArtifactHierarchy,
    TaskIn,
    TaskManager,
    TaskRegistry,
    TaskRepository,
)
from chapkit.api import ServiceBuilder, ServiceInfo


# Register Python task functions
@TaskRegistry.register("health_check")
async def health_check() -> dict:
    """Perform system health check."""
    await asyncio.sleep(0.1)
    return {"status": "healthy", "checks": ["database", "scheduler", "artifacts"]}


@TaskRegistry.register("cleanup_temp_files")
async def cleanup_temp_files(older_than_days: int = 7) -> dict:
    """Simulate cleanup of temporary files."""
    await asyncio.sleep(0.2)
    return {"cleaned": 42, "criteria": f"older than {older_than_days} days"}


@TaskRegistry.register("backup_database")
def backup_database(destination: str = "/backups") -> dict:
    """Simulate database backup operation."""
    return {"success": True, "destination": destination, "size_mb": 150}


async def seed_readonly_tasks(app: FastAPI) -> None:
    """Seed predefined task templates - the only way to create tasks in this service."""
    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Check if tasks already exist
        existing_tasks = await task_manager.find_all()
        if len(existing_tasks) > 0:
            return  # Skip seeding if tasks already exist

        # Task 1: Health check (enabled)
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000READ1"),
                command="health_check",
                task_type="python",
                parameters={},
                enabled=True,
            )
        )

        # Task 2: Cleanup temp files (enabled)
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000READ2"),
                command="cleanup_temp_files",
                task_type="python",
                parameters={"older_than_days": 7},
                enabled=True,
            )
        )

        # Task 3: Database backup (enabled)
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000READ3"),
                command="backup_database",
                task_type="python",
                parameters={"destination": "/backups"},
                enabled=True,
            )
        )

        # Task 4: Shell task (enabled)
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000READ4"),
                command="echo 'System check complete'",
                task_type="shell",
                enabled=True,
            )
        )

        # Task 5: Disabled maintenance task
        await task_manager.save(
            TaskIn(
                id=ULID.from_str("01JCSEED0000000000000READ5"),
                command="backup_database",
                task_type="python",
                parameters={"destination": "/archive"},
                enabled=False,
            )
        )


info = ServiceInfo(
    display_name="Read-Only Task Service",
    summary="Secure task execution service with pre-defined tasks only",
    version="1.0.0",
    description="""
    This service demonstrates a read-only task API pattern for production use.

    Security Features:
    - No runtime task creation (prevent command injection)
    - No runtime task updates (prevent tampering)
    - No runtime task deletion (preserve audit trail)
    - All tasks defined in code (version controlled)
    - API users can only view and execute pre-defined tasks

    This pattern is ideal for production deployments where tasks should be
    managed through code/configuration rather than runtime APIs.
    """,
)

# Simple hierarchy for task execution artifacts
TASK_HIERARCHY = ArtifactHierarchy(
    name="task_executions",
    level_labels={0: "execution"},
)

# Read-only CRUD permissions (no create, update, or delete)
READONLY_PERMISSIONS = CrudPermissions(
    create=False,  # Tasks can only be created via seeding
    read=True,  # Users can list and view tasks
    update=False,  # No runtime modifications
    delete=False,  # No runtime deletions
)

app = (
    ServiceBuilder(info=info)
    .with_health()
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks(permissions=READONLY_PERMISSIONS)  # Apply read-only permissions, validate_on_startup=True by default
    .on_startup(seed_readonly_tasks)  # Pre-seed tasks
    .build()
)

if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("readonly_task_api:app")
