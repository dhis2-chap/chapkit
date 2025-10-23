"""Task validation utilities for detecting orphaned Python tasks."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from servicekit import Database

from .manager import TaskManager
from .registry import TaskRegistry
from .repository import TaskRepository
from .schemas import TaskIn

logger = logging.getLogger(__name__)


async def validate_and_disable_orphaned_tasks(app: FastAPI) -> int:
    """Validate Python tasks and disable orphaned ones that reference missing functions."""
    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        logger.debug("No database configured, skipping task validation")
        return 0

    disabled_count = 0

    async with database.session() as session:
        task_repo = TaskRepository(session)
        task_manager = TaskManager(task_repo, scheduler=None, database=None, artifact_manager=None)

        # Get all tasks
        all_tasks = await task_manager.find_all()

        # Get registered function names
        registered_functions = set(TaskRegistry.list_all())

        # Find orphaned Python tasks
        orphaned_tasks = [
            task for task in all_tasks if task.task_type == "python" and task.command not in registered_functions
        ]

        if orphaned_tasks:
            logger.warning(
                "Found orphaned Python tasks - disabling them",
                extra={
                    "count": len(orphaned_tasks),
                    "task_ids": [str(task.id) for task in orphaned_tasks],
                    "commands": [task.command for task in orphaned_tasks],
                },
            )

            # Disable each orphaned task
            for task in orphaned_tasks:
                logger.info(
                    f"Disabling orphaned task {task.id}: function '{task.command}' not found in registry",
                    extra={"task_id": str(task.id), "command": task.command, "task_type": task.task_type},
                )

                # Create TaskIn with enabled=False
                task_type_value = task.task_type if task.task_type in ("shell", "python") else "shell"
                task_in = TaskIn(
                    id=task.id,
                    command=task.command,
                    task_type=task_type_value,  # type: ignore[arg-type]
                    parameters=task.parameters,
                    enabled=False,
                )
                await task_manager.save(task_in)
                disabled_count += 1

    if disabled_count > 0:
        logger.warning(f"Disabled {disabled_count} orphaned Python task(s)")
    else:
        logger.debug("No orphaned Python tasks found")

    return disabled_count
