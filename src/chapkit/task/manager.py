"""Task manager for reusable command templates with artifact-based execution results."""

from __future__ import annotations

import asyncio
import inspect
import traceback
import types
from typing import Any, Union, get_origin, get_type_hints

from servicekit import Database
from servicekit.manager import BaseManager
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitJobScheduler

from .models import Task
from .registry import TaskRegistry
from .repository import TaskRepository
from .schemas import TaskIn, TaskOut

# Framework-provided types that can be injected into Python task functions
INJECTABLE_TYPES = {
    AsyncSession,
    Database,
    ArtifactManager,
    ChapkitJobScheduler,
}


class TaskManager(BaseManager[Task, TaskIn, TaskOut, ULID]):
    """Manager for Task template entities with artifact-based execution."""

    def __init__(
        self,
        repo: TaskRepository,
        scheduler: ChapkitJobScheduler | None = None,
        database: Database | None = None,
        artifact_manager: ArtifactManager | None = None,
    ) -> None:
        """Initialize task manager with repository, scheduler, database, and artifact manager."""
        super().__init__(repo, Task, TaskOut)
        self.repository: TaskRepository = repo
        self.scheduler = scheduler
        self.database = database
        self.artifact_manager = artifact_manager

    async def find_all(self, *, enabled: bool | None = None) -> list[TaskOut]:
        """Find all tasks, optionally filtered by enabled status."""
        tasks = await self.repository.find_all(enabled=enabled)
        return [self._to_output_schema(task) for task in tasks]

    def _is_injectable_type(self, param_type: type | None) -> bool:
        """Check if a parameter type should be injected by the framework."""
        if param_type is None:
            return False

        # Handle Optional[Type] -> extract the non-None type
        origin = get_origin(param_type)
        if origin is types.UnionType or origin is Union:  # Union type (both syntaxes)
            # For Optional types, we still want to inject if the non-None type is injectable
            # This allows Optional[AsyncSession] to work
            args = getattr(param_type, "__args__", ())
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                param_type = non_none_types[0]

        # Check if type is in injectable set
        return param_type in INJECTABLE_TYPES

    def _build_injection_map(self, task_id: ULID, session: AsyncSession | None) -> dict[type, Any]:
        """Build map of injectable types to their instances."""
        return {
            AsyncSession: session,
            Database: self.database,
            ArtifactManager: self.artifact_manager,
            ChapkitJobScheduler: self.scheduler,
        }

    def _inject_parameters(
        self, func: Any, user_params: dict[str, Any], task_id: ULID, session: AsyncSession | None
    ) -> dict[str, Any]:
        """Merge user parameters with framework injections based on function signature."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build injection map
        injection_map = self._build_injection_map(task_id, session)

        # Start with user parameters
        final_params = dict(user_params)

        # Inspect each parameter in function signature
        for param_name, param in sig.parameters.items():
            # Skip self, *args, **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get type hint for this parameter
            param_type = type_hints.get(param_name)

            # Check if this type should be injected
            if self._is_injectable_type(param_type):
                # Get the actual type (handle Optional)
                actual_type = param_type
                origin = get_origin(param_type)
                if origin is types.UnionType or origin is Union:
                    args = getattr(param_type, "__args__", ())
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if non_none_types:
                        actual_type = non_none_types[0]

                # Inject if we have an instance of this type
                if actual_type in injection_map:
                    injectable_value = injection_map[actual_type]
                    # For required parameters, inject even if None
                    # For optional parameters, only inject if not None
                    if param.default is param.empty:
                        # Required parameter - inject whatever we have (even None)
                        final_params[param_name] = injectable_value
                    elif injectable_value is not None:
                        # Optional parameter - only inject if we have a value
                        final_params[param_name] = injectable_value
                continue

            # Not injectable - must come from user parameters
            if param_name not in final_params:
                # Check if parameter has a default value
                if param.default is not param.empty:
                    continue  # Will use default

                # Required parameter missing
                raise ValueError(
                    f"Missing required parameter '{param_name}' for task function. "
                    f"Parameter is not injectable and not provided in task.parameters."
                )

        return final_params

    async def execute_task(self, task_id: ULID) -> ULID:
        """Execute a task by submitting it to the scheduler and return the job ID."""
        if self.scheduler is None:
            raise ValueError("Task execution requires a scheduler. Use ServiceBuilder.with_jobs() to enable.")

        if self.artifact_manager is None:
            raise ValueError(
                "Task execution requires artifacts. Use ServiceBuilder.with_artifacts() before with_tasks()."
            )

        task = await self.repository.find_by_id(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        # Check if task is enabled
        if not task.enabled:
            raise ValueError(f"Cannot execute disabled task {task_id}")

        # Route based on task type
        if task.task_type == "python":
            job_id = await self.scheduler.add_job(self._execute_python, task_id)
        else:  # shell
            job_id = await self.scheduler.add_job(self._execute_command, task_id)

        return job_id

    async def _execute_command(self, task_id: ULID) -> ULID:
        """Execute command and return artifact_id containing results."""
        if self.database is None:
            raise RuntimeError("Database instance required for task execution")

        if self.artifact_manager is None:
            raise RuntimeError("ArtifactManager instance required for task execution")

        # Fetch task and serialize snapshot before execution
        async with self.database.session() as session:
            task_repo = TaskRepository(session)
            task = await task_repo.find_by_id(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")

            # Capture task snapshot
            task_snapshot = {
                "id": str(task.id),
                "command": task.command,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
            }

        # Execute command using asyncio subprocess
        process = await asyncio.create_subprocess_shell(
            task.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for completion and capture output
        stdout_bytes, stderr_bytes = await process.communicate()

        # Decode outputs
        stdout_text = stdout_bytes.decode("utf-8") if stdout_bytes else ""
        stderr_text = stderr_bytes.decode("utf-8") if stderr_bytes else ""

        # Create artifact with execution results
        result_data: dict[str, Any] = {
            "task": task_snapshot,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exit_code": process.returncode,
        }

        async with self.database.session() as session:
            artifact_repo = ArtifactRepository(session)
            artifact_mgr = ArtifactManager(artifact_repo)

            artifact_out = await artifact_mgr.save(
                ArtifactIn(
                    data=result_data,
                    parent_id=None,
                )
            )

        return artifact_out.id

    async def _execute_python(self, task_id: ULID) -> ULID:
        """Execute Python function and return artifact_id containing results."""
        if self.database is None:
            raise RuntimeError("Database instance required for task execution")

        if self.artifact_manager is None:
            raise RuntimeError("ArtifactManager instance required for task execution")

        # Create a database session for potential injection
        session_context = self.database.session()
        session = await session_context.__aenter__()

        try:
            # Fetch task and serialize snapshot
            task_repo = TaskRepository(session)
            task = await task_repo.find_by_id(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")

            # Capture task snapshot
            task_snapshot = {
                "id": str(task.id),
                "command": task.command,
                "task_type": task.task_type,
                "parameters": task.parameters,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
            }

            # Get function from registry
            try:
                func = TaskRegistry.get(task.command)
            except KeyError:
                raise ValueError(f"Python function '{task.command}' not found in registry")

            # Execute function with type-based injection
            result_data: dict[str, Any]
            try:
                user_params = task.parameters or {}

                # Inject framework dependencies based on function signature
                final_params = self._inject_parameters(func, user_params, task_id, session)

                # Handle sync/async functions
                if inspect.iscoroutinefunction(func):
                    result = await func(**final_params)
                else:
                    result = await asyncio.to_thread(func, **final_params)

                result_data = {
                    "task": task_snapshot,
                    "result": result,
                    "error": None,
                }
            except Exception as e:
                result_data = {
                    "task": task_snapshot,
                    "result": None,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }
        finally:
            # Always close the session
            await session_context.__aexit__(None, None, None)

        # Create artifact (with a new session)
        async with self.database.session() as artifact_session:
            artifact_repo = ArtifactRepository(artifact_session)
            artifact_mgr = ArtifactManager(artifact_repo)
            artifact_out = await artifact_mgr.save(ArtifactIn(data=result_data, parent_id=None))

        return artifact_out.id
