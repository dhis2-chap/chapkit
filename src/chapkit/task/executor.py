"""Task executor for registry-based execution with dependency injection."""

from __future__ import annotations

import asyncio
import inspect
import traceback
import types
from typing import Any, Union, get_origin, get_type_hints

from servicekit import Database
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitJobScheduler

from .registry import TaskRegistry

# Framework-provided types that can be injected into task functions
INJECTABLE_TYPES = {
    AsyncSession,
    Database,
    ArtifactManager,
    ChapkitJobScheduler,
}


class TaskExecutor:
    """Executes registered task functions with dependency injection."""

    def __init__(
        self,
        scheduler: ChapkitJobScheduler,
        database: Database,
        artifact_manager: ArtifactManager,
    ) -> None:
        """Initialize task executor with framework dependencies."""
        self.scheduler = scheduler
        self.database = database
        self.artifact_manager = artifact_manager

    async def execute(self, name: str, params: dict[str, Any] | None = None) -> ULID:
        """Execute registered function by name with runtime parameters."""
        # Verify function exists
        if not TaskRegistry.has(name):
            raise ValueError(f"Task '{name}' not found in registry")

        # Submit to scheduler
        job_id = await self.scheduler.add_job(self._execute_task, name, params or {})
        return job_id

    def _is_injectable_type(self, param_type: type | None) -> bool:
        """Check if a parameter type should be injected by the framework."""
        if param_type is None:
            return False

        # Handle Optional[Type] -> extract the non-None type
        origin = get_origin(param_type)
        if origin is types.UnionType or origin is Union:
            args = getattr(param_type, "__args__", ())
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                param_type = non_none_types[0]

        return param_type in INJECTABLE_TYPES

    def _build_injection_map(self, session: AsyncSession | None) -> dict[type, Any]:
        """Build map of injectable types to their instances."""
        return {
            AsyncSession: session,
            Database: self.database,
            ArtifactManager: self.artifact_manager,
            ChapkitJobScheduler: self.scheduler,
        }

    def _inject_parameters(
        self,
        func: Any,
        user_params: dict[str, Any],
        session: AsyncSession | None,
    ) -> dict[str, Any]:
        """Merge user parameters with framework injections based on function signature."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build injection map
        injection_map = self._build_injection_map(session)

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
                    f"Missing required parameter '{param_name}' for task '{func.__name__}'. "
                    f"Parameter is not injectable and not provided in params."
                )

        return final_params

    async def _execute_task(self, name: str, params: dict[str, Any]) -> ULID:
        """Execute task function and return artifact_id containing results."""
        # Create a database session for potential injection
        session_context = self.database.session()
        session = await session_context.__aenter__()

        try:
            # Get function from registry
            func = TaskRegistry.get(name)

            # Execute function with type-based injection
            result_data: dict[str, Any]
            try:
                # Inject framework dependencies based on function signature
                final_params = self._inject_parameters(func, params, session)

                # Handle sync/async functions
                if inspect.iscoroutinefunction(func):
                    result = await func(**final_params)
                else:
                    result = await asyncio.to_thread(func, **final_params)

                result_data = {
                    "task_name": name,
                    "params": params,
                    "result": result,
                    "error": None,
                }
            except Exception as e:
                result_data = {
                    "task_name": name,
                    "params": params,
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
