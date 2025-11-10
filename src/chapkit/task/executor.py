"""Task executor for registry-based execution with dependency injection."""

from __future__ import annotations

import asyncio
import inspect
import types
from typing import Any, Union, get_origin, get_type_hints

from servicekit import Database
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit.artifact import ArtifactManager
from chapkit.scheduler import ChapkitScheduler

from .registry import TaskRegistry

# Framework-provided types that can be injected into task functions
INJECTABLE_TYPES = {
    AsyncSession,
    Database,
    ChapkitScheduler,
    ArtifactManager,
}


class TaskExecutor:
    """Executes registered task functions with dependency injection."""

    def __init__(
        self,
        database: Database,
        scheduler: ChapkitScheduler | None = None,
        artifact_manager: ArtifactManager | None = None,
    ) -> None:
        """Initialize task executor with framework dependencies."""
        self.database = database
        self.scheduler = scheduler
        self.artifact_manager = artifact_manager

    async def execute(self, name: str, params: dict[str, Any] | None = None) -> Any:
        """Execute registered function by name with runtime parameters and return result."""
        # Verify function exists
        if not TaskRegistry.has(name):
            raise ValueError(f"Task '{name}' not found in registry")

        # Get function from registry
        func = TaskRegistry.get(name)

        # Create a database session for potential injection
        async with self.database.session() as session:
            # Inject framework dependencies based on function signature
            final_params = self._inject_parameters(func, params or {}, session)

            # Handle sync/async functions
            if inspect.iscoroutinefunction(func):
                result = await func(**final_params)
            else:
                result = await asyncio.to_thread(func, **final_params)

        return result

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
        injection_map: dict[type, Any] = {
            AsyncSession: session,
            Database: self.database,
        }
        # Add optional dependencies if available
        if self.scheduler is not None:
            injection_map[ChapkitScheduler] = self.scheduler
        if self.artifact_manager is not None:
            injection_map[ArtifactManager] = self.artifact_manager
        return injection_map

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
