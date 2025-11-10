"""Task router for registry-based execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from .executor import TaskExecutor
from .registry import TaskRegistry
from .schemas import TaskExecuteRequest, TaskExecuteResponse, TaskInfo


class TaskRouter:
    """Router for task execution (registry-based, no CRUD)."""

    def __init__(
        self,
        prefix: str,
        tags: Sequence[str],
        executor_factory: Any,
    ) -> None:
        """Initialize task router with executor factory."""
        self.prefix = prefix
        self.tags = tags
        self.executor_factory = executor_factory
        self.router = APIRouter(prefix=prefix, tags=list(tags))
        self._register_routes()

    @classmethod
    def create(
        cls,
        prefix: str,
        tags: Sequence[str],
        executor_factory: Any,
    ) -> TaskRouter:
        """Create a task router with executor factory."""
        return cls(prefix=prefix, tags=tags, executor_factory=executor_factory)

    def _register_routes(self) -> None:
        """Register task routes."""
        executor_factory = self.executor_factory

        @self.router.get("", response_model=list[TaskInfo])
        async def list_tasks() -> list[TaskInfo]:
            """List all registered tasks."""
            return TaskRegistry.list_all_info()

        @self.router.get("/{name}", response_model=TaskInfo)
        async def get_task(name: str) -> TaskInfo:
            """Get task metadata by name."""
            try:
                return TaskRegistry.get_info(name)
            except KeyError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e),
                ) from e

        @self.router.post("/{name}/$execute", response_model=TaskExecuteResponse)
        async def execute_task(
            name: str,
            request: TaskExecuteRequest = TaskExecuteRequest(),
            executor: TaskExecutor = Depends(executor_factory),
        ) -> TaskExecuteResponse:
            """Execute task by name with runtime parameters and return result."""
            import traceback

            # Check if task exists
            if not TaskRegistry.has(name):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task '{name}' not found in registry",
                )

            params = request.params or {}

            # Execute task and handle errors
            try:
                result = await executor.execute(name, params)
                return TaskExecuteResponse(
                    task_name=name,
                    params=params,
                    result=result,
                    error=None,
                )
            except Exception as e:
                # Return error in response (don't raise exception)
                return TaskExecuteResponse(
                    task_name=name,
                    params=params,
                    result=None,
                    error={
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
