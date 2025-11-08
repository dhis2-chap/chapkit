"""Task router for registry-based execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

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
        async def list_tasks(
            tags: str | None = Query(None, description="Comma-separated tags to filter by (requires ALL tags)"),
        ) -> list[TaskInfo]:
            """List all registered tasks, optionally filtered by tags."""
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",")]
                matching_names = TaskRegistry.list_by_tags(tag_list)
                return [TaskRegistry.get_info(name) for name in matching_names]
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

        @self.router.post("/{name}/$execute", response_model=TaskExecuteResponse, status_code=status.HTTP_202_ACCEPTED)
        async def execute_task(
            name: str,
            request: TaskExecuteRequest = TaskExecuteRequest(),
            executor: TaskExecutor = Depends(executor_factory),
        ) -> TaskExecuteResponse:
            """Execute task by name with runtime parameters."""
            try:
                job_id = await executor.execute(name, request.params)
                return TaskExecuteResponse(
                    job_id=str(job_id),
                    message=f"Task '{name}' submitted for execution. Job ID: {job_id}",
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e),
                ) from e
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Task execution failed: {str(e)}",
                ) from e
