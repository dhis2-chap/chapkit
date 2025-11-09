"""Task schemas for registry-based execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParameterInfo(BaseModel):
    """Function parameter metadata."""

    name: str = Field(description="Parameter name")
    annotation: str | None = Field(default=None, description="Type annotation as string")
    default: str | None = Field(default=None, description="Default value as string")
    required: bool = Field(description="Whether parameter is required")


class TaskInfo(BaseModel):
    """Task metadata from registry."""

    name: str = Field(description="Task name (URL-safe)")
    docstring: str | None = Field(default=None, description="Function docstring")
    signature: str = Field(description="Function signature")
    parameters: list[ParameterInfo] = Field(default_factory=list, description="Function parameters")
    tags: list[str] = Field(default_factory=list, description="Task tags for filtering")


class TaskExecuteRequest(BaseModel):
    """Request to execute a task."""

    params: dict[str, Any] | None = Field(default=None, description="Runtime parameters for task execution")


class TaskExecuteResponse(BaseModel):
    """Response from task execution."""

    task_name: str = Field(description="Name of the executed task")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters used for execution")
    result: Any = Field(description="Task execution result")
    error: dict[str, str] | None = Field(default=None, description="Error information if execution failed")
