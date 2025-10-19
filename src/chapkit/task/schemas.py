"""Task schemas for reusable command templates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from servicekit.schemas import EntityIn, EntityOut


class TaskIn(EntityIn):
    """Input schema for creating or updating task templates."""

    command: str = Field(description="Shell command or Python function name to execute")
    task_type: Literal["shell", "python"] = Field(default="shell", description="Type of task: 'shell' or 'python'")
    parameters: dict[str, Any] | None = Field(
        default=None, description="Parameters to pass to Python function (ignored for shell tasks)"
    )
    enabled: bool = Field(default=True, description="Whether task is enabled for execution")


class TaskOut(EntityOut):
    """Output schema for task template entities."""

    command: str = Field(description="Shell command or Python function name to execute")
    task_type: str = Field(description="Type of task: 'shell' or 'python'")
    parameters: dict[str, Any] | None = Field(default=None, description="Parameters to pass to Python function")
    enabled: bool = Field(description="Whether task is enabled for execution")
