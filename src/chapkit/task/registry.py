"""Global registry for Python task functions with metadata support."""

import inspect
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from .schemas import TaskInfo


class TaskMetadata(TypedDict):
    """Metadata for a registered task."""

    func: Callable[..., Any]
    tags: list[str]


class TaskRegistry:
    """Global registry for Python task functions with tags and metadata."""

    _registry: dict[str, TaskMetadata] = {}

    @classmethod
    def register(
        cls,
        name: str,
        tags: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a task function with optional tags."""
        # Validate URL-safe name
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Task name '{name}' must be URL-safe (alphanumeric, underscore, hyphen only)")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in cls._registry:
                raise ValueError(f"Task '{name}' already registered")
            cls._registry[name] = {
                "func": func,
                "tags": tags or [],
            }
            return func

        return decorator

    @classmethod
    def register_function(
        cls,
        name: str,
        func: Callable[..., Any],
        tags: list[str] | None = None,
    ) -> None:
        """Imperatively register a task function with optional tags."""
        # Validate URL-safe name
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Task name '{name}' must be URL-safe (alphanumeric, underscore, hyphen only)")

        if name in cls._registry:
            raise ValueError(f"Task '{name}' already registered")
        cls._registry[name] = {
            "func": func,
            "tags": tags or [],
        }

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Retrieve a registered task function."""
        if name not in cls._registry:
            raise KeyError(f"Task '{name}' not found in registry")
        return cls._registry[name]["func"]

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a task is registered."""
        return name in cls._registry

    @classmethod
    def get_tags(cls, name: str) -> list[str]:
        """Get tags for a registered task."""
        if name not in cls._registry:
            raise KeyError(f"Task '{name}' not found in registry")
        return cls._registry[name]["tags"]

    @classmethod
    def get_info(cls, name: str) -> "TaskInfo":
        """Get metadata for a registered task."""
        from .schemas import ParameterInfo, TaskInfo

        if name not in cls._registry:
            raise KeyError(f"Task '{name}' not found in registry")

        metadata = cls._registry[name]
        func = metadata["func"]
        sig = inspect.signature(func)

        # Extract parameter info
        parameters = []
        for param_name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            parameters.append(
                ParameterInfo(
                    name=param_name,
                    annotation=str(param.annotation) if param.annotation != param.empty else None,
                    default=str(param.default) if param.default != param.empty else None,
                    required=param.default == param.empty,
                )
            )

        return TaskInfo(
            name=name,
            docstring=inspect.getdoc(func),
            signature=str(sig),
            parameters=parameters,
            tags=metadata["tags"],
        )

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered task names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_all_info(cls) -> list["TaskInfo"]:
        """List metadata for all registered tasks."""
        return [cls.get_info(name) for name in cls.list_all()]

    @classmethod
    def list_by_tags(cls, tags: list[str]) -> list[str]:
        """List task names that have ALL specified tags."""
        if not tags:
            return cls.list_all()

        matching_tasks = []
        for name, metadata in cls._registry.items():
            task_tags = set(metadata["tags"])
            if all(tag in task_tags for tag in tags):
                matching_tasks.append(name)

        return sorted(matching_tasks)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks (useful for testing and hot-reload)."""
        cls._registry.clear()
