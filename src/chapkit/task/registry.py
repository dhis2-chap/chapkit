"""Global registry for Python task functions."""

from collections.abc import Callable
from typing import Any


class TaskRegistry:
    """Global registry for Python task functions."""

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a task function with support for type-based dependency injection."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in cls._registry:
                raise ValueError(f"Task '{name}' already registered")
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def register_function(cls, name: str, func: Callable[..., Any]) -> None:
        """Imperatively register a task function."""
        if name in cls._registry:
            raise ValueError(f"Task '{name}' already registered")
        cls._registry[name] = func

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Retrieve a registered task function."""
        if name not in cls._registry:
            raise KeyError(f"Task '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered task names."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks (useful for testing)."""
        cls._registry.clear()
