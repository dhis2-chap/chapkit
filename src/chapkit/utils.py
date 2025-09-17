import asyncio
import inspect
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def make_awaitable(func: Callable[..., T | Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    if inspect.iscoroutinefunction(func):
        return func

    async def wrapper(*args, **kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper
