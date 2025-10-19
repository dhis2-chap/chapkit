"""Task repository for database access and querying."""

from __future__ import annotations

from servicekit.repository import BaseRepository
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from .models import Task


class TaskRepository(BaseRepository[Task, ULID]):
    """Repository for Task template entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize task repository with database session."""
        super().__init__(session, Task)

    async def find_by_enabled(self, enabled: bool) -> list[Task]:
        """Find all tasks by enabled status."""
        stmt = select(Task).where(Task.enabled == enabled).order_by(Task.created_at.desc())
        result = await self.s.execute(stmt)
        return list(result.scalars().all())

    async def find_all(self, *, enabled: bool | None = None) -> list[Task]:
        """Find all tasks, optionally filtered by enabled status."""
        if enabled is None:
            result = await super().find_all()
            return list(result)
        return await self.find_by_enabled(enabled)
