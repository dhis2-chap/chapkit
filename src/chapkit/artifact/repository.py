"""Artifact repository for hierarchical data access with tree traversal."""

from __future__ import annotations

from typing import Iterable

from servicekit.repository import BaseRepository
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from ulid import ULID

from .models import Artifact


class ArtifactRepository(BaseRepository[Artifact, ULID]):
    """Repository for Artifact entities with tree traversal operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize artifact repository with database session."""
        super().__init__(session, Artifact)

    async def find_by_id(self, id: ULID) -> Artifact | None:
        """Find an artifact by ID with children eagerly loaded."""
        return await self.s.get(self.model, id, options=[selectinload(self.model.children)])

    async def find_subtree(self, start_id: ULID) -> Iterable[Artifact]:
        """Find all artifacts in the subtree rooted at the given ID using recursive CTE."""
        cte = select(self.model.id).where(self.model.id == start_id).cte(name="descendants", recursive=True)
        cte = cte.union_all(select(self.model.id).where(self.model.parent_id == cte.c.id))

        subtree_ids = (await self.s.scalars(select(cte.c.id))).all()
        rows = (await self.s.scalars(select(self.model).where(self.model.id.in_(subtree_ids)))).all()
        return rows

    async def get_root_artifact(self, artifact_id: ULID) -> Artifact | None:
        """Find the root artifact by traversing up the parent chain."""
        artifact = await self.s.get(self.model, artifact_id)
        if artifact is None:
            return None

        while artifact.parent_id is not None:
            parent = await self.s.get(self.model, artifact.parent_id)
            if parent is None:
                break
            artifact = parent

        return artifact
