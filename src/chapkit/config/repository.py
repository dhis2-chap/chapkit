"""Config repository for database access and artifact linking."""

from __future__ import annotations

from servicekit.repository import BaseRepository
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from chapkit.artifact.models import Artifact

from .models import Config, ConfigArtifact


class ConfigRepository(BaseRepository[Config, ULID]):
    """Repository for Config entities with artifact linking operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize config repository with database session."""
        super().__init__(session, Config)

    async def find_by_name(self, name: str) -> Config | None:
        """Find a config by its unique name."""
        result = await self.s.scalars(select(self.model).where(self.model.name == name))
        return result.one_or_none()

    async def link_artifact(self, config_id: ULID, artifact_id: ULID) -> None:
        """Link a config to a root artifact."""
        artifact = await self.s.get(Artifact, artifact_id)
        if artifact is None:
            raise ValueError(f"Artifact {artifact_id} not found")
        if artifact.parent_id is not None:
            raise ValueError(f"Artifact {artifact_id} is not a root artifact (parent_id={artifact.parent_id})")

        link = ConfigArtifact(config_id=config_id, artifact_id=artifact_id)
        self.s.add(link)

    async def unlink_artifact(self, artifact_id: ULID) -> None:
        """Unlink an artifact from its config."""
        stmt = sql_delete(ConfigArtifact).where(ConfigArtifact.artifact_id == artifact_id)
        await self.s.execute(stmt)

    async def delete_by_id(self, id: ULID) -> None:
        """Delete a config and cascade delete all linked artifact trees."""
        from chapkit.artifact.repository import ArtifactRepository

        linked_artifacts = await self.find_artifacts_for_config(id)

        artifact_repo = ArtifactRepository(self.s)
        for root_artifact in linked_artifacts:
            subtree = await artifact_repo.find_subtree(root_artifact.id)
            for artifact in subtree:
                await self.s.delete(artifact)

        await super().delete_by_id(id)

    async def find_by_root_artifact_id(self, artifact_id: ULID) -> Config | None:
        """Find the config linked to a root artifact."""
        stmt = (
            select(Config)
            .join(ConfigArtifact, Config.id == ConfigArtifact.config_id)
            .where(ConfigArtifact.artifact_id == artifact_id)
        )
        result = await self.s.scalars(stmt)
        return result.one_or_none()

    async def find_artifacts_for_config(self, config_id: ULID) -> list[Artifact]:
        """Find all root artifacts linked to a config."""
        stmt = (
            select(Artifact)
            .join(ConfigArtifact, Artifact.id == ConfigArtifact.artifact_id)
            .where(ConfigArtifact.config_id == config_id)
        )
        result = await self.s.scalars(stmt)
        return list(result.all())
