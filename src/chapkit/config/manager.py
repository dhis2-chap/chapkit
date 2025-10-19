"""Config manager for CRUD operations and artifact linking."""

from __future__ import annotations

from servicekit.manager import BaseManager
from ulid import ULID

from chapkit.artifact.repository import ArtifactRepository
from chapkit.artifact.schemas import ArtifactOut

from .models import Config
from .repository import ConfigRepository
from .schemas import BaseConfig, ConfigIn, ConfigOut


class ConfigManager[DataT: BaseConfig](BaseManager[Config, ConfigIn[DataT], ConfigOut[DataT], ULID]):
    """Manager for Config entities with artifact linking operations."""

    def __init__(self, repo: ConfigRepository, data_cls: type[DataT]) -> None:
        """Initialize config manager with repository and data class."""
        super().__init__(repo, Config, ConfigOut)
        self.repository: ConfigRepository = repo
        self.data_cls = data_cls

    async def find_by_name(self, name: str) -> ConfigOut[DataT] | None:
        """Find a config by its unique name."""
        config = await self.repository.find_by_name(name)
        if config:
            return self._to_output_schema(config)
        return None

    async def link_artifact(self, config_id: ULID, artifact_id: ULID) -> None:
        """Link a config to a root artifact."""
        await self.repository.link_artifact(config_id, artifact_id)
        await self.repository.commit()

    async def unlink_artifact(self, artifact_id: ULID) -> None:
        """Unlink an artifact from its config."""
        await self.repository.unlink_artifact(artifact_id)
        await self.repository.commit()

    async def get_config_for_artifact(
        self, artifact_id: ULID, artifact_repo: ArtifactRepository
    ) -> ConfigOut[DataT] | None:
        """Get the config for an artifact by traversing to its root."""
        root = await artifact_repo.get_root_artifact(artifact_id)
        if root is None:
            return None

        config = await self.repository.find_by_root_artifact_id(root.id)
        if config is None:
            return None

        return self._to_output_schema(config)

    async def get_linked_artifacts(self, config_id: ULID) -> list[ArtifactOut]:
        """Get all root artifacts linked to a config."""
        artifacts = await self.repository.find_artifacts_for_config(config_id)
        return [ArtifactOut.model_validate(artifact, from_attributes=True) for artifact in artifacts]

    def _to_output_schema(self, entity: Config) -> ConfigOut[DataT]:
        """Convert ORM entity to output schema with proper data class validation."""
        return ConfigOut[DataT].model_validate(entity, from_attributes=True, context={"data_cls": self.data_cls})
