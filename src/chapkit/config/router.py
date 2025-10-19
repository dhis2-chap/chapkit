"""Config CRUD router with artifact linking operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import Depends, HTTPException, status
from servicekit.api.crud import CrudPermissions, CrudRouter

from chapkit.artifact.schemas import ArtifactOut

from .manager import ConfigManager
from .schemas import BaseConfig, ConfigIn, ConfigOut, LinkArtifactRequest, UnlinkArtifactRequest


class ConfigRouter(CrudRouter[ConfigIn[BaseConfig], ConfigOut[BaseConfig]]):
    """CRUD router for Config entities with artifact linking operations."""

    def __init__(
        self,
        prefix: str,
        tags: Sequence[str],
        manager_factory: Any,
        entity_in_type: type[ConfigIn[BaseConfig]],
        entity_out_type: type[ConfigOut[BaseConfig]],
        permissions: CrudPermissions | None = None,
        enable_artifact_operations: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize config router with entity types and manager factory."""
        self.enable_artifact_operations = enable_artifact_operations
        super().__init__(
            prefix=prefix,
            tags=list(tags),
            entity_in_type=entity_in_type,
            entity_out_type=entity_out_type,
            manager_factory=manager_factory,
            permissions=permissions,
            **kwargs,
        )

    def _register_routes(self) -> None:
        """Register config CRUD routes and artifact linking operations."""
        super()._register_routes()

        if not self.enable_artifact_operations:
            return

        manager_factory = self.manager_factory

        async def link_artifact(
            entity_id: str,
            request: LinkArtifactRequest,
            manager: ConfigManager[BaseConfig] = Depends(manager_factory),
        ) -> None:
            config_id = self._parse_ulid(entity_id)

            try:
                await manager.link_artifact(config_id, request.artifact_id)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

        async def unlink_artifact(
            entity_id: str,
            request: UnlinkArtifactRequest,
            manager: ConfigManager[BaseConfig] = Depends(manager_factory),
        ) -> None:
            try:
                await manager.unlink_artifact(request.artifact_id)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

        async def get_linked_artifacts(
            entity_id: str,
            manager: ConfigManager[BaseConfig] = Depends(manager_factory),
        ) -> list[ArtifactOut]:
            config_id = self._parse_ulid(entity_id)
            return await manager.get_linked_artifacts(config_id)

        self.register_entity_operation(
            "link-artifact",
            link_artifact,
            http_method="POST",
            status_code=status.HTTP_204_NO_CONTENT,
            summary="Link artifact to config",
            description="Link a config to a root artifact (parent_id IS NULL)",
        )

        self.register_entity_operation(
            "unlink-artifact",
            unlink_artifact,
            http_method="POST",
            status_code=status.HTTP_204_NO_CONTENT,
            summary="Unlink artifact from config",
            description="Remove the link between a config and an artifact",
        )

        self.register_entity_operation(
            "artifacts",
            get_linked_artifacts,
            http_method="GET",
            response_model=list[ArtifactOut],
            summary="Get linked artifacts",
            description="Get all root artifacts linked to this config",
        )
