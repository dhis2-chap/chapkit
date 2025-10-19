"""Artifact CRUD router with hierarchical tree operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import Depends, HTTPException, status
from servicekit.api.crud import CrudPermissions, CrudRouter

from .manager import ArtifactManager
from .schemas import ArtifactIn, ArtifactOut, ArtifactTreeNode


class ArtifactRouter(CrudRouter[ArtifactIn, ArtifactOut]):
    """CRUD router for Artifact entities with tree operations."""

    def __init__(
        self,
        prefix: str,
        tags: Sequence[str],
        manager_factory: Any,
        entity_in_type: type[ArtifactIn],
        entity_out_type: type[ArtifactOut],
        permissions: CrudPermissions | None = None,
        enable_config_access: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize artifact router with entity types and manager factory."""
        # Store enable_config_access for potential future use
        self.enable_config_access = enable_config_access

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
        """Register artifact CRUD routes and tree operations."""
        super()._register_routes()

        manager_factory = self.manager_factory

        async def expand_artifact(
            entity_id: str,
            manager: ArtifactManager = Depends(manager_factory),
        ) -> ArtifactTreeNode:
            ulid_id = self._parse_ulid(entity_id)

            expanded = await manager.expand_artifact(ulid_id)
            if expanded is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Artifact with id {entity_id} not found",
                )
            return expanded

        async def build_tree(
            entity_id: str,
            manager: ArtifactManager = Depends(manager_factory),
        ) -> ArtifactTreeNode:
            ulid_id = self._parse_ulid(entity_id)

            tree = await manager.build_tree(ulid_id)
            if tree is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Artifact with id {entity_id} not found",
                )
            return tree

        self.register_entity_operation(
            "expand",
            expand_artifact,
            response_model=ArtifactTreeNode,
            summary="Expand artifact",
            description="Get artifact with hierarchy metadata but without children",
        )

        self.register_entity_operation(
            "tree",
            build_tree,
            response_model=ArtifactTreeNode,
            summary="Build artifact tree",
            description="Build hierarchical tree structure rooted at the given artifact",
        )
