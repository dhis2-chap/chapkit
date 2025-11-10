"""Artifact CRUD router with hierarchical tree operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import Depends, HTTPException, Response, status
from servicekit.api.crud import CrudPermissions, CrudRouter

from ..config.schemas import BaseConfig, ConfigOut
from ..data import DataFrame
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
        # Store enable_config_access to conditionally register config endpoint
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

        # Conditionally register config access endpoint
        if self.enable_config_access:
            from ..api.dependencies import get_config_manager
            from ..config.manager import ConfigManager

            async def get_config(
                entity_id: str,
                artifact_manager: ArtifactManager = Depends(manager_factory),
                config_manager: ConfigManager[BaseConfig] = Depends(get_config_manager),
            ) -> ConfigOut[BaseConfig]:
                """Get the config linked to this artifact."""
                ulid_id = self._parse_ulid(entity_id)

                # Get config by traversing to root artifact
                config = await config_manager.get_config_for_artifact(
                    artifact_id=ulid_id, artifact_repo=artifact_manager.repository
                )

                if config is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No config linked to artifact {entity_id}",
                    )

                return config

            self.register_entity_operation(
                "config",
                get_config,
                response_model=ConfigOut[BaseConfig],
                summary="Get artifact config",
                description="Get configuration linked to this artifact by traversing to root",
            )

        # Download endpoint
        async def download_artifact(
            entity_id: str,
            manager: ArtifactManager = Depends(manager_factory),
        ) -> Response:
            """Download artifact content as binary file."""
            ulid_id = self._parse_ulid(entity_id)

            artifact = await manager.find_by_id(ulid_id)
            if artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Artifact with id {entity_id} not found",
                )

            if not isinstance(artifact.data, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Artifact has no downloadable content",
                )

            content = artifact.data.get("content")
            if content is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Artifact has no content",
                )

            content_type = artifact.data.get("content_type", "application/octet-stream")

            # Serialize content to bytes based on type
            if isinstance(content, bytes):
                # Most common case: ZIP files, PNG images, etc.
                binary = content
            elif isinstance(content, DataFrame):
                # Serialize DataFrame based on content_type
                if content_type == "text/csv":
                    csv_string = content.to_csv()
                    binary = csv_string.encode() if csv_string else b""
                else:
                    # Default to JSON for all other types
                    binary = content.to_json().encode()
            elif isinstance(content, dict):
                # DataFrame serialized to dict in database - reconstruct and serialize
                if content_type == "application/x-pandas-dataframe":
                    df = DataFrame.model_validate(content)
                    binary = df.to_json().encode()
                else:
                    # Generic dict content - serialize to JSON
                    import json

                    binary = json.dumps(content).encode()
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot serialize content of type {type(content).__name__}",
                )

            # Determine filename extension
            extension_map = {
                "application/zip": "zip",
                "text/csv": "csv",
                "application/json": "json",
                "application/x-pandas-dataframe": "json",
                "image/png": "png",
            }
            ext = extension_map.get(content_type, "bin")

            return Response(
                content=binary,
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename=artifact_{entity_id}.{ext}"},
            )

        # Metadata endpoint
        async def get_artifact_metadata(
            entity_id: str,
            manager: ArtifactManager = Depends(manager_factory),
        ) -> dict[str, Any]:
            """Get only JSON-serializable metadata, excluding binary content."""
            ulid_id = self._parse_ulid(entity_id)

            artifact = await manager.find_by_id(ulid_id)
            if artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Artifact with id {entity_id} not found",
                )

            if not isinstance(artifact.data, dict):
                return {}

            return artifact.data.get("metadata", {})

        self.register_entity_operation(
            "download",
            download_artifact,
            response_model=None,  # Raw Response, don't serialize
            summary="Download artifact content",
            description="Download artifact content as binary file (ZIP, CSV, etc.)",
        )

        self.register_entity_operation(
            "metadata",
            get_artifact_metadata,
            response_model=dict[str, Any],
            summary="Get artifact metadata",
            description="Get JSON-serializable metadata without binary content",
        )
