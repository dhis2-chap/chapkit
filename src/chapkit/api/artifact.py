from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from ulid import ULID

from chapkit.api.types import ChapApi
from chapkit.database import ChapDatabase
from chapkit.types import TChapConfig, ArtifactInfo, ArtifactTree

if TYPE_CHECKING:
    from chapkit.service import ChapService


class ArtifactApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        database: ChapDatabase[TChapConfig],
        service: "ChapService[TChapConfig]",
    ) -> None:
        self._database = database
        self._service = service

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["artifacts"])

        async def get_artifact_tree_for_config(config_id: ULID) -> list[ArtifactTree]:
            """Get all artifacts linked to a specific config as a tree."""
            # First check if config exists
            config = self._database.get_config(config_id)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config {config_id} not found")

            # Get all artifacts for this config
            artifact_rows = self._database.get_artifact_rows_for_config(config_id)

            nodes = {}
            for row in artifact_rows:
                level = 0
                parent_id = row.parent_id
                while parent_id:
                    level += 1
                    parent = next((r for r in artifact_rows if r.id == parent_id), None)
                    parent_id = parent.parent_id if parent else None

                level_name = (
                    self._service.artifact_level_names[level]
                    if level < len(self._service.artifact_level_names)
                    else None
                )
                try:
                    jsonable_data = jsonable_encoder(row.data)
                except (TypeError, ValueError):
                    jsonable_data = None
                nodes[row.id] = ArtifactTree(
                    id=row.id,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    config_id=config.id,
                    config_name=config.name,
                    artifact_level_name=level_name,
                    data=jsonable_data,
                )

            for row in artifact_rows:
                if row.parent_id:
                    parent = nodes.get(row.parent_id)
                    if parent:
                        parent.children.append(nodes[row.id])

            return [
                node for node in nodes.values() if node.id in {row.id for row in artifact_rows if not row.parent_id}
            ]

        async def get_artifacts_for_config(config_id: ULID) -> list[ArtifactInfo]:
            """Get all artifacts linked to a specific config."""
            # First check if config exists
            config = self._database.get_config(config_id)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config {config_id} not found")

            # Get artifacts for this config
            artifact_rows = self._database.get_artifact_rows_for_config(config_id)

            results = []
            for row in artifact_rows:
                level = 0
                parent_id = row.parent_id
                while parent_id:
                    level += 1
                    parent = next((r for r in artifact_rows if r.id == parent_id), None)
                    parent_id = parent.parent_id if parent else None

                level_name = (
                    self._service.artifact_level_names[level]
                    if level < len(self._service.artifact_level_names)
                    else None
                )
                try:
                    jsonable_data = jsonable_encoder(row.data)
                except (TypeError, ValueError):
                    jsonable_data = None
                results.append(
                    ArtifactInfo(
                        id=row.id,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                        config_id=config_id,
                        config_name=config.name,
                        artifact_level_name=level_name,
                        data=jsonable_data,
                    )
                )
            return results

        async def get_artifact(artifact_id: ULID) -> ArtifactInfo:
            """Get a specific artifact by ID."""
            artifact_row = self._database.get_artifact_row(artifact_id)
            if artifact_row is None:
                raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

            config = self._database.get_config_for_artifact(artifact_id)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config for artifact {artifact_id} not found")

            try:
                jsonable_data = jsonable_encoder(artifact_row.data)
            except (TypeError, ValueError):
                jsonable_data = None

            return ArtifactInfo(
                id=artifact_row.id,
                created_at=artifact_row.created_at,
                updated_at=artifact_row.updated_at,
                config_id=config.id,
                config_name=config.name,
                data=jsonable_data,
            )

        async def delete_artifact(artifact_id: ULID) -> None:
            """Delete a specific artifact by ID."""
            artifact = self._database.get_artifact(artifact_id)
            if artifact is None:
                raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
            self._database.delete_artifact(artifact_id)

        router.add_api_route(
            path="/artifacts/config/{config_id}",
            endpoint=get_artifacts_for_config,
            methods=["GET"],
            response_model=list[ArtifactInfo],
            name="get_artifacts_for_config",
            summary="Get all artifacts linked to a config",
            responses={404: {"description": "Config not found"}},
        )

        router.add_api_route(
            path="/artifacts/tree",
            endpoint=get_artifact_tree_for_config,
            methods=["GET"],
            response_model=list[ArtifactTree],
            name="get_artifact_tree_for_config",
            summary="Get all artifacts linked to a config as a tree",
            responses={404: {"description": "Config not found"}},
        )

        router.add_api_route(
            path="/artifacts/{artifact_id}",
            endpoint=get_artifact,
            methods=["GET"],
            response_model=ArtifactInfo,
            name="get_artifact",
            summary="Get an artifact by ID",
            responses={404: {"description": "Artifact not found"}},
        )

        router.add_api_route(
            path="/artifacts/{artifact_id}",
            endpoint=delete_artifact,
            methods=["DELETE"],
            status_code=204,
            response_model=None,
            name="delete_artifact",
            summary="Delete an artifact by ID",
            responses={404: {"description": "Artifact not found"}},
        )

        return router
