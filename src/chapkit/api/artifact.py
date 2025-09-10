from uuid import UUID
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from chapkit.api.types import ChapApi
from chapkit.database import ChapDatabase
from chapkit.types import TChapConfig, ArtifactInfo


class ArtifactApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        database: ChapDatabase[TChapConfig],
    ) -> None:
        self._database = database

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["artifacts"])

        async def get_artifacts_for_config(config_id: UUID) -> list[ArtifactInfo]:
            """Get all artifacts linked to a specific config."""
            # First check if config exists
            config = self._database.get_config(config_id)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config {config_id} not found")

            # Get artifacts for this config
            artifacts = self._database.get_artifacts_for_config(config_id)

            return [
                ArtifactInfo(id=artifact_id, config_id=config_id, config_name=config.name)
                for artifact_id, _ in artifacts
            ]

        async def get_artifact(artifact_id: UUID) -> ArtifactInfo:
            """Get a specific artifact by ID."""
            artifact = self._database.get_artifact(artifact_id)
            if artifact is None:
                raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

            config = self._database.get_config_for_artifact(artifact_id)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config for artifact {artifact_id} not found")

            try:
                jsonable_data = jsonable_encoder(artifact)
            except (TypeError, ValueError):
                jsonable_data = None

            return ArtifactInfo(id=artifact_id, config_id=config.id, config_name=config.name, data=jsonable_data)

        async def delete_artifact(artifact_id: UUID) -> None:
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
            name="delete_artifact",
            summary="Delete an artifact by ID",
            responses={404: {"description": "Artifact not found"}},
        )

        return router
