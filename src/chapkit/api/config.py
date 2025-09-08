from uuid import UUID

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, Response

from chapkit.api.types import ChapApi
from chapkit.database import ChapDatabase
from chapkit.types import TChapConfig


class ConfigApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        database: ChapDatabase[TChapConfig],
        model_type: type[TChapConfig],
    ) -> None:
        self._database = database
        self._model_type = model_type

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["configs"])

        Model = self._model_type  # concrete Pydantic class

        async def get_config(id: UUID):
            cfg = self._database.get_config(id)

            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Config {id} not found")

            return cfg

        async def get_configs():
            return self._database.get_configs()

        async def get_schema():
            return Model.model_json_schema()

        async def add_config(cfg: dict = Body(...)):
            validated = Model.model_validate(cfg)
            self._database.add_config(validated)

            return JSONResponse(
                status_code=201,
                content=validated.model_dump(mode="json"),
                headers={"Location": f"/configs/{validated.id}"},
            )

        async def update_config(id: UUID, cfg: dict = Body(...)):
            validated = Model.model_validate(cfg)

            if id != validated.id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Config ID in path ({id}) does not match ID in body ({validated.id})",
                )

            self._database.add_config(validated)

            return validated

        async def delete_config(id: UUID):
            if not self._database.del_config(id):
                raise HTTPException(status_code=404, detail=f"Config {id} not found")

            return Response(status_code=204)

        router.add_api_route(
            path="/configs",
            endpoint=get_configs,
            methods=["GET"],
            response_model=list[Model],  # OK to pass the actual class here
            name="list_configs",
            summary="List all configs",
        )

        router.add_api_route(
            path="/configs/schema",
            endpoint=get_schema,
            methods=["GET"],
            response_model=dict,
            responses={200: {"content": {"application/schema+json": {}}}},
            name="get_config_schema",
            summary="Get JSON Schema for the current config model",
        )

        router.add_api_route(
            path="/configs/{id}",
            endpoint=get_config,
            methods=["GET"],
            response_model=Model,
            name="get_config",
            summary="Get a config by ID",
            responses={404: {"description": "Config not found"}},
        )

        router.add_api_route(
            path="/configs",
            endpoint=add_config,
            methods=["POST"],
            response_model=Model,
            status_code=201,
            name="create_config",
            summary="Create (or replace) a config",
            responses={
                201: {"description": "Config created"},
                422: {"description": "Validation error"},
            },
        )

        router.add_api_route(
            path="/configs/{id}",
            endpoint=update_config,
            methods=["PUT"],
            response_model=Model,
            name="update_config",
            summary="Update a config by ID",
            responses={
                200: {"description": "Config updated"},
                400: {"description": "ID mismatch"},
                422: {"description": "Validation error"},
            },
        )

        router.add_api_route(
            path="/configs/{id}",
            endpoint=delete_config,
            status_code=204,
            methods=["DELETE"],
            name="delete_config",
            summary="Delete a config by ID",
            responses={
                204: {"description": "Config deleted"},
                404: {"description": "Config not found"},
            },
        )

        return router
