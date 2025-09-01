# chapkit/service.py
from typing import Generic
from uuid import UUID

from fastapi import APIRouter, Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from chapkit.runner import ChapRunner
from chapkit.storage import ChapStorage
from chapkit.type import ChapServiceInfo, HealthResponse, TChapConfig


class ChapService(Generic[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        storage: ChapStorage[TChapConfig],
    ) -> None:
        self._runner = runner
        self._storage = storage
        self._model_type = self._runner.config_type

    def create_fastapi(self) -> FastAPI:
        app = FastAPI()
        router = APIRouter(prefix="/api/v1")
        self._setup_routes(router)
        app.include_router(router)

        return app

    def _setup_routes(self, router: APIRouter) -> None:
        self._setup_health(router)
        self._setup_info(router)
        self._setup_configs(router)
        self._setup_jobs(router)

    def _setup_health(self, router: APIRouter) -> None:
        router.add_api_route(
            path="/health",
            endpoint=self._runner.on_health,
            methods=["GET"],
            tags=["information"],
            name="health",
            response_model=HealthResponse,
        )

    def _setup_info(self, router: APIRouter) -> None:
        router.add_api_route(
            path="/info",
            endpoint=self._runner.on_info,
            methods=["GET"],
            tags=["information"],
            name="info",
            response_model=ChapServiceInfo,
        )

    def _setup_configs(self, router: APIRouter) -> None:
        Model = self._model_type  # concrete Pydantic class

        async def get_config(id: UUID):
            cfg = self._storage.get_config(id)

            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Config {id} not found")

            return cfg

        async def get_configs():
            return self._storage.get_configs()

        async def get_schema():
            return Model.model_json_schema()

        async def add_config(cfg: dict = Body(...)):
            validated = Model.model_validate(cfg)
            self._storage.add_config(validated)

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

            self._storage.add_config(validated)

            return validated

        async def delete_config(id: UUID):
            if not self._storage.del_config(id):
                raise HTTPException(status_code=404, detail=f"Config {id} not found")

            return Response(status_code=204)

        router.add_api_route(
            path="/configs",
            endpoint=get_configs,
            methods=["GET"],
            response_model=list[Model],  # OK to pass the actual class here
            tags=["configs"],
            name="list_configs",
            summary="List all configs",
        )

        router.add_api_route(
            path="/configs/schema",
            endpoint=get_schema,
            methods=["GET"],
            response_model=dict,
            responses={200: {"content": {"application/schema+json": {}}}},
            tags=["configs"],
            name="get_config_schema",
            summary="Get JSON Schema for the current config model",
        )

        router.add_api_route(
            path="/configs/{id}",
            endpoint=get_config,
            methods=["GET"],
            response_model=Model,
            tags=["configs"],
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
            tags=["configs"],
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
            tags=["configs"],
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
            tags=["configs"],
            name="delete_config",
            summary="Delete a config by ID",
            responses={
                204: {"description": "Config deleted"},
                404: {"description": "Config not found"},
            },
        )

    def _setup_jobs(self, router: APIRouter) -> None:
        pass
