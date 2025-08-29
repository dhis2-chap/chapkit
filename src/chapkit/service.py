from uuid import UUID

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from chapkit.runner import ChapRunner
from chapkit.storage import ChapStorage
from chapkit.types import ChapConfig, ChapServiceInfo, HealthResponse, JobResponse, JobStatus, JobType


class ChapService[T: ChapConfig]:
    def __init__(
        self,
        info: ChapServiceInfo,
        runner: ChapRunner[T],
        storage: ChapStorage[T],
        model_type: type[T],
    ) -> None:
        self._info = info
        self._runner = runner
        self._storage = storage
        self._model_type = model_type

    def create_fastapi(self):
        app = FastAPI()
        self._setup_health(app)
        self._setup_info(app)
        self._setup_configs(app)
        self._setup_jobs(app)
        self._setup_train(app)
        self._setup_predict(app)

        return app

    def _setup_health(self, app: FastAPI) -> None:
        method = getattr(self._runner, "on_health", None)

        if method:
            app.add_api_route(
                path="/health",
                endpoint=method,
                methods=["GET"],
                tags=["information"],
                name="health",
                response_model=HealthResponse,
            )

    def _setup_info(self, app: FastAPI) -> None:
        def endpoint() -> ChapServiceInfo:
            return self._info

        app.add_api_route(
            path="/info",
            endpoint=endpoint,
            methods=["GET"],
            tags=["information"],
            name="info",
            response_model=ChapServiceInfo,
        )

    def _setup_configs(self, app: FastAPI) -> None:
        type TModelType = self._model_type  # type: ignore

        async def get_config(id: UUID) -> TModelType:
            cfg = self._storage.get_config(id)

            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Config {id} not found")

            return cfg

        async def get_configs() -> list[TModelType]:
            return self._storage.get_configs()

        async def get_schema() -> dict:
            return self._model_type.model_json_schema()

        async def add_config(cfg: TModelType = Body(...)) -> JSONResponse:
            validated = self._model_type.model_validate(cfg)
            self._storage.add_config(validated)

            return JSONResponse(
                status_code=201,
                content=validated.model_dump(mode="json"),
                headers={"Location": f"/configs/{validated.id}"},
            )

        async def delete_config(id: UUID) -> Response:
            if not self._storage.del_config(id):
                raise HTTPException(status_code=404, detail=f"Config {id} not found")

            return Response(status_code=204)

        app.add_api_route(
            path="/configs",
            endpoint=get_configs,
            methods=["GET"],
            response_model=list[TModelType],
            tags=["configs"],
            name="list_configs",
            summary="List all configs",
        )

        app.add_api_route(
            path="/configs/schema",
            endpoint=get_schema,
            methods=["GET"],
            response_model=dict,
            responses={200: {"content": {"application/schema+json": {}}}},
            tags=["configs"],
            name="get_config_schema",
            summary="Get JSON Schema for the current config model",
        )

        app.add_api_route(
            path="/configs/{id}",
            endpoint=get_config,
            methods=["GET"],
            response_model=TModelType,
            tags=["configs"],
            name="get_config",
            summary="Get a config by ID",
            responses={
                404: {"description": "Config not found"},
            },
        )

        app.add_api_route(
            path="/configs",
            endpoint=add_config,
            methods=["POST"],
            response_model=TModelType,
            status_code=201,
            tags=["configs"],
            name="create_config",
            summary="Create (or replace) a config",
            responses={
                201: {"description": "Config created"},
                422: {"description": "Validation error"},
            },
        )

        app.add_api_route(
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

    def _setup_jobs(self, app: FastAPI) -> None:
        pass

    def _setup_train(self, app: FastAPI) -> None:
        def endpoint(config: UUID) -> JobResponse:
            return JobResponse(status=JobStatus.pending, type=JobType.train)

        app.add_api_route(
            path="/train",
            endpoint=endpoint,
            methods=["POST"],
            response_model=JobResponse,
            status_code=202,
            tags=["chap"],
            responses={
                202: {"description": "Job accepted"},
                422: {"description": "Validation error"},
            },
        )

    def _setup_predict(self, app: FastAPI) -> None:
        def endpoint(config: UUID) -> JobResponse:
            return JobResponse(status=JobStatus.pending, type=JobType.predict)

        app.add_api_route(
            path="/predict",
            endpoint=endpoint,
            methods=["POST"],
            response_model=JobResponse,
            status_code=202,
            tags=["chap"],
            responses={
                202: {"description": "Job accepted"},
                422: {"description": "Validation error"},
            },
        )
