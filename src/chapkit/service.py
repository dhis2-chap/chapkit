# chapkit/service.py
from typing import Generic
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException

from chapkit.api.config import ConfigApi
from chapkit.api.health import HealthApi
from chapkit.api.info import InfoApi
from chapkit.api.job import JobApi
from chapkit.api.type import ChapApi
from chapkit.runner import ChapRunner
from chapkit.storage import ChapStorage
from chapkit.type import TChapConfig


class ChapService(Generic[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        storage: ChapStorage[TChapConfig],
    ) -> None:
        self._runner = runner
        self._storage = storage
        self._model_type = self._runner.config_type

    def create_fastapi(self, app: FastAPI | None = None) -> FastAPI:
        if app is None:
            app = FastAPI()

        router = self.create_api_routers()
        app.include_router(router)

        return app

    def create_api_routers(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1")

        self._include_api(router, HealthApi(self._runner))
        self._include_api(router, InfoApi(self._runner))
        self._include_api(router, ConfigApi(self._storage, self._model_type))
        self._include_api(router, JobApi(self._runner, self._storage))

        return router

    def _include_api(self, router: APIRouter, api: ChapApi) -> None:
        router.include_router(api.create_router())

    def _resolve_cfg(self, id: UUID) -> TChapConfig:
        cfg = self._storage.get_config(id)

        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {id} not found")

        return cfg
