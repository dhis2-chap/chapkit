# chapkit/service.py
from typing import Generic
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException

from chapkit.api.config import ConfigApi
from chapkit.api.health import HealthApi
from chapkit.api.info import InfoApi
from chapkit.api.job import JobApi
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

    def create_fastapi(self) -> FastAPI:
        app = FastAPI()
        router = self.create_router()
        app.include_router(router)
        return app

    def create_router(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1")
        router.include_router(HealthApi(self._runner).create_router())
        router.include_router(InfoApi(self._runner).create_router())
        router.include_router(ConfigApi(self._storage, self._model_type).create_router())
        router.include_router(JobApi(self._runner, self._storage).create_router())
        return router

    def _resolve_cfg(self, id: UUID) -> TChapConfig:
        cfg = self._storage.get_config(id)

        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {id} not found")

        return cfg
