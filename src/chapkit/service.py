# chapkit/service.py
from typing import Generic

from fastapi import APIRouter, FastAPI

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
        self._health_api = HealthApi(self._runner)
        self._info_api = InfoApi(self._runner)
        self._config_api = ConfigApi(self._storage, self._model_type)
        self._job_api = JobApi(self._runner, self._storage)

    def create_fastapi(self) -> FastAPI:
        app = FastAPI()
        router = APIRouter(prefix="/api/v1")

        router.include_router(self._health_api.create_router())
        router.include_router(self._info_api.create_router())
        router.include_router(self._config_api.create_router())
        router.include_router(self._job_api.create_router())

        app.include_router(router)

        return app
