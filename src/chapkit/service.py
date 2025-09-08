from typing import Generic
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from chapkit.api.config import ConfigApi
from chapkit.api.health import HealthApi
from chapkit.api.info import InfoApi
from chapkit.api.job import JobApi
from chapkit.api.types import ChapApi
from chapkit.runner import ChapRunner
from chapkit.scheduler import JobScheduler, Scheduler
from chapkit.database import ChapDatabase
from chapkit.types import TChapConfig

templates = Jinja2Templates(directory="templates")


class ChapService(Generic[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        database: ChapDatabase[TChapConfig],
        scheduler: Scheduler | None = None,
    ) -> None:
        self._runner = runner
        self._database = database
        self._model_type = self._runner.config_type
        self._scheduler = scheduler or JobScheduler()

    def create_fastapi(self, app: FastAPI | None = None) -> FastAPI:
        if app is None:
            app = FastAPI()

            @app.get("/", response_class=HTMLResponse, include_in_schema=False)
            async def index(request: Request):
                return templates.TemplateResponse("index.html", {"request": request})

        router = self.create_api_routers()
        app.include_router(router)

        return app

    def create_api_routers(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1")

        self._include_api(router, HealthApi(self._runner))
        self._include_api(router, InfoApi(self._runner))
        self._include_api(router, ConfigApi(self._database, self._model_type))
        self._include_api(router, JobApi(self._runner, self._database, self._scheduler))

        return router

    def _include_api(self, router: APIRouter, api: ChapApi) -> None:
        router.include_router(api.create_router())

    def _resolve_cfg(self, id: UUID) -> TChapConfig:
        cfg = self._database.get_config(id)

        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {id} not found")

        return cfg
