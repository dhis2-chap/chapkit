from pathlib import Path
from typing import Generic

from fastapi import APIRouter, FastAPI, HTTPException, Request
import structlog
from ulid import ULID
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from chapkit.api.artifact import ArtifactApi
from chapkit.api.config import ConfigApi
from chapkit.api.health import HealthApi
from chapkit.api.info import InfoApi
from chapkit.api.job import JobApi
from chapkit.api.types import ChapApi
from chapkit.runner import ChapRunner
from chapkit.scheduler import JobScheduler, Scheduler
from chapkit.database import ChapDatabase
from chapkit.types import TChapConfig
import logging.config
from chapkit.logging import LOGGING_CONFIG
import logging
import logging.config

TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

logging.config.dictConfig(LOGGING_CONFIG)

structlog_processors = [
    structlog.stdlib.filter_by_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
]

structlog.configure(
    processors=structlog_processors,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class ChapService(Generic[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        database: ChapDatabase[TChapConfig],
        scheduler: Scheduler | None = None,
        artifact_level_names: list[str] | None = None,
    ) -> None:
        self._runner = runner
        self._database = database
        self._scheduler = scheduler or JobScheduler()
        self._config_type = self._runner.config_type
        self.artifact_level_names = artifact_level_names or []

    def create_fastapi(self, app: FastAPI | None = None) -> FastAPI:
        if app is None:
            app = FastAPI()
            app.logger = structlog.get_logger("fastapi")

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
        self._include_api(router, ConfigApi(self._database, config_type=self._config_type))
        self._include_api(router, JobApi(self._runner, self._database, self._scheduler))
        self._include_api(router, ArtifactApi(self._database, self))

        return router

    def _include_api(self, router: APIRouter, api: ChapApi) -> None:
        router.include_router(api.create_router())

    def _resolve_cfg(self, id: ULID) -> TChapConfig:
        cfg = self._database.get_config(id)

        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {id} not found")

        return cfg
