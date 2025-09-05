# chapkit/model/service.py
from typing import Generic, TypeVar

from fastapi import APIRouter

from chapkit.model.api.predict import PredictApi
from chapkit.model.api.train import TrainApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.types import ChapModelConfig
from chapkit.scheduler import Scheduler
from chapkit.service import ChapService
from chapkit.storage import ChapStorage

TModelConfig = TypeVar("TModelConfig", bound=ChapModelConfig)


class ChapModelService(ChapService[TModelConfig], Generic[TModelConfig]):
    def __init__(
        self,
        runner: ChapModelRunner[TModelConfig],
        storage: ChapStorage[TModelConfig],
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(runner, storage, scheduler)
        self._runner: ChapModelRunner[TModelConfig] = runner

    def create_api_routers(self) -> APIRouter:
        router = super().create_api_routers()

        self._include_api(router, TrainApi(self._runner, self._storage, self._scheduler))
        self._include_api(router, PredictApi(self._runner, self._storage, self._scheduler))

        return router
