# chapkit/model/service.py
from typing import Generic, TypeVar

from fastapi import APIRouter

from chapkit.model.api.predict import PredictApi
from chapkit.model.api.train import TrainApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.type import ChapModelConfig
from chapkit.service import ChapService
from chapkit.storage import ChapStorage

TModelConfig = TypeVar("TModelConfig", bound=ChapModelConfig)


class ChapModelService(ChapService[TModelConfig], Generic[TModelConfig]):
    def __init__(
        self,
        runner: ChapModelRunner[TModelConfig],
        storage: ChapStorage[TModelConfig],
    ) -> None:
        super().__init__(runner, storage)
        self._runner = runner  # narrowed type

    def create_router(self) -> APIRouter:
        router = super().create_router()
        router.include_router(TrainApi(self._runner, self).create_router())
        router.include_router(PredictApi(self._runner, self).create_router())
        return router
