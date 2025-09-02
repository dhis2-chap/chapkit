from abc import ABC, abstractmethod
from typing import Generic
from uuid import UUID, uuid4

from chapkit.model.types import ChapModelServiceInfo, TChapModelConfig
from chapkit.runner import ChapRunner
from chapkit.types import HealthResponse, HealthStatus, PredictParams, TrainParams


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    @abstractmethod
    def on_health(self) -> HealthResponse: ...

    @abstractmethod
    async def on_train(self, params: TrainParams) -> UUID: ...

    @abstractmethod
    async def on_predict(self, params: PredictParams) -> UUID: ...


class ChapModelRunnerBase(ChapModelRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)

    def on_info(self) -> ChapModelServiceInfo:
        return self._info

    async def on_train(self, params: TrainParams) -> UUID:
        print("Training with params:", params)

        return uuid4()

    async def on_predict(self, params: PredictParams) -> UUID:
        print("Predicting with params:", params)

        return uuid4()
