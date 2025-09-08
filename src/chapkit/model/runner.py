from abc import ABC, abstractmethod
from typing import Generic
from uuid import UUID, uuid4

from chapkit.model.types import ChapModelServiceInfo, TChapModelConfig
from chapkit.runner import ChapRunner
from chapkit.database import ChapDatabase
from chapkit.types import HealthResponse, HealthStatus, PredictParams, TrainParams


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def __init__(
        self,
        info: ChapModelServiceInfo,
        config_type: type[TChapModelConfig],
        database: ChapDatabase[TChapModelConfig],
    ) -> None:
        super().__init__(info, config_type)
        self._database = database

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

        artifact_id = uuid4()
        self._database.add_artifact(artifact_id, params.config, {"a": 1})

        return artifact_id

    async def on_predict(self, params: PredictParams) -> UUID:
        print("Predicting with params:", params)
        # a real implementation would use the model to make a prediction

        artifact_id = uuid4()
        self._database.add_artifact(artifact_id, params.config, {"b": 2})

        return artifact_id
