from abc import ABC, abstractmethod
from typing import Generic
from uuid import UUID, uuid4


from chapkit.model.types import OnPredictCallable, OnTrainCallable, TChapModelConfig
from chapkit.types import ChapServiceInfo, DataFrameSplit
from chapkit.runner import ChapRunner
from chapkit.database import ChapDatabase
from chapkit.types import HealthResponse, HealthStatus, PredictParams, TrainParams
from chapkit.utils import make_awaitable


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def __init__(
        self,
        info: ChapServiceInfo,
        database: ChapDatabase[TChapModelConfig],
        *,
        config_type: type[TChapModelConfig],
    ) -> None:
        super().__init__(info, config_type=config_type)
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

    def on_info(self) -> ChapServiceInfo:
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


class FunctionalChapModelRunner(ChapModelRunnerBase[TChapModelConfig], Generic[TChapModelConfig]):
    def __init__(
        self,
        info: ChapServiceInfo,
        database: ChapDatabase[TChapModelConfig],
        *,
        config_type: type[TChapModelConfig],
        on_train: OnTrainCallable,
        on_predict: OnPredictCallable,
    ):
        self._on_train_func = make_awaitable(on_train)
        self._on_predict_func = make_awaitable(on_predict)
        super().__init__(info, database, config_type=config_type)

    async def on_train(self, params: TrainParams) -> UUID:
        model = await self._on_train_func(
            config=params.config,
            data=params.body.data,
            geo=params.body.geo,
        )

        artifact_id = uuid4()
        self._database.add_artifact(artifact_id, params.config, model)

        return artifact_id

    async def on_predict(self, params: PredictParams) -> UUID:
        result = await self._on_predict_func(
            config=params.config,
            model=params.artifact,
            historic=params.body.historic,
            future=params.body.future,
            geo=params.body.geo,
        )

        artifact_id = uuid4()
        self._database.add_artifact(
            artifact_id,
            params.config,
            DataFrameSplit.from_pandas(result),
            parent_id=params.artifact_id,
        )

        return artifact_id
