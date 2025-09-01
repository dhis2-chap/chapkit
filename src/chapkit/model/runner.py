from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from chapkit.model.type import ChapModelServiceInfo, TChapModelConfig
from chapkit.runner import ChapRunner
from chapkit.type import JobResponse, JobStatus, JobType


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    @abstractmethod
    def on_train(self, cfg: TChapModelConfig, data: pd.DataFrame) -> TChapModelConfig: ...

    @abstractmethod
    def on_predict(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse: ...


class ChapModelRunnerBase(ChapModelRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def on_info(self) -> ChapModelServiceInfo:
        return self._info

    def on_train(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse:
        return JobResponse(status=JobStatus.pending, type=JobType.train)

    def on_predict(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse:
        return JobResponse(status=JobStatus.pending, type=JobType.predict)
