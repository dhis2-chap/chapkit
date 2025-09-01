from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from chapkit.model.type import ChapModelServiceInfo, TChapModelConfig
from chapkit.runner import ChapRunner
from chapkit.type import Job, JobResponse, JobStatus, JobType


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    @abstractmethod
    def on_train(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse: ...

    @abstractmethod
    def on_predict(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse: ...


class ChapModelRunnerBase(ChapModelRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def on_info(self) -> ChapModelServiceInfo:
        return self._info

    def on_train(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse:
        response = JobResponse(status=JobStatus.completed, type=JobType.train)

        job = Job(id=response.id, status=JobStatus.completed, type=JobType.train, config=cfg, data=data)
        print(job)

        return response

    def on_predict(self, cfg: TChapModelConfig, data: pd.DataFrame) -> JobResponse:
        response = JobResponse(status=JobStatus.completed, type=JobType.predict)

        job = Job(id=response.id, status=JobStatus.completed, type=JobType.predict, config=cfg, data=data)
        print(job)

        return response
