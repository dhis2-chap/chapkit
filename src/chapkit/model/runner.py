from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from chapkit.model.type import ChapModelServiceInfo, TChapModelConfig
from chapkit.runner import ChapRunner
from chapkit.type import HealthResponse, HealthStatus, JobRequest, JobResponse, JobStatus, JobType


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    @abstractmethod
    def on_health(self) -> HealthResponse: ...

    @abstractmethod
    async def on_train(self, job: JobRequest) -> JobResponse: ...

    @abstractmethod
    async def on_predict(self, job: JobRequest) -> JobResponse: ...


class ChapModelRunnerBase(ChapModelRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)

    def on_info(self) -> ChapModelServiceInfo:
        return self._info

    async def on_train(self, job: JobRequest, data: pd.DataFrame) -> JobResponse:
        return JobResponse(id=job.id, status=JobStatus.completed, type=JobType.train)

    async def on_predict(self, job: JobRequest) -> JobResponse:
        return JobResponse(id=job.id, status=JobStatus.completed, type=JobType.predict)
