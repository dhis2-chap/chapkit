import asyncio
import inspect
from typing import Any, Generic
from uuid import UUID, uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query

from chapkit.api.type import ChapApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.type import TChapModelConfig
from chapkit.scheduler import Scheduler
from chapkit.storage import ChapStorage
from chapkit.type import JobRequest, JobResponse, JobStatus, JobType, TChapConfig


class PredictApi(ChapApi[TChapModelConfig], Generic[TChapModelConfig]):
    def __init__(
        self,
        runner: ChapModelRunner[TChapModelConfig],
        storage: ChapStorage[TChapModelConfig],
        scheduler: Scheduler,
    ) -> None:
        self._runner = runner
        self._storage = storage
        self._scheduler = scheduler

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["chap"])

        async def endpoint(
            background_tasks: BackgroundTasks,
            config: UUID = Query(..., description="Config ID"),
            rows: list[dict[str, Any]] = Body(
                ...,
                description="Data as JSON records: [{...}, {...}]",
                example=[
                    {"k1": "v1", "k2": 1.0},
                    {"k1": "v2", "k2": 2.0},
                ],
            ),
        ) -> JobResponse:
            cfg = self._resolve_cfg(config)
            df = self._df_from_json(rows)

            async def task(id: UUID) -> JobResponse:
                jr = JobRequest(id=id, type=JobType.train, config=cfg, data=df)

                if inspect.iscoroutinefunction(self._runner.on_predict):
                    return await self._runner.on_predict(jr)

                return await asyncio.to_thread(self._runner.on_predict, jr)

            id = uuid4()
            background_tasks.add_task(task, id)

            return JobResponse(id=id, status=JobStatus.completed, type=JobType.predict)

        router.add_api_route(
            path="/predict",
            endpoint=endpoint,
            methods=["POST"],
            response_model=JobResponse,
            status_code=202,
            tags=["chap"],
            name="predict",
            responses={
                202: {"description": "Job accepted"},
                404: {"description": "Config not found"},
                422: {"description": "Validation error"},
            },
        )

        return router

    @staticmethod
    def _df_from_json(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(rows)

    def _resolve_cfg(self, id: UUID) -> TChapConfig:
        cfg = self._storage.get_config(id)

        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {id} not found")

        return cfg
