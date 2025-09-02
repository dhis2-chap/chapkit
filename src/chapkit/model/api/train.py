# train.py
import asyncio
import inspect
from typing import Any, Generic
from uuid import UUID, uuid4

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query

from chapkit.api.type import ChapApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.type import TChapModelConfig
from chapkit.scheduler import Scheduler
from chapkit.storage import ChapStorage
from chapkit.type import JobRequest, JobResponse, JobStatus, JobType, TChapConfig


class TrainApi(ChapApi[TChapModelConfig], Generic[TChapModelConfig]):
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
            # Resolve config
            cfg = self._resolve_cfg(config)
            df = self._df_from_json(rows)

            # Create JobRequest
            job_id = uuid4()
            jr = JobRequest(id=job_id, type=JobType.train, config=cfg)

            # Wrap runner call so it works for sync/async methods
            async def runner_task(job: JobRequest) -> Any:
                if inspect.iscoroutinefunction(self._runner.on_train):
                    return await self._runner.on_train(job)
                return await asyncio.to_thread(self._runner.on_train, job)

            # Schedule the job
            await self._scheduler.add_job(jr, runner_task, jr, df)

            # Return immediately with status=pending
            return JobResponse(id=job_id, type=JobType.train, status=JobStatus.pending)

        router.add_api_route(
            path="/train",
            endpoint=endpoint,
            methods=["POST"],
            response_model=JobResponse,
            status_code=202,
            tags=["chap"],
            name="train",
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
