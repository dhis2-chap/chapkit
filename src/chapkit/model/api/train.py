from typing import Generic
from uuid import UUID

from fastapi import APIRouter, Body, HTTPException, Query

from chapkit.api.types import ChapApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.types import TChapModelConfig
from chapkit.scheduler import Scheduler
from chapkit.storage import ChapStorage
from chapkit.types import JobResponse, JobStatus, JobType, TrainBody, TrainData, TrainParams


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
            body: TrainBody = Body(
                ...,
                description="Training request body containing a DataFrame (orient='split') "
                "and optional GeoJSON FeatureCollection.",
                example={
                    "df": {
                        "columns": ["Name", "Age", "City"],
                        "index": [0, 1, 2, 3],
                        "data": [
                            ["Alice", 25, "Oslo"],
                            ["Bob", 30, "Bergen"],
                            ["Charlie", 35, "Trondheim"],
                            ["Diana", 40, "Stavanger"],
                        ],
                    },
                    "geo": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [10.7461, 59.9127]},
                                "properties": {"city": "Oslo"},
                            }
                        ],
                    },
                },
            ),
        ) -> JobResponse:
            cfg = self._storage.get_config(config)

            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Config {config} not found")

            params = TrainParams(config=cfg, data=TrainData(df=body.df.to_pandas(), geo=body.geo))

            id = await self._scheduler.add_job(self._runner.on_train, params)

            return JobResponse(id=id, type=JobType.train, status=JobStatus.pending)

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
