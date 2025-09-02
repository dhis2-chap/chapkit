from typing import Generic
from uuid import UUID

from fastapi import APIRouter, Body, HTTPException, Query

from chapkit.api.types import ChapApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.types import TChapModelConfig
from chapkit.scheduler import Scheduler
from chapkit.storage import ChapStorage
from chapkit.types import JobResponse, JobStatus, JobType, PredictBody, PredictData, PredictParams


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
            config: UUID = Query(..., description="Config ID"),
            model: UUID = Query(..., description="Trained Model ID"),
            body: PredictBody = Body(
                ...,
                description="Prediction request body containing a DataFrame (orient='split') "
                "and optional GeoJSON FeatureCollection.",
                example={
                    "df": {
                        "columns": ["Name", "Age", "City"],
                        "index": [0, 1],
                        "data": [["Eva", 28, "Tromsø"], ["Frank", 50, "Kristiansand"]],
                    },
                    "geo": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [18.9553, 69.6496]},
                                "properties": {"city": "Tromsø"},
                            },
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [8.0000, 58.1467]},
                                "properties": {"city": "Kristiansand"},
                            },
                        ],
                    },
                },
            ),
        ) -> JobResponse:
            cfg = self._storage.get_config(config)

            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Config {config} not found")

            params = PredictParams(config=cfg, data=PredictData(df=body.df, geo=body.geo))

            id = await self._scheduler.add_job(self._runner.on_predict, params)

            return JobResponse(id=id, type=JobType.predict, status=JobStatus.pending)

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
