from typing import Generic

from fastapi import APIRouter, Body, HTTPException, Query
from ulid import ULID

from chapkit.api.types import ChapApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.types import TChapModelConfig
from chapkit.scheduler import JobScheduler
from chapkit.database import ChapDatabase
from chapkit.types import JobResponse, JobStatus, JobType, PredictBody, PredictData, PredictParams


class PredictApi(ChapApi[TChapModelConfig], Generic[TChapModelConfig]):
    def __init__(
        self,
        runner: ChapModelRunner[TChapModelConfig],
        database: ChapDatabase[TChapModelConfig],
        scheduler: JobScheduler,
    ) -> None:
        self._runner = runner
        self._database = database
        self._scheduler = scheduler

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["chap"])

        async def endpoint(
            artifact: ULID = Query(..., description="Trained Artifact ID"),
            body: PredictBody = Body(
                ...,
                description="Prediction request body containing a DataFrame (orient='split') "
                "and optional GeoJSON FeatureCollection.",
                examples={
                    "default": {
                        "value": {
                            "historic": {
                                "columns": ["Name", "Age", "City"],
                                "index": [0, 1],
                                "data": [["Eva", 28, "Tromsø"], ["Frank", 50, "Kristiansand"]],
                            },
                            "future": {
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
                        }
                    }
                },
            ),
        ) -> JobResponse:
            cfg = self._database.get_config_for_artifact(artifact)
            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Artifact {artifact} not found")

            artifact_obj = self._database.get_artifact(artifact)
            if artifact_obj is None:
                raise HTTPException(status_code=404, detail=f"Artifact {artifact} not found")

            params = PredictParams(
                config=cfg,
                artifact_id=artifact,
                artifact=artifact_obj,
                body=PredictData(historic=body.historic.to_pandas(), future=body.future.to_pandas(), geo=body.geo),
            )

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
