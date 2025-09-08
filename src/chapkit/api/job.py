from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, Response

from chapkit.api.types import ChapApi
from chapkit.runner import ChapRunner
from chapkit.scheduler import Scheduler
from chapkit.database import ChapDatabase
from chapkit.types import JobRecord, JobResponse, JobStatus, TChapConfig


class JobApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        database: ChapDatabase[TChapConfig],
        scheduler: Scheduler,
    ) -> None:
        self._runner = runner
        self._database = database
        self._scheduler = scheduler

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["jobs"])

        async def get_job(id: UUID) -> JobRecord:
            try:
                return await self._scheduler.get_record(id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Job not found")

        async def get_status(id: UUID) -> dict[str, JobStatus]:
            try:
                st = await self._scheduler.get_status(id)
                return {"status": st}
            except KeyError:
                raise HTTPException(status_code=404, detail="Job not found")

        async def get_result(id: UUID) -> Any:
            try:
                result = await self._scheduler.get_result(id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Job not found")
            except RuntimeError as e:
                msg = str(e)
                if msg == "Result not ready":
                    return JSONResponse(
                        status_code=status.HTTP_409_CONFLICT,
                        content={"detail": msg},
                    )
                raise HTTPException(status_code=400, detail=msg)
            return JobResponse(id=result, status=JobStatus.completed)

        async def delete_job(id: UUID) -> Response:
            try:
                await self._scheduler.delete(id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Job not found")
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        router.add_api_route(
            "/jobs/{id}",
            get_job,
            methods=["GET"],
            response_model=JobRecord,
            summary="Get full job record",
        )

        router.add_api_route(
            "/jobs/{id}/status",
            get_status,
            response_model_exclude_none=True,
            methods=["GET"],
            response_model=dict,
            summary="Get job status only",
        )

        router.add_api_route(
            "/jobs/{id}/result",
            get_result,
            response_model_exclude_none=True,
            methods=["GET"],
            summary="Get job result (200 on success, 409 if not ready)",
        )

        router.add_api_route(
            "/jobs/{id}",
            delete_job,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
            response_model=None,
            summary="Cancel (if running) and delete a job",
        )

        return router
