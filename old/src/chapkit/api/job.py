from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from ulid import ULID

from chapkit.api.types import ChapApi
from chapkit.runner import ChapRunner
from chapkit.scheduler import JobScheduler
from chapkit.database import ChapDatabase
from chapkit.types import JobRecord, JobStatus, TChapConfig


class JobApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        database: ChapDatabase[TChapConfig],
        scheduler: JobScheduler,
    ) -> None:
        self._runner = runner
        self._database = database
        self._scheduler = scheduler

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["jobs"])

        async def get_jobs(status: JobStatus = None) -> List[JobRecord]:
            jobs = await self._scheduler.get_all_records()
            if status:
                return [job for job in jobs if job.status == status]
            return jobs

        async def get_job(id: ULID) -> JobRecord:
            try:
                return await self._scheduler.get_record(id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Job not found")

        async def delete_job(id: ULID) -> Response:
            try:
                await self._scheduler.delete(id)
            except KeyError:
                raise HTTPException(status_code=404, detail="Job not found")
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        router.add_api_route(
            "/jobs",
            get_jobs,
            methods=["GET"],
            response_model=List[JobRecord],
            summary="Get all jobs, optionally filtered by status",
        )

        router.add_api_route(
            "/jobs/{id}",
            get_job,
            methods=["GET"],
            response_model=JobRecord,
            summary="Get full job record",
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
