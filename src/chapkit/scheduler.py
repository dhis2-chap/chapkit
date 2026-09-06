"""Chapkit-specific job scheduler with artifact tracking."""

from __future__ import annotations

from abc import ABC
from datetime import datetime
from typing import Any, cast

import ulid
from pydantic import Field, PrivateAttr
from servicekit.scheduler import InMemoryScheduler
from servicekit.schemas import JobRecord, JobStatus

ULID = ulid.ULID


class ChapkitJobRecord(JobRecord):
    """Job record extended with artifact_id tracking for ML/task workflows."""

    artifact_id: ULID | None = Field(default=None, description="ID of artifact created by job (if job returns a ULID)")


class ChapkitScheduler(InMemoryScheduler, ABC):
    """Abstract base class for Chapkit job schedulers with artifact tracking."""

    async def get_record(self, job_id: ULID) -> ChapkitJobRecord:
        """Get complete job record with artifact_id if available."""
        raise NotImplementedError

    async def list_records(
        self, *, status_filter: JobStatus | None = None, reverse: bool = False
    ) -> list[ChapkitJobRecord]:
        """List all job records with optional status filtering."""
        raise NotImplementedError


class InMemoryChapkitScheduler(ChapkitScheduler):
    """In-memory scheduler with automatic artifact tracking for jobs that return ULIDs."""

    # Override with ChapkitJobRecord type to support artifact_id tracking
    # dict is invariant, but we always use ChapkitJobRecord in this subclass
    _records: dict[ULID, ChapkitJobRecord] = PrivateAttr(default_factory=dict)  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleVariableOverride]

    async def get_record(self, job_id: ULID) -> ChapkitJobRecord:
        """Get a copy of the complete job record with artifact_id if available."""
        async with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(f"Job {job_id} not found")
            return record.model_copy(deep=True)

    async def list_records(
        self, *, status_filter: JobStatus | None = None, reverse: bool = False
    ) -> list[ChapkitJobRecord]:
        """List copies of all job records with optional status filtering."""
        async with self._lock:
            records = [record.model_copy(deep=True) for record in self._records.values()]
        if status_filter:
            records = [record for record in records if record.status == status_filter]
        if reverse:
            records = list(reversed(records))
        return records

    def _make_record(self, job_id: ULID, submitted_at: datetime) -> ChapkitJobRecord:
        """Create a chapkit job record for a newly submitted job."""
        return ChapkitJobRecord(id=job_id, status=JobStatus.pending, submitted_at=submitted_at)

    async def _on_job_result(self, record: JobRecord, result: Any) -> None:
        """Track the artifact id when a job returns a ULID."""
        # Every record this scheduler tracks is created by `_make_record` above.
        chapkit_record = cast(ChapkitJobRecord, record)
        chapkit_record.artifact_id = result if isinstance(result, ULID) else None
