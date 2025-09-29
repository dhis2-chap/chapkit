import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import ulid
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from chapkit.types import JobRecord, JobStatus

ULID = ulid.ULID


class JobScheduler(BaseModel, ABC):
    """Abstract scheduler that uses your existing Job* models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def add_job(
        self,
        target: Callable[..., Any] | Callable[..., Awaitable[Any]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> ULID: ...

    @abstractmethod
    async def get_status(self, job_id: ULID) -> JobStatus: ...

    @abstractmethod
    async def get_record(self, job_id: ULID) -> JobRecord: ...

    @abstractmethod
    async def get_all_records(self) -> list[JobRecord]: ...

    @abstractmethod
    async def cancel(self, job_id: ULID) -> bool: ...

    @abstractmethod
    async def delete(self, job_id: ULID) -> None: ...


class AIOJobScheduler(JobScheduler):
    """In-memory asyncio scheduler with zero globals, using your Job* models."""

    name: str = Field(default="chap")

    # Runtime-only state (kept out of serialization)
    _records: dict[ULID, JobRecord] = PrivateAttr(default_factory=dict)
    _results: dict[ULID, Any] = PrivateAttr(default_factory=dict)
    _tasks: dict[ULID, asyncio.Task[Any]] = PrivateAttr(default_factory=dict)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def add_job(
        self,
        target: Callable[..., Any] | Callable[..., Awaitable[Any]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> ULID:
        """Register and schedule `target` (sync or async) for this job."""
        now = datetime.now(timezone.utc)
        id = ULID()

        record = JobRecord(
            id=id,
            status=JobStatus.pending,
            submitted_at=now,  # ensure JobRecord uses datetime fields
        )

        async with self._lock:
            if id in self._tasks:
                raise RuntimeError(f"Job {id!r} already scheduled")
            self._records[id] = record

        async def _runner() -> Any:
            # Mark running
            async with self._lock:
                rec = self._records[id]
                rec.status = JobStatus.running
                rec.started_at = datetime.now(timezone.utc)

            try:
                # Support async or sync targets seamlessly
                if inspect.iscoroutinefunction(target):
                    result = await target(*args, **kwargs)  # type: ignore[misc]
                else:
                    result = await asyncio.to_thread(target, *args, **kwargs)

                async with self._lock:
                    rec = self._records[id]
                    rec.status = JobStatus.completed
                    rec.finished_at = datetime.now(timezone.utc)
                    rec.artifact_id = result
                    self._results[id] = result
                return result

            except asyncio.CancelledError:
                async with self._lock:
                    rec = self._records[id]
                    rec.status = JobStatus.canceled
                    rec.finished_at = datetime.now(timezone.utc)
                raise

            except Exception as e:
                async with self._lock:
                    rec = self._records[id]
                    rec.status = JobStatus.failed
                    rec.finished_at = datetime.now(timezone.utc)
                    if hasattr(rec, "error"):
                        rec.error = f"{type(e).__name__}: {e!s}"
                raise

        task = asyncio.create_task(_runner(), name=f"{self.name}-job-{id}")

        async with self._lock:
            self._tasks[id] = task

        return id

    async def get_status(self, job_id: ULID) -> JobStatus:
        async with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError("Job not found")
            return rec.status

    async def get_record(self, job_id: ULID) -> JobRecord:
        async with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError("Job not found")
            return rec

    async def get_all_records(self) -> list[JobRecord]:
        async with self._lock:
            return list(self._records.values())

    async def cancel(self, job_id: ULID) -> bool:
        """True if a running task was canceled; False if already done/missing."""
        async with self._lock:
            task = self._tasks.get(job_id)
            exists = job_id in self._records

        if not exists:
            raise KeyError("Job not found")

        if not task or task.done():
            return False

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def delete(self, job_id: ULID) -> None:
        """Remove job record, task, and result; cancels if still running."""
        async with self._lock:
            rec = self._records.get(job_id)
            task = self._tasks.get(job_id)

        if rec is None:
            raise KeyError("Job not found")

        if task and not task.done():
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            self._records.pop(job_id, None)
            self._tasks.pop(job_id, None)
            self._results.pop(job_id, None)
