# scheduler.py — Python 3.12+/3.13, Pydantic v2
from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr

from chapkit.type import JobRecord, JobRequest, JobStatus

T = TypeVar("T")


class Scheduler(BaseModel, ABC):
    """Abstract scheduler that uses your existing Job* models."""

    model_config = {"arbitrary_types_allowed": True}

    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @abstractmethod
    async def add_job(
        self,
        job: JobRequest[Any],
        target: Callable[..., Any] | Callable[..., Awaitable[Any]],
        /,
        *args: Any,
        delay: float | None = None,
        run_at: datetime | None = None,
        **kwargs: Any,
    ) -> None: ...

    @abstractmethod
    async def get_status(self, job_id: UUID) -> JobStatus: ...

    @abstractmethod
    async def get_record(self, job_id: UUID) -> JobRecord: ...

    @abstractmethod
    async def get_result(self, job_id: UUID) -> Any: ...

    @abstractmethod
    async def cancel(self, job_id: UUID) -> bool: ...

    @abstractmethod
    async def delete(self, job_id: UUID) -> None: ...


class JobScheduler(Scheduler):
    """In-memory asyncio scheduler with zero globals, using your Job* models."""

    name: str = Field(default="chap")

    # Runtime-only state (kept out of serialization)
    _records: dict[UUID, JobRecord] = PrivateAttr(default_factory=dict)
    _results: dict[UUID, Any] = PrivateAttr(default_factory=dict)
    _tasks: dict[UUID, asyncio.Task[Any]] = PrivateAttr(default_factory=dict)

    async def add_job(
        self,
        job: JobRequest[Any],
        target: Callable[..., Any] | Callable[..., Awaitable[Any]],
        /,
        *args: Any,
        delay: float | None = None,
        run_at: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        """Register and schedule `target` (sync or async) for this job."""
        # Create initial record using YOUR JobRecord model
        now = datetime.now(timezone.utc)
        record = JobRecord(
            id=job.id,
            type=job.type,
            status=JobStatus.pending,
            submitted_at=now,  # If your JobRecord uses str, Pydantic coerces to ISO.
        )

        async with self._lock:
            if job.id in self._tasks:
                raise RuntimeError(f"Job {job.id} already scheduled")
            self._records[job.id] = record

        # Compute effective delay
        eff_delay = 0.0
        if run_at is not None:
            run_at_utc = run_at if run_at.tzinfo else run_at.replace(tzinfo=timezone.utc)
            eff_delay = max(0.0, (run_at_utc - now).total_seconds())
        elif delay:
            eff_delay = max(0.0, delay)

        async def _runner() -> Any:
            if eff_delay > 0:
                await asyncio.sleep(eff_delay)

            # Mark running
            async with self._lock:
                rec = self._records[job.id]
                rec.status = JobStatus.running
                rec.started_at = datetime.now(timezone.utc)

            try:
                # Support async or sync targets seamlessly
                if inspect.iscoroutinefunction(target):
                    result = await target(*args, **kwargs)  # type: ignore[misc]
                else:
                    result = await asyncio.to_thread(target, *args, **kwargs)

                async with self._lock:
                    rec = self._records[job.id]
                    rec.status = JobStatus.completed
                    rec.finished_at = datetime.now(timezone.utc)
                    self._results[job.id] = result
                return result

            except asyncio.CancelledError:
                async with self._lock:
                    rec = self._records[job.id]
                    rec.status = JobStatus.canceled
                    rec.finished_at = datetime.now(timezone.utc)
                raise

            except Exception as e:
                async with self._lock:
                    rec = self._records[job.id]
                    rec.status = JobStatus.failed
                    rec.finished_at = datetime.now(timezone.utc)
                    # If your JobRecord has `error: str | None`
                    if hasattr(rec, "error"):
                        rec.error = f"{type(e).__name__}: {e}"
                raise

        task = asyncio.create_task(_runner(), name=f"{self.name}-job-{job.id}")
        async with self._lock:
            self._tasks[job.id] = task

    # -------------------------- Introspection API --------------------------

    async def get_status(self, job_id: UUID) -> JobStatus:
        async with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError("Job not found")
            return rec.status

    async def get_record(self, job_id: UUID) -> JobRecord:
        async with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError("Job not found")
            return rec

    async def get_result(self, job_id: UUID) -> Any:
        async with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError("Job not found")

            if rec.status in (JobStatus.pending, JobStatus.running):
                raise RuntimeError("Result not ready")

            if rec.status is JobStatus.completed:
                # Note: result may legitimately be None if target returned None
                return self._results.get(job_id)

            if rec.status is JobStatus.failed:
                # Prefer your record's `error` if present
                msg = getattr(rec, "error", None) or "Job failed"
                raise RuntimeError(msg)

            if rec.status is JobStatus.canceled:
                raise RuntimeError("Job canceled")

            raise RuntimeError("No result available")

    async def cancel(self, job_id: UUID) -> bool:
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

    async def delete(self, job_id: UUID) -> None:
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
