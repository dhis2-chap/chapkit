"""Chapkit-specific job scheduler with artifact tracking."""

from __future__ import annotations

import asyncio
import inspect
import traceback
from abc import ABC
from datetime import datetime, timezone
from typing import Any

import ulid
from pydantic import Field, PrivateAttr
from servicekit.scheduler import InMemoryScheduler, JobExecutor, JobTarget
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

    async def add_job(
        self,
        target: JobTarget,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> ULID:
        """Add a job to the scheduler and return its ID."""
        now = datetime.now(timezone.utc)
        jid = ULID()

        record = ChapkitJobRecord(
            id=jid,
            status=JobStatus.pending,
            submitted_at=now,
        )

        async with self._lock:
            if jid in self._tasks:
                raise RuntimeError(f"Job {jid!r} already scheduled")
            self._records[jid] = record

        async def _execute_target() -> Any:
            if inspect.isawaitable(target):
                if args or kwargs:
                    # Close the coroutine to avoid "coroutine was never awaited" warning
                    if inspect.iscoroutine(target):
                        target.close()
                    raise TypeError("Args/kwargs not supported when target is an awaitable object.")
                return await target
            if inspect.iscoroutinefunction(target):
                return await target(*args, **kwargs)
            return await asyncio.to_thread(target, *args, **kwargs)

        async def _runner() -> Any:
            if self._sema:
                async with self._sema:
                    return await self._run_with_state(jid, _execute_target)
            else:
                return await self._run_with_state(jid, _execute_target)

        task = asyncio.create_task(_runner(), name=f"{self.name}-job-{jid}")

        def _drain(t: asyncio.Task[Any]) -> None:
            try:
                t.result()
            except Exception:
                pass

        task.add_done_callback(_drain)

        async with self._lock:
            self._tasks[jid] = task

        return jid

    async def get_record(self, job_id: ULID) -> ChapkitJobRecord:
        """Get complete job record with artifact_id if available."""
        async with self._lock:
            if job_id not in self._records:
                raise KeyError(f"Job {job_id} not found")
            return self._records[job_id]

    async def list_records(
        self, *, status_filter: JobStatus | None = None, reverse: bool = False
    ) -> list[ChapkitJobRecord]:
        """List all job records with optional status filtering."""
        async with self._lock:
            records = list(self._records.values())
            if status_filter:
                records = [r for r in records if r.status == status_filter]
            if reverse:
                records = list(reversed(records))
            return records

    async def _run_with_state(
        self,
        jid: ULID,
        exec_fn: JobExecutor,
    ) -> Any:
        """Execute job function and track artifact_id if result is a ULID."""
        async with self._lock:
            rec = self._records[jid]
            rec.status = JobStatus.running
            rec.started_at = datetime.now(timezone.utc)

        try:
            result = await exec_fn()

            # Track artifact_id if job returns a ULID
            artifact_id: ULID | None = result if isinstance(result, ULID) else None

            async with self._lock:
                rec = self._records[jid]
                rec.status = JobStatus.completed
                rec.finished_at = datetime.now(timezone.utc)
                rec.artifact_id = artifact_id
                self._results[jid] = result

            return result

        except asyncio.CancelledError:
            async with self._lock:
                rec = self._records[jid]
                rec.status = JobStatus.canceled
                rec.finished_at = datetime.now(timezone.utc)

            raise

        except Exception as e:
            tb = traceback.format_exc()
            # Extract clean error message (exception type and message only)
            error_lines = tb.strip().split("\n")
            clean_error = error_lines[-1] if error_lines else str(e)

            async with self._lock:
                rec = self._records[jid]
                rec.status = JobStatus.failed
                rec.finished_at = datetime.now(timezone.utc)
                rec.error = clean_error
                rec.error_traceback = tb

            raise
