import asyncio
import inspect
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import ulid
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from chapkit.types import JobRecord, JobStatus

ULID = ulid.ULID
P = ParamSpec("P")
R = TypeVar("R")

# Targets may be:
# - sync callable -> R
# - async callable -> Awaitable[R]
# - already-created coroutine/awaitable object -> Awaitable[R]
Target = Callable[P, R] | Callable[P, Awaitable[R]] | Awaitable[R]


class JobScheduler(BaseModel, ABC):
    """Abstract scheduler that uses your existing Job* models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def add_job(self, target: Target, /, *args: P.args, **kwargs: P.kwargs) -> ULID: ...
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
    @abstractmethod
    async def wait(self, job_id: ULID, timeout: float | None = None) -> None: ...
    @abstractmethod
    async def get_result(self, job_id: ULID) -> Any: ...


class AIOJobScheduler(JobScheduler):
    """In-memory asyncio scheduler with zero globals, using your Job* models."""

    name: str = Field(default="chap")
    max_concurrency: int | None = Field(default=None, description="Limit concurrent jobs; None or <=0 = unlimited")

    # Runtime-only state
    _records: dict[ULID, JobRecord] = PrivateAttr(default_factory=dict)
    _results: dict[ULID, Any] = PrivateAttr(default_factory=dict)
    _tasks: dict[ULID, asyncio.Task[Any]] = PrivateAttr(default_factory=dict)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _sema: asyncio.Semaphore | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.max_concurrency and self.max_concurrency > 0:
            self._sema = asyncio.Semaphore(self.max_concurrency)

    async def set_max_concurrency(self, n: int | None) -> None:
        """Change the concurrency cap at runtime."""
        async with self._lock:
            self.max_concurrency = n
            if n and n > 0:
                self._sema = asyncio.Semaphore(n)
            else:
                self._sema = None

    async def add_job(self, target: Target, /, *args: P.args, **kwargs: P.kwargs) -> ULID:
        """Register and schedule `target` (sync, async, or coroutine object) for this job."""
        now = datetime.now(timezone.utc)
        jid = ULID()

        record = JobRecord(
            id=jid,
            status=JobStatus.pending,
            submitted_at=now,
        )

        async with self._lock:
            if jid in self._tasks:
                raise RuntimeError(f"Job {jid!r} already scheduled")
            self._records[jid] = record

        async def _execute_target() -> Any:
            # Decide how to invoke the target
            if inspect.isawaitable(target):
                if args or kwargs:
                    raise TypeError("Args/kwargs not supported when target is an awaitable object.")
                return await target  # type: ignore[misc]
            if inspect.iscoroutinefunction(target):
                return await target(*args, **kwargs)  # type: ignore[misc]
            # Sync callable: run in a thread so we don't block the loop
            return await asyncio.to_thread(target, *args, **kwargs)

        async def _runner() -> Any:
            # Concurrency gate
            if self._sema:
                async with self._sema:
                    return await self._run_with_state(jid, _execute_target)
            else:
                return await self._run_with_state(jid, _execute_target)

        task = asyncio.create_task(_runner(), name=f"{self.name}-job-{jid}")

        # Drain result/exception to avoid "Task exception was never retrieved".
        def _drain(t: asyncio.Task[Any]) -> None:
            try:
                t.result()
            except Exception:
                # Already recorded in JobRecord; ignore here.
                pass

        task.add_done_callback(_drain)

        async with self._lock:
            self._tasks[jid] = task

        return jid

    async def _run_with_state(
        self,
        jid: ULID,
        exec_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        # Mark running
        async with self._lock:
            rec = self._records[jid]
            rec.status = JobStatus.running
            rec.started_at = datetime.now(timezone.utc)

        try:
            result = await exec_fn()

            # artifact_id = result if the result IS a ULID, else None.
            artifact: ULID | None = result if isinstance(result, ULID) else None

            async with self._lock:
                rec = self._records[jid]
                rec.status = JobStatus.completed
                rec.finished_at = datetime.now(timezone.utc)
                rec.artifact_id = artifact  # ULID | None
                self._results[jid] = result

            return result

        except asyncio.CancelledError:
            async with self._lock:
                rec = self._records[jid]
                rec.status = JobStatus.canceled
                rec.finished_at = datetime.now(timezone.utc)

            raise

        except Exception:
            tb = traceback.format_exc()
            async with self._lock:
                rec = self._records[jid]
                rec.status = JobStatus.failed
                rec.finished_at = datetime.now(timezone.utc)
                rec.error = tb  # rich traceback

            raise

    async def get_all_records(self) -> list[JobRecord]:
        async with self._lock:
            records = [r.model_copy(deep=True) for r in self._records.values()]  # type: ignore[attr-defined]

        # newest first
        records.sort(
            key=lambda r: getattr(r, "submitted_at", datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )

        return records

    async def get_record(self, job_id: ULID) -> JobRecord:
        async with self._lock:
            rec = self._records.get(job_id)

            if rec is None:
                raise KeyError("Job not found")

            # Return a deep copy so callers cannot mutate internal state
            return rec.model_copy(deep=True)  # type: ignore[attr-defined]

    async def get_status(self, job_id: ULID) -> JobStatus:
        async with self._lock:
            rec = self._records.get(job_id)

            if rec is None:
                raise KeyError("Job not found")

            return rec.status

    async def get_result(self, job_id: ULID) -> Any:
        """Return result for completed job; raise if failed or not finished."""
        async with self._lock:
            rec = self._records.get(job_id)

            if rec is None:
                raise KeyError("Job not found")

            if rec.status == JobStatus.completed:
                return self._results.get(job_id)

            if rec.status == JobStatus.failed:
                msg = getattr(rec, "error", "Job failed")
                raise RuntimeError(msg)

            raise RuntimeError(f"Job not finished (status={rec.status})")

    async def wait(self, job_id: ULID, timeout: float | None = None) -> None:
        """Wait until the job reaches a terminal state (completed/failed/canceled)."""
        async with self._lock:
            task = self._tasks.get(job_id)

            if task is None:
                raise KeyError("Job not found")

        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

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
