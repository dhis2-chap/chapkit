"""Tests for InMemoryScheduler edge cases and error handling."""

import asyncio

import pytest
from servicekit import JobStatus
from ulid import ULID

from chapkit.scheduler import ChapkitScheduler, InMemoryScheduler


async def test_abstract_base_class_methods() -> None:
    """Test ChapkitScheduler abstract methods raise NotImplementedError."""

    class MinimalScheduler(ChapkitScheduler):
        """Minimal scheduler that doesn't override abstract methods."""

        pass

    scheduler = MinimalScheduler()

    # Test get_record raises NotImplementedError
    with pytest.raises(NotImplementedError):
        await scheduler.get_record(ULID())

    # Test list_records raises NotImplementedError
    with pytest.raises(NotImplementedError):
        await scheduler.list_records()


async def test_add_job_with_sync_function() -> None:
    """Test submitting a synchronous function (uses asyncio.to_thread)."""
    scheduler = InMemoryScheduler()

    def sync_task(value: int) -> int:
        return value * 2

    job_id = await scheduler.add_job(sync_task, 21)

    # Wait for completion
    await asyncio.sleep(0.1)

    record = await scheduler.get_record(job_id)
    assert record.status == JobStatus.completed
    assert record.error is None


async def test_add_job_with_semaphore() -> None:
    """Test job submission with concurrency semaphore."""
    scheduler = InMemoryScheduler(max_concurrency=2)

    async def slow_task(duration: float) -> str:
        await asyncio.sleep(duration)
        return "done"

    # Submit 3 jobs with max_concurrent=2
    job1 = await scheduler.add_job(slow_task, 0.1)
    job2 = await scheduler.add_job(slow_task, 0.1)
    job3 = await scheduler.add_job(slow_task, 0.1)

    # Wait for all to complete
    await asyncio.sleep(0.3)

    # All should complete successfully
    for job_id in [job1, job2, job3]:
        record = await scheduler.get_record(job_id)
        assert record.status == JobStatus.completed


async def test_get_record_not_found_raises_key_error() -> None:
    """Test that getting a non-existent job record raises KeyError."""
    scheduler = InMemoryScheduler()

    missing_id = ULID()
    with pytest.raises(KeyError, match=f"Job {missing_id} not found"):
        await scheduler.get_record(missing_id)


async def test_list_records_with_status_filter() -> None:
    """Test listing job records with status filtering."""
    scheduler = InMemoryScheduler()

    async def success_task() -> str:
        return "ok"

    async def fail_task() -> None:
        raise ValueError("intentional")

    # Submit jobs
    job1 = await scheduler.add_job(success_task)
    job2 = await scheduler.add_job(fail_task)

    # Wait for completion
    await asyncio.sleep(0.2)

    # List all records
    all_records = await scheduler.list_records()
    assert len(all_records) == 2

    # List only completed
    completed_records = await scheduler.list_records(status_filter=JobStatus.completed)
    assert len(completed_records) == 1
    assert completed_records[0].id == job1

    # List only failed
    failed_records = await scheduler.list_records(status_filter=JobStatus.failed)
    assert len(failed_records) == 1
    assert failed_records[0].id == job2


async def test_list_records_reverse() -> None:
    """Test listing job records in reverse order."""
    scheduler = InMemoryScheduler()

    async def dummy() -> None:
        pass

    # Submit 3 jobs
    job1 = await scheduler.add_job(dummy)
    job2 = await scheduler.add_job(dummy)
    job3 = await scheduler.add_job(dummy)

    # List in normal order
    normal = await scheduler.list_records(reverse=False)
    assert [r.id for r in normal] == [job1, job2, job3]

    # List in reverse order
    reversed_list = await scheduler.list_records(reverse=True)
    assert [r.id for r in reversed_list] == [job3, job2, job1]


async def test_cancelled_job_updates_status() -> None:
    """Test that cancelling a job updates its status to canceled."""
    scheduler = InMemoryScheduler()

    async def long_task() -> None:
        await asyncio.sleep(10)  # Long running task

    job_id = await scheduler.add_job(long_task)

    # Give it a moment to start
    await asyncio.sleep(0.05)

    # Cancel the task
    async with scheduler._lock:
        task = scheduler._tasks[job_id]
        task.cancel()

    # Wait for cancellation to propagate
    await asyncio.sleep(0.1)

    # Check status
    record = await scheduler.get_record(job_id)
    assert record.status == JobStatus.canceled
    assert record.finished_at is not None


async def test_add_job_duplicate_id_raises_runtime_error() -> None:
    """Test that adding a job with duplicate ID raises RuntimeError."""
    import re

    scheduler = InMemoryScheduler()

    # Create a job with a specific ID by manually inserting into _tasks
    duplicate_id = ULID()

    # Manually add a task with this ID to simulate collision
    async with scheduler._lock:
        scheduler._tasks[duplicate_id] = asyncio.create_task(asyncio.sleep(10))

    # Try to add a new job with the same ID - this requires monkey-patching ULID generation
    import chapkit.scheduler

    old_ulid = chapkit.scheduler.ULID
    try:
        chapkit.scheduler.ULID = lambda: duplicate_id  # type: ignore[assignment,misc]

        async def dummy() -> None:
            pass

        # This should raise RuntimeError - use re.escape for the ULID repr
        with pytest.raises(RuntimeError, match=re.escape(f"Job {duplicate_id!r} already scheduled")):
            await scheduler.add_job(dummy)
    finally:
        chapkit.scheduler.ULID = old_ulid
        # Clean up the task we created
        async with scheduler._lock:
            if duplicate_id in scheduler._tasks:
                scheduler._tasks[duplicate_id].cancel()


async def test_add_job_with_awaitable_and_args_raises_type_error() -> None:
    """Test that submitting an awaitable with args/kwargs raises TypeError."""
    scheduler = InMemoryScheduler()

    # Create a coroutine (awaitable object)
    async def some_coro() -> str:
        return "result"

    awaitable = some_coro()

    # Add the awaitable with args - error happens during execution, not during add_job
    job_id = await scheduler.add_job(awaitable, "extra_arg")

    # Wait for job to execute and fail
    await asyncio.sleep(0.1)

    # Check that the job failed with TypeError
    record = await scheduler.get_record(job_id)
    assert record.status == JobStatus.failed
    assert "Args/kwargs not supported when target is an awaitable object" in (record.error or "")
