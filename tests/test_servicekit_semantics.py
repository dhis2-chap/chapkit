"""Regression tests for the CRUD, hierarchy and scheduler semantics introduced by servicekit 2.0.0."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from servicekit import Database, SqliteDatabaseBuilder
from servicekit.api.crud import CrudPermissions
from servicekit.schemas import JobStatus
from ulid import ULID

from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.artifact import ArtifactHierarchy, ArtifactIn, ArtifactManager, ArtifactRepository
from chapkit.scheduler import InMemoryChapkitScheduler

ROOT_ARTIFACT_ID = "01K72PWT05GEXK1S24AVKAZ9V1"
CHILD_ARTIFACT_ID = "01K72PWT05GEXK1S24AVKAZ9V2"
CONFIG_ID = "01K72PWT05GEXK1S24AVKAZ9V3"


class SampleConfig(BaseConfig):
    """Minimal config schema for the create-only router tests."""

    prediction_periods: int = 3
    label: str = "initial"


@pytest.fixture(name="create_only_client")
def create_only_client_fixture() -> Generator[TestClient, None, None]:
    """Serve config and artifact routers that allow create but not update."""
    permissions = CrudPermissions(create=True, read=True, update=False, delete=True)
    app = (
        ServiceBuilder(info=ServiceInfo(id="create-only", display_name="Create Only"))
        .with_health()
        .with_config(SampleConfig, permissions=permissions)
        .with_artifacts(hierarchy=ArtifactHierarchy(name="test"), permissions=permissions)
        .build()
    )

    with TestClient(app) as client:
        yield client


@pytest.fixture(name="artifact_manager")
async def artifact_manager_fixture() -> AsyncGenerator[tuple[Database, ArtifactManager], None]:
    """Provide an artifact manager backed by a throwaway in-memory database."""
    database = SqliteDatabaseBuilder.in_memory().build()
    await database.init()

    try:
        async with database.session() as session:
            yield database, ArtifactManager(ArtifactRepository(session), hierarchy=ArtifactHierarchy(name="test"))
    finally:
        await database.dispose()


def test_config_post_with_existing_id_returns_conflict(create_only_client: TestClient) -> None:
    """POSTing an existing config id conflicts and leaves the stored row untouched."""
    payload = {"id": CONFIG_ID, "name": "sample", "data": {"prediction_periods": 3, "label": "initial"}}
    created = create_only_client.post("/api/v1/configs", json=payload)
    assert created.status_code == 201

    conflicting = create_only_client.post(
        "/api/v1/configs",
        json={"id": CONFIG_ID, "name": "renamed", "data": {"prediction_periods": 9, "label": "changed"}},
    )
    assert conflicting.status_code == 409

    stored = create_only_client.get(f"/api/v1/configs/{CONFIG_ID}")
    assert stored.status_code == 200
    assert stored.json()["name"] == "sample"
    assert stored.json()["data"]["label"] == "initial"


def test_artifact_post_with_existing_id_returns_conflict(create_only_client: TestClient) -> None:
    """POSTing an existing artifact id conflicts and leaves the stored row untouched."""
    created = create_only_client.post("/api/v1/artifacts", json={"id": ROOT_ARTIFACT_ID, "data": {"step": "first"}})
    assert created.status_code == 201

    conflicting = create_only_client.post(
        "/api/v1/artifacts",
        json={"id": ROOT_ARTIFACT_ID, "data": {"step": "second"}},
    )
    assert conflicting.status_code == 409

    stored = create_only_client.get(f"/api/v1/artifacts/{ROOT_ARTIFACT_ID}")
    assert stored.status_code == 200
    assert stored.json()["data"] == {"step": "first"}


def test_artifact_list_rejects_out_of_range_pagination(create_only_client: TestClient) -> None:
    """The artifact list route validates pagination bounds like the servicekit CRUD router."""
    assert create_only_client.get("/api/v1/artifacts", params={"page": 0, "size": 10}).status_code == 422
    assert create_only_client.get("/api/v1/artifacts", params={"page": 1, "size": 1000}).status_code == 422


async def test_artifact_reparenting_with_explicit_null_parent(
    artifact_manager: tuple[Database, ArtifactManager],
) -> None:
    """An explicit null parent_id detaches a child, while omitting parent_id keeps it attached."""
    _, manager = artifact_manager

    root = await manager.save(ArtifactIn(id=ULID.from_str(ROOT_ARTIFACT_ID), data={"step": "root"}))
    child = await manager.save(
        ArtifactIn(id=ULID.from_str(CHILD_ARTIFACT_ID), data={"step": "child"}, parent_id=root.id)
    )
    assert child.level == 1

    kept = await manager.save(ArtifactIn.model_validate({"id": str(child.id), "data": {"step": "kept"}}))
    assert kept.parent_id == root.id
    assert kept.level == 1

    detached = await manager.save(
        ArtifactIn.model_validate({"id": str(child.id), "data": {"step": "detached"}, "parent_id": None})
    )
    assert detached.parent_id is None
    assert detached.level == 0


async def test_save_all_computes_levels_within_one_batch(
    artifact_manager: tuple[Database, ArtifactManager],
) -> None:
    """A batch containing a new root and its child resolves the child's level against the new root."""
    _, manager = artifact_manager

    root_id = ULID.from_str(ROOT_ARTIFACT_ID)
    saved = await manager.save_all(
        [
            ArtifactIn(id=root_id, data={"step": "root"}),
            ArtifactIn(id=ULID.from_str(CHILD_ARTIFACT_ID), data={"step": "child"}, parent_id=root_id),
        ]
    )

    assert [artifact.level for artifact in saved] == [0, 1]


async def test_canceled_queued_job_reaches_terminal_state() -> None:
    """Cancelling a queued job leaves a terminal canceled record with a finish timestamp."""
    scheduler = InMemoryChapkitScheduler(max_concurrency=1)
    release = asyncio.Event()

    async def blocking_job() -> None:
        await release.wait()

    async def queued_job() -> None:
        await asyncio.sleep(0)

    running_id = await scheduler.add_job(blocking_job)
    queued_id = await scheduler.add_job(queued_job)
    await asyncio.sleep(0)

    assert await scheduler.cancel(queued_id) is True

    record = await scheduler.get_record(queued_id)
    assert record.status is JobStatus.canceled
    assert record.finished_at is not None
    assert record.artifact_id is None

    release.set()
    await scheduler.wait(running_id)
    await scheduler.shutdown()


async def test_completed_job_returning_ulid_sets_artifact_id() -> None:
    """A completed job whose result is a ULID records it as the artifact id."""
    scheduler = InMemoryChapkitScheduler(max_concurrency=1)
    artifact_id = ULID()

    async def producing_job() -> ULID:
        return artifact_id

    async def plain_job() -> str:
        return "no artifact"

    producing_id = await scheduler.add_job(producing_job)
    plain_id = await scheduler.add_job(plain_job)
    await scheduler.wait(producing_id)
    await scheduler.wait(plain_id)

    assert (await scheduler.get_record(producing_id)).artifact_id == artifact_id
    assert (await scheduler.get_record(plain_id)).artifact_id is None

    await scheduler.shutdown()


async def test_get_record_returns_a_copy() -> None:
    """Mutating a returned record must not corrupt the scheduler's own state."""
    scheduler = InMemoryChapkitScheduler(max_concurrency=1)

    async def plain_job() -> str:
        return "done"

    job_id = await scheduler.add_job(plain_job)
    await scheduler.wait(job_id)

    record = await scheduler.get_record(job_id)
    record.status = JobStatus.failed

    assert (await scheduler.get_record(job_id)).status is JobStatus.completed

    await scheduler.shutdown()
