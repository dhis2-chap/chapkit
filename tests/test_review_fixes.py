"""Regression tests for the pre-release review findings (config data, cycles, permissions, cancellation)."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from ulid import ULID

from chapkit import ArtifactHierarchy, BaseConfig, run_shell
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.cli.migrate import _migrate_chapkit_floor
from chapkit.ml.runner import ShellModelRunner

_seed_counter = itertools.count(1)


class Nested(BaseModel):
    """Nested model used to exercise schema definitions."""

    depth: int = 1


class ReviewConfig(BaseConfig):
    """Config with a default, a default factory and a nested model."""

    threshold: float = 0.5
    seed: int = Field(default_factory=lambda: next(_seed_counter))
    nested: Nested = Field(default_factory=Nested)


def _build_app(**config_permissions: Any):
    """Build a config plus artifact service with config linking enabled."""
    return (
        ServiceBuilder(info=ServiceInfo(id="review", display_name="Review"))
        .with_health()
        .with_config(ReviewConfig, **config_permissions)
        .with_artifacts(
            hierarchy=ArtifactHierarchy(name="h", level_labels={0: "root", 1: "child"}),
            enable_config_linking=True,
        )
        .build()
    )


def _create_config(client: TestClient, name: str) -> str:
    """Create a config with only the required field set and return its id."""
    response = client.post("/api/v1/configs", json={"name": name, "data": {"prediction_periods": 3}})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def test_config_defaults_are_persisted_on_create_and_update() -> None:
    """Defaults and default factories are frozen at write time, not re-evaluated on every read."""
    with TestClient(_build_app()) as client:
        config_id = _create_config(client, "defaults")
        created = client.post("/api/v1/configs", json={"name": "x", "data": {"prediction_periods": 1}}).json()
        first = client.get(f"/api/v1/configs/{config_id}").json()["data"]
        second = client.get(f"/api/v1/configs/{config_id}").json()["data"]

        assert first == second
        assert first["threshold"] == 0.5
        assert first["nested"] == {"depth": 1}
        assert created["data"]["seed"] != first["seed"]

        updated = client.put(
            f"/api/v1/configs/{config_id}",
            json={"id": config_id, "name": "defaults", "data": {"prediction_periods": 7}},
        )
        assert updated.status_code == 200, updated.text
        after_update = client.get(f"/api/v1/configs/{config_id}").json()["data"]
        assert after_update["prediction_periods"] == 7
        assert after_update == client.get(f"/api/v1/configs/{config_id}").json()["data"]


def test_config_schema_keeps_nested_definitions() -> None:
    """The $schema operation returns a self-contained document including referenced $defs."""
    with TestClient(_build_app()) as client:
        schema = client.get("/api/v1/configs/$schema").json()

    assert schema["properties"]["nested"] == {"$ref": "#/$defs/Nested"}
    assert "Nested" in schema["$defs"]
    assert schema["$defs"]["Nested"]["properties"]["depth"]["default"] == 1


def test_artifact_rejects_self_parent_and_reparenting_under_descendant() -> None:
    """Cycles are rejected on create and on update, and tree traversal still terminates."""
    with TestClient(_build_app()) as client:
        artifact_id = str(ULID())
        response = client.post("/api/v1/artifacts", json={"id": artifact_id, "parent_id": artifact_id, "data": {}})
        assert response.status_code == 400
        assert "own parent" in response.text

        root = client.post("/api/v1/artifacts", json={"data": {"n": "root"}}).json()["id"]
        child = client.post("/api/v1/artifacts", json={"parent_id": root, "data": {"n": "child"}}).json()["id"]
        grandchild = client.post("/api/v1/artifacts", json={"parent_id": child, "data": {"n": "gc"}}).json()["id"]

        response = client.put(f"/api/v1/artifacts/{root}", json={"id": root, "parent_id": grandchild, "data": {}})
        assert response.status_code == 400
        assert "descendants" in response.text

        tree = client.get(f"/api/v1/artifacts/{root}/$tree")
        assert tree.status_code == 200
        assert tree.json()["children"][0]["children"][0]["id"] == grandchild


def test_unlink_is_scoped_to_the_config_in_the_url() -> None:
    """Unlinking through another config does not remove the link, and reports 404."""
    with TestClient(_build_app()) as client:
        config_a = _create_config(client, "a")
        config_b = _create_config(client, "b")
        artifact = client.post("/api/v1/artifacts", json={"data": {}}).json()["id"]

        assert (
            client.post(f"/api/v1/configs/{config_a}/$link-artifact", json={"artifact_id": artifact}).status_code == 204
        )
        wrong = client.post(f"/api/v1/configs/{config_b}/$unlink-artifact", json={"artifact_id": artifact})
        assert wrong.status_code == 404
        assert [a["id"] for a in client.get(f"/api/v1/configs/{config_a}/$artifacts").json()] == [artifact]

        right = client.post(f"/api/v1/configs/{config_a}/$unlink-artifact", json={"artifact_id": artifact})
        assert right.status_code == 204
        assert client.get(f"/api/v1/configs/{config_a}/$artifacts").json() == []


def test_config_artifact_operations_follow_crud_permissions() -> None:
    """Link and unlink are only registered when updates are allowed, listing when reads are allowed."""
    with TestClient(_build_app(allow_update=False)) as client:
        config_id = _create_config(client, "locked")
        artifact = client.post("/api/v1/artifacts", json={"data": {}}).json()["id"]
        # Unregistered operations fall through to the generic entity routes, which answer 404 or 405.
        link = client.post(f"/api/v1/configs/{config_id}/$link-artifact", json={"artifact_id": artifact})
        assert link.status_code in (404, 405)
        unlink = client.post(f"/api/v1/configs/{config_id}/$unlink-artifact", json={"artifact_id": artifact})
        assert unlink.status_code in (404, 405)
        assert client.get(f"/api/v1/configs/{config_id}/$artifacts").status_code == 200

    with TestClient(_build_app(allow_read=False)) as client:
        config_id = _create_config(client, "unreadable")
        assert client.get(f"/api/v1/configs/{config_id}/$artifacts").status_code in (404, 405)


@pytest.mark.asyncio
async def test_run_shell_cancellation_kills_the_process_tree(tmp_path: Path) -> None:
    """Canceling the awaiting task terminates the shell and its children instead of orphaning them."""
    marker = tmp_path / "done"
    task = asyncio.create_task(run_shell(f"sleep 1; touch {marker}", cwd=str(tmp_path)))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.3)
    assert not marker.exists()


def test_migrate_floor_follows_the_running_major() -> None:
    """The migrate floor stays at the fixed minimum within a major and lifts to N.0.0 for a newer major."""
    assert _migrate_chapkit_floor("1.2.0.dev0") == "1.1.0"
    assert _migrate_chapkit_floor("1.9.3") == "1.1.0"
    assert _migrate_chapkit_floor("2.0.0.dev0") == "2.0.0"
    assert _migrate_chapkit_floor("2.3.1") == "2.0.0"
    assert _migrate_chapkit_floor("unknown") == "1.1.0"


@pytest.mark.asyncio
async def test_failed_prediction_keeps_workspace_when_output_is_unparseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing script that leaves a broken predictions.csv still yields the diagnostic workspace."""
    from chapkit.data import DataFrame

    def broken_csv(path: Any) -> DataFrame:
        raise IndexError("list index out of range")

    monkeypatch.setattr("chapkit.ml.runner.DataFrame.from_csv", broken_csv)

    project = tmp_path / "project"
    project.mkdir()
    runner: ShellModelRunner[BaseConfig] = ShellModelRunner(
        train_command="true",
        predict_command=": > predictions.csv; exit 3",
    )
    runner.project_root = project
    frame = DataFrame(columns=["time_period", "location", "disease_cases"], data=[["2024-01", "a", 1.0]])
    config = ReviewConfig(prediction_periods=1)
    trained = await runner.on_train(config, frame)
    result = await runner.on_predict(config, trained, frame, frame)

    assert result["exit_code"] == 3
    assert result["content"] is None
    assert Path(result["workspace_dir"]).is_dir()


class _FakeProcess:
    """Minimal stand-in for an asyncio subprocess used to exercise kill fallbacks."""

    def __init__(self, pid: int) -> None:
        """Record the pid and start with no kill calls."""
        self.pid = pid
        self.killed = False

    def kill(self) -> None:
        """Record that the direct kill fallback was used."""
        self.killed = True

    async def wait(self) -> int:
        """Pretend the process has been reaped."""
        return -9


@pytest.mark.asyncio
async def test_kill_process_group_tolerates_missing_or_unkillable_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vanished process group is ignored and a forbidden one falls back to killing the shell."""
    from chapkit import utils

    def missing(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(utils.os, "killpg", missing)
    gone = _FakeProcess(pid=1)
    await utils._kill_process_group(gone)  # type: ignore[arg-type]
    assert gone.killed is False

    def forbidden(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(utils.os, "killpg", forbidden)
    stubborn = _FakeProcess(pid=2)
    await utils._kill_process_group(stubborn)  # type: ignore[arg-type]
    assert stubborn.killed is True


@pytest.mark.asyncio
async def test_run_shell_timeout_kills_the_process_tree(tmp_path: Path) -> None:
    """A timed-out command is killed as a group and reported with a negative return code."""
    marker = tmp_path / "late"
    result = await run_shell(f"sleep 1; touch {marker}", cwd=str(tmp_path), timeout=0.2)
    assert result["returncode"] == -1
    assert "timed out" in result["stderr"]
    await asyncio.sleep(1.1)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_ensure_acyclic_walks_up_and_stops_at_missing_ancestors() -> None:
    """The cycle guard accepts a dangling parent chain and rejects one that loops back to the entity."""
    from types import SimpleNamespace
    from typing import cast

    from servicekit.exceptions import BadRequestError

    from chapkit import ArtifactManager, ArtifactRepository

    root, child, orphan_parent = ULID(), ULID(), ULID()
    rows = {child: SimpleNamespace(id=child, parent_id=root)}

    class Repo:
        """Repository stub answering find_by_id from a dict."""

        async def find_by_id(self, artifact_id: ULID) -> Any:
            """Return the stubbed row, or None for unknown ids."""
            return rows.get(artifact_id)

    manager = ArtifactManager(cast(ArtifactRepository, Repo()))

    # Parent chain ends at an unknown ancestor: nothing to reject.
    await manager._ensure_acyclic(cast(Any, SimpleNamespace(id=ULID(), parent_id=orphan_parent)))
    # No parent at all: nothing to check.
    await manager._ensure_acyclic(cast(Any, SimpleNamespace(id=ULID(), parent_id=None)))
    # Moving root beneath its own child loops back to root.
    with pytest.raises(BadRequestError, match="descendants"):
        await manager._ensure_acyclic(cast(Any, SimpleNamespace(id=root, parent_id=child)))


def test_artifact_read_operations_follow_read_permission() -> None:
    """Tree, expand and config lookups are not registered when reads are disallowed."""
    app = (
        ServiceBuilder(info=ServiceInfo(id="review-art", display_name="Review"))
        .with_health()
        .with_config(ReviewConfig)
        .with_artifacts(
            hierarchy=ArtifactHierarchy(name="h", level_labels={0: "root"}),
            enable_config_linking=True,
            allow_read=False,
        )
        .build()
    )
    with TestClient(app) as client:
        artifact = client.post("/api/v1/artifacts", json={"data": {}}).json()["id"]
        assert client.get(f"/api/v1/artifacts/{artifact}/$tree").status_code in (404, 405)
        assert client.get(f"/api/v1/artifacts/{artifact}/$expand").status_code in (404, 405)
        assert client.get(f"/api/v1/artifacts/{artifact}/$config").status_code in (404, 405)


@pytest.mark.asyncio
async def test_shell_predict_cleans_workspace_on_python_error(tmp_path: Path) -> None:
    """A Python-level failure before the script runs removes the temporary workspace."""
    import chapkit.ml.runner as runner_module

    runner: ShellModelRunner[BaseConfig] = ShellModelRunner(train_command="true", predict_command="true")
    runner.project_root = tmp_path
    before = set(Path(runner_module.tempfile.gettempdir()).glob("chapkit_ml_predict_*"))
    with pytest.raises(ValueError, match="workspace artifact"):
        await runner.on_predict(ReviewConfig(prediction_periods=1), {"not": "a workspace"}, None, None)  # type: ignore[arg-type]
    after = set(Path(runner_module.tempfile.gettempdir()).glob("chapkit_ml_predict_*"))
    assert after == before


@pytest.mark.asyncio
async def test_config_manager_unlink_reports_missing_link() -> None:
    """Unlinking an artifact that is not linked to the config raises NotFoundError and links stay intact."""
    from servicekit import SqliteDatabaseBuilder
    from servicekit.exceptions import NotFoundError

    from chapkit import Artifact, ArtifactRepository
    from chapkit.config import ConfigIn, ConfigManager, ConfigRepository

    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        async with db.session() as session:
            manager = ConfigManager[ReviewConfig](ConfigRepository(session), ReviewConfig)
            config = await manager.save(ConfigIn[ReviewConfig](name="a", data=ReviewConfig(prediction_periods=1)))
            artifact_repository = ArtifactRepository(session)
            artifact = Artifact(parent_id=None, data={})
            await artifact_repository.save(artifact)
            await artifact_repository.commit()
            await artifact_repository.refresh_many([artifact])
            await manager.link_artifact(config.id, artifact.id)

            with pytest.raises(NotFoundError):
                await manager.unlink_artifact(ULID(), artifact.id)
            assert [a.id for a in await manager.get_linked_artifacts(config.id)] == [artifact.id]

            await manager.unlink_artifact(config.id, artifact.id)
            assert await manager.get_linked_artifacts(config.id) == []
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_shell_train_cancellation_removes_workspace_and_process(tmp_path: Path) -> None:
    """Canceling a training run kills the script and removes the temporary workspace."""
    import chapkit.ml.runner as runner_module
    from chapkit.data import DataFrame

    marker = tmp_path / "trained"
    runner: ShellModelRunner[BaseConfig] = ShellModelRunner(
        train_command=f"sleep 1; touch {marker}",
        predict_command="true",
    )
    runner.project_root = tmp_path
    frame = DataFrame(columns=["time_period", "location", "disease_cases"], data=[["2024-01", "a", 1.0]])
    before = set(Path(runner_module.tempfile.gettempdir()).glob("chapkit_ml_train_*"))

    task = asyncio.create_task(runner.on_train(ReviewConfig(prediction_periods=1), frame))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    after = set(Path(runner_module.tempfile.gettempdir()).glob("chapkit_ml_train_*"))
    assert after == before
    await asyncio.sleep(1.0)
    assert not marker.exists()
