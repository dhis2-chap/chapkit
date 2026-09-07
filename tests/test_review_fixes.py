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
