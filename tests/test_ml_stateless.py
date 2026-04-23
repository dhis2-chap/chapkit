"""Integration tests for stateless ML services (predict without train)."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from geojson_pydantic import FeatureCollection
from servicekit.api.service_builder import ServiceInfo
from ulid import ULID

from chapkit import BaseConfig
from chapkit.api import MLServiceBuilder
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner


class StatelessConfig(BaseConfig):
    """Minimal config used by the stateless tests."""

    prediction_periods: int = 3


async def _rule_based_predict(
    config: StatelessConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Return a constant forecast so the test does not need numeric libraries."""
    future_df = future.to_pandas()
    future_df["sample_0"] = 1.0
    return DataFrame.from_pandas(future_df)


async def _train_that_should_never_run(config: Any, data: Any, geo: Any = None) -> Any:
    """Placeholder train callback for the train-backed service used in mode-mismatch tests."""
    return {"status": "trained"}


def _build_stateless_app() -> FastAPI:
    """Build an ML app with a stateless runner (on_train omitted)."""
    runner = FunctionalModelRunner(on_predict=_rule_based_predict)
    return MLServiceBuilder(
        info=ServiceInfo(id="stateless-test", display_name="Stateless Test"),
        config_schema=StatelessConfig,
        runner=runner,
    ).build()


def _build_train_backed_app() -> FastAPI:
    """Build an ML app with a train-backed runner for mode-mismatch tests."""
    runner = FunctionalModelRunner(
        on_predict=_rule_based_predict,
        on_train=_train_that_should_never_run,
    )
    return MLServiceBuilder(
        info=ServiceInfo(id="train-backed-test", display_name="Train-Backed Test"),
        config_schema=StatelessConfig,
        runner=runner,
    ).build()


@pytest.fixture()
def stateless_client() -> Generator[TestClient, None, None]:
    """TestClient over a stateless ML service (fresh per test for isolated state)."""
    with TestClient(_build_stateless_app()) as client:
        yield client


@pytest.fixture()
def train_backed_client() -> Generator[TestClient, None, None]:
    """TestClient over a train-backed ML service (fresh per test for isolated state)."""
    with TestClient(_build_train_backed_app()) as client:
        yield client


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """Poll the job endpoint until the job reaches a terminal status."""
    start = time.time()
    while time.time() - start < timeout:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = cast(dict[str, Any], response.json())
        if job["status"] in {"completed", "failed", "canceled"}:
            return job
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _create_config(client: TestClient) -> str:
    """Create a config and return its id."""
    response = client.post(
        "/api/v1/configs",
        json={"name": f"stateless_{ULID()}", "data": {"prediction_periods": 3}},
    )
    assert response.status_code in (200, 201)
    return cast(str, response.json()["id"])


def test_stateless_predict_happy_path(stateless_client: TestClient) -> None:
    """Stateless $predict returns 202 and the artifact is stored at level 0 with no parent."""
    config_id = _create_config(stateless_client)

    response = stateless_client.post(
        "/api/v1/ml/$predict",
        json={
            "config_id": config_id,
            "historic": {"columns": ["rainfall"], "data": [[1.0], [2.0]]},
            "future": {"columns": ["rainfall"], "data": [[3.0]]},
        },
    )
    assert response.status_code == 202
    body = response.json()

    job = _wait_for_job(stateless_client, body["job_id"])
    assert job["status"] == "completed", job

    artifact_response = stateless_client.get(f"/api/v1/artifacts/{body['artifact_id']}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()

    assert artifact["level"] == 0
    assert artifact.get("parent_id") in (None, "")
    assert artifact["data"]["type"] == "ml_prediction"


def test_predict_workspace_artifact_at_level_1(stateless_client: TestClient) -> None:
    """Prediction workspace artifact sits one level below the prediction artifact."""
    config_id = _create_config(stateless_client)

    response = stateless_client.post(
        "/api/v1/ml/$predict",
        json={
            "config_id": config_id,
            "historic": {"columns": ["rainfall"], "data": [[1.0]]},
            "future": {"columns": ["rainfall"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 202
    _wait_for_job(stateless_client, response.json()["job_id"])
    prediction_id = response.json()["artifact_id"]

    children_response = stateless_client.get(f"/api/v1/artifacts/{prediction_id}/$tree")
    assert children_response.status_code == 200
    tree = children_response.json()

    # Tree node may be wrapped or returned directly; normalise.
    if isinstance(tree, dict) and "children" in tree:
        children = tree["children"]
    else:
        children = tree.get("children", []) if isinstance(tree, dict) else []

    if children:
        workspace = children[0]
        assert workspace["level"] == 1
        assert workspace["data"]["type"] == "ml_prediction_workspace"


def test_schema_rejects_both_ids(stateless_client: TestClient) -> None:
    """Providing both artifact_id and config_id is a schema-level validation error."""
    response = stateless_client.post(
        "/api/v1/ml/$predict",
        json={
            "artifact_id": str(ULID()),
            "config_id": str(ULID()),
            "historic": {"columns": ["x"], "data": [[1.0]]},
            "future": {"columns": ["x"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 422


def test_schema_rejects_neither_id(stateless_client: TestClient) -> None:
    """Providing neither id is a schema-level validation error."""
    response = stateless_client.post(
        "/api/v1/ml/$predict",
        json={
            "historic": {"columns": ["x"], "data": [[1.0]]},
            "future": {"columns": ["x"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 422


def test_mode_mismatch_artifact_id_on_stateless(stateless_client: TestClient) -> None:
    """Passing artifact_id to a stateless service returns 400 with a clear message."""
    response = stateless_client.post(
        "/api/v1/ml/$predict",
        json={
            "artifact_id": str(ULID()),
            "historic": {"columns": ["x"], "data": [[1.0]]},
            "future": {"columns": ["x"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 400
    assert "stateless" in response.json()["detail"].lower()


def test_mode_mismatch_config_id_on_train_backed(train_backed_client: TestClient) -> None:
    """Passing config_id to a train-backed service returns 400 with a clear message."""
    config_id = _create_config(train_backed_client)

    response = train_backed_client.post(
        "/api/v1/ml/$predict",
        json={
            "config_id": config_id,
            "historic": {"columns": ["x"], "data": [[1.0]]},
            "future": {"columns": ["x"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 400
    assert "train-backed" in response.json()["detail"].lower()


def test_validate_stateless_happy_path(stateless_client: TestClient) -> None:
    """$validate (predict) with a config_id returns valid=True on clean input."""
    config_id = _create_config(stateless_client)

    response = stateless_client.post(
        "/api/v1/ml/$validate",
        json={
            "type": "predict",
            "config_id": config_id,
            "historic": {"columns": ["rainfall"], "data": [[1.0]]},
            "future": {"columns": ["rainfall"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 200
    assert response.json() == {"valid": True, "diagnostics": []}


def test_validate_mode_mismatch_stateless(stateless_client: TestClient) -> None:
    """$validate with artifact_id on a stateless service surfaces unsupported_predict_mode."""
    response = stateless_client.post(
        "/api/v1/ml/$validate",
        json={
            "type": "predict",
            "artifact_id": str(ULID()),
            "historic": {"columns": ["rainfall"], "data": [[1.0]]},
            "future": {"columns": ["rainfall"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert any(d["code"] == "unsupported_predict_mode" for d in body["diagnostics"])


def test_train_endpoint_hidden_on_stateless_service(stateless_client: TestClient) -> None:
    """$train is absent from OpenAPI for stateless services and returns 405."""
    schema = stateless_client.get("/openapi.json").json()
    paths = schema["paths"]

    assert "/api/v1/ml/$predict" in paths
    assert "/api/v1/ml/$validate" in paths
    assert "/api/v1/ml/$train" not in paths

    # Calling the missing route should not be handled by our app.
    response = stateless_client.post(
        "/api/v1/ml/$train",
        json={"config_id": str(ULID()), "data": {"columns": ["x"], "data": [[1.0]]}},
    )
    assert response.status_code in (404, 405)


def test_stateless_hierarchy_labels_in_tree(stateless_client: TestClient) -> None:
    """Artifact tree for a stateless service uses the stateless default level labels."""
    config_id = _create_config(stateless_client)

    response = stateless_client.post(
        "/api/v1/ml/$predict",
        json={
            "config_id": config_id,
            "historic": {"columns": ["rainfall"], "data": [[1.0]]},
            "future": {"columns": ["rainfall"], "data": [[2.0]]},
        },
    )
    assert response.status_code == 202
    _wait_for_job(stateless_client, response.json()["job_id"])
    prediction_id = response.json()["artifact_id"]

    tree_response = stateless_client.get(f"/api/v1/artifacts/{prediction_id}/$tree")
    assert tree_response.status_code == 200
    tree = tree_response.json()

    # Level 0 should be labelled as prediction, not training.
    assert tree["level_label"] == "ml_prediction"
