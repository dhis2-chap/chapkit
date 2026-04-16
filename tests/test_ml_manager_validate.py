"""Tests for MLManager.validate and _check_prediction_periods."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from ulid import ULID

from chapkit.config import BaseConfig
from chapkit.ml import MLManager


class SampleConfig(BaseConfig):
    """Test configuration schema."""

    prediction_periods: int = 3


class TestCheckPredictionPeriods:
    """Unit tests for _check_prediction_periods (returns diagnostics, does not raise)."""

    def _bare_manager(self, *, minimum: int, maximum: int) -> MLManager[BaseConfig]:
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = minimum
        manager.max_prediction_periods = maximum
        return manager

    def test_within_bounds_returns_empty(self) -> None:
        manager = self._bare_manager(minimum=1, maximum=10)
        assert manager._check_prediction_periods(SampleConfig(prediction_periods=5)) == []

    def test_below_minimum_returns_error_diagnostic(self) -> None:
        manager = self._bare_manager(minimum=5, maximum=10)
        diagnostics = manager._check_prediction_periods(SampleConfig(prediction_periods=3))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].code == "prediction_periods_out_of_bounds"
        assert diagnostics[0].field == "config.prediction_periods"
        assert "below the minimum" in diagnostics[0].message

    def test_above_maximum_returns_error_diagnostic(self) -> None:
        manager = self._bare_manager(minimum=1, maximum=5)
        diagnostics = manager._check_prediction_periods(SampleConfig(prediction_periods=10))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].code == "prediction_periods_out_of_bounds"
        assert "exceeds the maximum" in diagnostics[0].message

    def test_at_boundaries_returns_empty(self) -> None:
        manager = self._bare_manager(minimum=3, maximum=7)
        assert manager._check_prediction_periods(SampleConfig(prediction_periods=3)) == []
        assert manager._check_prediction_periods(SampleConfig(prediction_periods=7)) == []


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Spin up the ml_class example app for validate integration tests."""
    pytest.importorskip("pandas")
    from examples.ml_class.main import app

    with TestClient(app) as test_client:
        yield test_client


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[Any, Any]:
    start = time.time()
    while time.time() - start < timeout:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = cast(dict[Any, Any], response.json())
        if job["status"] in ["completed", "failed", "canceled"]:
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _create_config(client: TestClient, *, min_samples: int = 3, prediction_periods: int = 3) -> str:
    response = client.post(
        "/api/v1/configs",
        json={
            "name": f"validate_config_{ULID()}",
            "data": {
                "min_samples": min_samples,
                "normalize_features": True,
                "prediction_periods": prediction_periods,
            },
        },
    )
    assert response.status_code == 201, response.text
    return cast(str, response.json()["id"])


def test_validate_train_happy_path(client: TestClient) -> None:
    """A well-formed train payload produces valid=True."""
    config_id = _create_config(client)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [8.0, 22.0, 55.0, 3.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["diagnostics"] == []


def test_validate_train_config_not_found(client: TestClient) -> None:
    """Non-existent config produces a config_not_found diagnostic, not a 404."""
    body = {
        "type": "train",
        "config_id": str(ULID()),
        "data": {"columns": ["rainfall"], "data": [[1.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "config_not_found" in codes


def test_validate_train_data_empty(client: TestClient) -> None:
    """Empty training data produces a data_empty diagnostic."""
    config_id = _create_config(client)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {"columns": ["rainfall"], "data": []},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "data_empty" in codes


def test_validate_predict_artifact_not_found(client: TestClient) -> None:
    """Non-existent artifact produces a training_artifact_not_found diagnostic."""
    body = {
        "type": "predict",
        "artifact_id": str(ULID()),
        "historic": {"columns": ["rainfall"], "data": [[1.0]]},
        "future": {"columns": ["rainfall"], "data": [[2.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "training_artifact_not_found" in codes


def test_validate_predict_happy_path(client: TestClient) -> None:
    """After a real training run the predict validate call returns valid=True."""
    config_id = _create_config(client)
    train_body = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [8.0, 22.0, 55.0, 3.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }

    train_response = client.post("/api/v1/ml/$train", json=train_body)
    assert train_response.status_code == 202
    train_data = train_response.json()
    job = _wait_for_job(client, train_data["job_id"])
    assert job["status"] == "completed", job

    validate_body = {
        "type": "predict",
        "artifact_id": train_data["artifact_id"],
        "historic": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [[10.0, 25.0, 60.0, 5.0]],
        },
        "future": {
            "columns": ["rainfall", "mean_temperature", "humidity"],
            "data": [[11.0, 26.0, 62.0]],
        },
    }

    response = client.post("/api/v1/ml/$validate", json=validate_body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True, payload


def test_validate_predict_empty_historic_and_future(client: TestClient) -> None:
    """Empty historic and future both produce diagnostics after a valid artifact."""
    config_id = _create_config(client)
    train_body = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [8.0, 22.0, 55.0, 3.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }
    train_response = client.post("/api/v1/ml/$train", json=train_body)
    train_data = train_response.json()
    job = _wait_for_job(client, train_data["job_id"])
    assert job["status"] == "completed", job

    validate_body = {
        "type": "predict",
        "artifact_id": train_data["artifact_id"],
        "historic": {"columns": ["rainfall"], "data": []},
        "future": {"columns": ["rainfall"], "data": []},
    }

    response = client.post("/api/v1/ml/$validate", json=validate_body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "historic_empty" in codes
    assert "future_empty" in codes


def test_validate_train_runner_diagnostic_flows_through(client: TestClient) -> None:
    """A domain diagnostic from on_validate_train reaches the response."""
    # WeatherConfig.min_samples defaults to 5 in the example; WeatherModelRunner's
    # on_validate_train override (see examples/ml_class/main.py) emits an error
    # diagnostic when data has fewer rows than min_samples.
    config_id = _create_config(client, min_samples=10)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
            ],
        },
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "insufficient_training_samples" in codes
    assert payload["valid"] is False
