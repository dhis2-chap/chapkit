"""Integration tests for ml_class example with class-based runner."""

import time
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from examples.ml_class import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient for testing with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job_completion(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[Any, Any]:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        job_response = client.get(f"/api/v1/jobs/{job_id}")
        assert job_response.status_code == 200
        job = cast(dict[Any, Any], job_response.json())

        if job["status"] in ["completed", "failed", "canceled"]:
            return job

        time.sleep(0.1)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_train_with_class_runner(client: TestClient) -> None:
    """Test training with class-based runner."""
    from ulid import ULID

    # Create config
    config_response = client.post(
        "/api/v1/configs",
        json={"name": f"weather_config_{ULID()}", "data": {"min_samples": 3, "normalize_features": True}},
    )
    config = config_response.json()
    config_id = config["id"]

    # Train with normalized features
    train_request = {
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

    response = client.post("/api/v1/ml/$train", json=train_request)
    assert response.status_code == 202
    train_data = response.json()

    job_id = train_data["job_id"]
    model_artifact_id = train_data["model_artifact_id"]

    # Wait for training
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "completed"

    # Verify model artifact contains preprocessing artifacts
    artifact_response = client.get(f"/api/v1/artifacts/{model_artifact_id}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()
    assert artifact["level"] == 0


def test_train_and_predict_with_preprocessing(client: TestClient) -> None:
    """Test full workflow with preprocessing."""
    from ulid import ULID

    # Create config with normalization
    config_response = client.post(
        "/api/v1/configs",
        json={"name": f"preprocess_config_{ULID()}", "data": {"min_samples": 3, "normalize_features": True}},
    )
    config_id = config_response.json()["id"]

    # Train model
    train_request = {
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

    train_response = client.post("/api/v1/ml/$train", json=train_request)
    train_data = train_response.json()
    model_artifact_id = train_data["model_artifact_id"]

    # Wait for training
    train_job = wait_for_job_completion(client, train_data["job_id"])
    assert train_job["status"] == "completed"

    # Make predictions
    predict_request = {
        "model_artifact_id": model_artifact_id,
        "historic": {
            "columns": ["rainfall", "mean_temperature", "humidity"],
            "data": [],
        },
        "future": {
            "columns": ["rainfall", "mean_temperature", "humidity"],
            "data": [
                [12.0, 26.0, 65.0],
                [18.0, 29.0, 75.0],
            ],
        },
    }

    predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
    assert predict_response.status_code == 202
    predict_data = predict_response.json()

    # Wait for prediction
    predict_job = wait_for_job_completion(client, predict_data["job_id"])
    assert predict_job["status"] == "completed"

    # Verify prediction artifact
    prediction_artifact_id = predict_data["prediction_artifact_id"]
    artifact_response = client.get(f"/api/v1/artifacts/{prediction_artifact_id}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()

    assert artifact["parent_id"] == model_artifact_id
    assert artifact["level"] == 1


def test_train_with_insufficient_samples(client: TestClient) -> None:
    """Test training fails with insufficient samples."""
    from ulid import ULID

    # Create config with min_samples requirement
    config_response = client.post(
        "/api/v1/configs", json={"name": f"min_samples_config_{ULID()}", "data": {"min_samples": 10}}
    )
    config_id = config_response.json()["id"]

    # Try to train with too few samples
    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
            ],
        },
    }

    response = client.post("/api/v1/ml/$train", json=train_request)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # Job should fail
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "failed"
    assert "Insufficient training data" in str(job["error"])


def test_train_without_normalization(client: TestClient) -> None:
    """Test training without feature normalization."""
    from ulid import ULID

    # Create config without normalization, with lower min_samples
    config_response = client.post(
        "/api/v1/configs",
        json={"name": f"no_norm_config_{ULID()}", "data": {"normalize_features": False, "min_samples": 3}},
    )
    config_id = config_response.json()["id"]

    # Train model
    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }

    response = client.post("/api/v1/ml/$train", json=train_request)
    assert response.status_code == 202

    job_id = response.json()["job_id"]
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "completed"


def test_multiple_models_from_same_config(client: TestClient) -> None:
    """Test training multiple models from the same config."""
    from ulid import ULID

    # Create config with lower min_samples
    config_response = client.post(
        "/api/v1/configs", json={"name": f"multi_model_config_{ULID()}", "data": {"min_samples": 1}}
    )
    config_id = config_response.json()["id"]

    model_artifact_ids = []

    # Train multiple models
    for i in range(2):
        train_request = {
            "config_id": config_id,
            "data": {
                "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
                "data": [[10.0 + i, 25.0 + i, 60.0 + i, 5.0 + i]],
            },
        }

        response = client.post("/api/v1/ml/$train", json=train_request)
        assert response.status_code == 202

        train_data = response.json()
        job = wait_for_job_completion(client, train_data["job_id"])
        assert job["status"] == "completed"

        model_artifact_ids.append(train_data["model_artifact_id"])

    # Verify all models are unique
    assert len(set(model_artifact_ids)) == 2

    # All should be level 0
    for artifact_id in model_artifact_ids:
        artifact_response = client.get(f"/api/v1/artifacts/{artifact_id}")
        artifact = artifact_response.json()
        assert artifact["level"] == 0
