"""Integration tests for ml_shell example with shell-based runner."""

import time
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from examples.ml_shell import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient for testing with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job_completion(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[Any, Any]:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        job_response = client.get(f"/api/v1/jobs/{job_id}")
        assert job_response.status_code == 200
        job = cast(dict[Any, Any], job_response.json())

        if job["status"] in ["completed", "failed", "canceled"]:
            return job

        time.sleep(0.2)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_train_with_shell_runner(client: TestClient) -> None:
    """Test training with external Python script."""
    from ulid import ULID

    # Create config
    config_response = client.post(
        "/api/v1/configs",
        json={"name": f"shell_config_{ULID()}", "data": {"min_samples": 3, "model_type": "linear_regression"}},
    )
    config = config_response.json()
    config_id = config["id"]

    # Train with external script
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
    assert job["status"] == "completed", f"Job failed: {job.get('error')}"

    # Verify model artifact
    artifact_response = client.get(f"/api/v1/artifacts/{model_artifact_id}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()
    assert artifact["level"] == 0


def test_train_and_predict_with_external_scripts(client: TestClient) -> None:
    """Test full workflow with external train and predict scripts."""
    from ulid import ULID

    # Create config
    config_response = client.post(
        "/api/v1/configs",
        json={"name": f"shell_workflow_config_{ULID()}", "data": {"model_type": "linear_regression"}},
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
    assert train_job["status"] == "completed", f"Training failed: {train_job.get('error')}"

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
    assert predict_job["status"] == "completed", f"Prediction failed: {predict_job.get('error')}"

    # Verify prediction artifact
    prediction_artifact_id = predict_data["prediction_artifact_id"]
    artifact_response = client.get(f"/api/v1/artifacts/{prediction_artifact_id}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()

    assert artifact["parent_id"] == model_artifact_id
    assert artifact["level"] == 1

    # Check predictions data
    data = artifact["data"]
    assert data["ml_type"] == "prediction"
    assert "predictions" in data
    predictions = data["predictions"]
    assert "sample_0" in predictions["columns"]


def test_train_with_minimal_data(client: TestClient) -> None:
    """Test training with minimal dataset."""
    from ulid import ULID

    # Create config with lower min_samples
    config_response = client.post(
        "/api/v1/configs", json={"name": f"minimal_config_{ULID()}", "data": {"min_samples": 1}}
    )
    config_id = config_response.json()["id"]

    # Train with just one sample
    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [[10.0, 25.0, 60.0, 5.0]],
        },
    }

    response = client.post("/api/v1/ml/$train", json=train_request)
    assert response.status_code == 202

    job_id = response.json()["job_id"]
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "completed"


def test_multiple_predictions_from_shell_model(client: TestClient) -> None:
    """Test making multiple predictions from the same shell-trained model."""
    from ulid import ULID

    # Create config and train
    config_response = client.post("/api/v1/configs", json={"name": f"multi_predict_shell_{ULID()}", "data": {}})
    config_id = config_response.json()["id"]

    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [[10.0, 25.0, 60.0, 5.0], [15.0, 28.0, 70.0, 8.0], [20.0, 30.0, 80.0, 12.0]],
        },
    }

    train_response = client.post("/api/v1/ml/$train", json=train_request)
    model_artifact_id = train_response.json()["model_artifact_id"]
    train_job = wait_for_job_completion(client, train_response.json()["job_id"])
    assert train_job["status"] == "completed"

    # Make multiple predictions
    prediction_artifact_ids = []

    for i in range(3):
        predict_request = {
            "model_artifact_id": model_artifact_id,
            "historic": {"columns": ["rainfall", "mean_temperature", "humidity"], "data": []},
            "future": {"columns": ["rainfall", "mean_temperature", "humidity"], "data": [[10 + i, 25 + i, 60 + i]]},
        }

        predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
        assert predict_response.status_code == 202

        predict_data = predict_response.json()
        predict_job = wait_for_job_completion(client, predict_data["job_id"])
        assert predict_job["status"] == "completed"

        prediction_artifact_ids.append(predict_data["prediction_artifact_id"])

    # Verify all predictions are unique and linked to the same model
    assert len(set(prediction_artifact_ids)) == 3

    for pred_id in prediction_artifact_ids:
        artifact_response = client.get(f"/api/v1/artifacts/{pred_id}")
        artifact = artifact_response.json()
        assert artifact["parent_id"] == model_artifact_id


def test_concurrent_shell_training_jobs(client: TestClient) -> None:
    """Test submitting multiple training jobs concurrently with shell runner."""
    from ulid import ULID

    # Create configs
    config_ids = []
    unique_id = ULID()
    for i in range(2):
        config_response = client.post("/api/v1/configs", json={"name": f"concurrent_shell_{unique_id}_{i}", "data": {}})
        config_ids.append(config_response.json()["id"])

    # Submit training jobs
    job_ids = []
    for config_id in config_ids:
        train_request = {
            "config_id": config_id,
            "data": {
                "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
                "data": [[10.0, 25.0, 60.0, 5.0]],
            },
        }

        response = client.post("/api/v1/ml/$train", json=train_request)
        assert response.status_code == 202
        job_ids.append(response.json()["job_id"])

    # Wait for all jobs
    for job_id in job_ids:
        job = wait_for_job_completion(client, job_id)
        assert job["status"] in ["completed", "failed", "canceled"]
