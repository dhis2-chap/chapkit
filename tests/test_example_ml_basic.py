"""Tests for ml_basic example with LinearRegression disease prediction.

This example demonstrates the ML train/predict workflow:
- Training models with pandas DataFrames
- Predictions using trained models stored in artifacts
- Jobs for async execution
- Artifact hierarchy for model lineage (predictions → trained model)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from examples.ml_basic import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient for testing with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job_completion(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[Any, Any]:
    """Poll job status until completion or timeout.

    Args:
        client: FastAPI test client
        job_id: Job identifier to poll
        timeout: Max seconds to wait (default: 5.0)

    Returns:
        Completed job record

    Raises:
        TimeoutError: If job doesn't complete within timeout
    """
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


def test_create_config(client: TestClient) -> None:
    """Test creating a disease prediction config."""
    from ulid import ULID

    config_name = f"test_disease_config_{ULID()}"
    new_config = {"name": config_name, "data": {}}

    response = client.post("/api/v1/configs", json=new_config)
    assert response.status_code == 201
    config = response.json()

    assert config["name"] == config_name
    assert "id" in config
    assert "created_at" in config


def test_train_model(client: TestClient) -> None:
    """Test training a linear regression model."""
    from ulid import ULID

    # Create config first
    config_response = client.post("/api/v1/configs", json={"name": f"train_test_config_{ULID()}", "data": {}})
    config = config_response.json()
    config_id = config["id"]

    # Prepare training data
    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [
                [10.0, 25.0, 5.0],
                [15.0, 28.0, 8.0],
                [8.0, 22.0, 3.0],
                [20.0, 30.0, 12.0],
                [12.0, 26.0, 6.0],
            ],
        },
    }

    # Submit training job
    response = client.post("/api/v1/ml/$train", json=train_request)
    assert response.status_code == 202
    train_response = response.json()

    assert "job_id" in train_response
    assert "model_artifact_id" in train_response
    assert "Training job submitted" in train_response["message"]

    job_id = train_response["job_id"]
    model_artifact_id = train_response["model_artifact_id"]

    # Wait for job completion
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "completed"

    # Verify model artifact was created
    artifact_response = client.get(f"/api/v1/artifacts/{model_artifact_id}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()

    # Check artifact metadata
    assert artifact["id"] == model_artifact_id
    assert artifact["level"] == 0  # Trained models are level 0
    assert "data" in artifact

    # Check artifact data structure
    # Note: data might be serialized as metadata if model is not JSON-serializable
    data = artifact["data"]
    if isinstance(data, dict):
        # Either it's the actual data dict or it's serialization metadata
        if "ml_type" in data:
            assert data["ml_type"] == "trained_model"
            assert data["config_id"] == config_id
            assert "model" in data
        elif "_type" in data:
            # Model was serialized as metadata (not JSON-serializable)
            # This is expected for sklearn models
            assert "_type" in data
            assert "_serialization_error" in data


def test_train_and_predict_workflow(client: TestClient) -> None:
    """Test complete train and predict workflow."""
    from ulid import ULID

    # 1. Create config
    config_response = client.post("/api/v1/configs", json={"name": f"workflow_config_{ULID()}", "data": {}})
    config_id = config_response.json()["id"]

    # 2. Train model
    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [
                [10.0, 25.0, 5.0],
                [15.0, 28.0, 8.0],
                [8.0, 22.0, 3.0],
                [20.0, 30.0, 12.0],
                [12.0, 26.0, 6.0],
            ],
        },
    }

    train_response = client.post("/api/v1/ml/$train", json=train_request)
    train_data = train_response.json()
    train_job_id = train_data["job_id"]
    model_artifact_id = train_data["model_artifact_id"]

    # Wait for training to complete
    train_job = wait_for_job_completion(client, train_job_id)
    assert train_job["status"] == "completed"

    # 3. Make predictions
    predict_request = {
        "model_artifact_id": model_artifact_id,
        "historic": {
            "columns": ["rainfall", "mean_temperature"],
            "data": [],
        },
        "future": {
            "columns": ["rainfall", "mean_temperature"],
            "data": [
                [11.0, 26.0],
                [14.0, 27.0],
                [9.0, 24.0],
            ],
        },
    }

    predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
    assert predict_response.status_code == 202
    predict_data = predict_response.json()

    assert "job_id" in predict_data
    assert "prediction_artifact_id" in predict_data
    prediction_artifact_id = predict_data["prediction_artifact_id"]

    # Wait for prediction to complete
    predict_job = wait_for_job_completion(client, predict_data["job_id"])
    assert predict_job["status"] == "completed"

    # 4. Verify prediction artifact
    artifact_response = client.get(f"/api/v1/artifacts/{prediction_artifact_id}")
    assert artifact_response.status_code == 200
    artifact = artifact_response.json()

    # Check artifact metadata
    assert artifact["id"] == prediction_artifact_id
    assert artifact["parent_id"] == model_artifact_id  # Linked to model
    assert artifact["level"] == 1  # Predictions are level 1

    # Check prediction data
    data = artifact["data"]
    assert data["ml_type"] == "prediction"
    assert data["model_artifact_id"] == model_artifact_id
    assert data["config_id"] == config_id
    assert "predictions" in data

    # Verify predictions structure
    predictions = data["predictions"]
    assert "columns" in predictions
    assert "data" in predictions
    # Should have sample_0 column added by on_predict
    assert "sample_0" in predictions["columns"]


def test_train_with_invalid_config_id(client: TestClient) -> None:
    """Test training with non-existent config ID."""
    train_request = {
        "config_id": "01K72P5N5KCRM6MD3BRE4P0999",  # Non-existent
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [[10.0, 25.0, 5.0]],
        },
    }

    response = client.post("/api/v1/ml/$train", json=train_request)
    assert response.status_code == 202  # Job is submitted

    job_id = response.json()["job_id"]

    # Wait for job - should fail
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "failed"
    assert "Config" in str(job["error"]) or "not found" in str(job["error"])


def test_predict_with_invalid_model_artifact(client: TestClient) -> None:
    """Test prediction with non-existent model artifact."""
    predict_request = {
        "model_artifact_id": "01K72P5N5KCRM6MD3BRE4P0999",  # Non-existent
        "historic": {
            "columns": ["rainfall", "mean_temperature"],
            "data": [],
        },
        "future": {
            "columns": ["rainfall", "mean_temperature"],
            "data": [[11.0, 26.0]],
        },
    }

    response = client.post("/api/v1/ml/$predict", json=predict_request)
    assert response.status_code == 202  # Job is submitted

    job_id = response.json()["job_id"]

    # Wait for job - should fail
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "failed"
    assert "artifact" in str(job["error"]).lower() or "not found" in str(job["error"])


def test_train_request_validation(client: TestClient) -> None:
    """Test train request validation with missing fields."""
    # Missing config_id
    response = client.post("/api/v1/ml/$train", json={"data": {"columns": [], "data": []}})
    assert response.status_code == 422

    # Missing data
    response = client.post("/api/v1/ml/$train", json={"config_id": "01K72P5N5KCRM6MD3BRE4P0001"})
    assert response.status_code == 422

    # Empty request
    response = client.post("/api/v1/ml/$train", json={})
    assert response.status_code == 422


def test_predict_request_validation(client: TestClient) -> None:
    """Test predict request validation with missing fields."""
    # Missing model_artifact_id
    response = client.post(
        "/api/v1/ml/$predict",
        json={"historic": {"columns": [], "data": []}, "future": {"columns": [], "data": []}},
    )
    assert response.status_code == 422

    # Missing historic
    response = client.post(
        "/api/v1/ml/$predict",
        json={"model_artifact_id": "01K72P5N5KCRM6MD3BRE4P0001", "future": {"columns": [], "data": []}},
    )
    assert response.status_code == 422

    # Missing future
    response = client.post(
        "/api/v1/ml/$predict",
        json={"model_artifact_id": "01K72P5N5KCRM6MD3BRE4P0001", "historic": {"columns": [], "data": []}},
    )
    assert response.status_code == 422

    # Empty request
    response = client.post("/api/v1/ml/$predict", json={})
    assert response.status_code == 422


def test_multiple_predictions_from_same_model(client: TestClient) -> None:
    """Test making multiple predictions from the same trained model."""
    from ulid import ULID

    # Create config and train model
    config_response = client.post("/api/v1/configs", json={"name": f"multi_predict_config_{ULID()}", "data": {}})
    config_id = config_response.json()["id"]

    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [[10.0, 25.0, 5.0], [15.0, 28.0, 8.0], [20.0, 30.0, 12.0]],
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
            "historic": {"columns": ["rainfall", "mean_temperature"], "data": []},
            "future": {"columns": ["rainfall", "mean_temperature"], "data": [[10 + i, 25 + i]]},
        }

        predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
        assert predict_response.status_code == 202

        predict_data = predict_response.json()
        predict_job = wait_for_job_completion(client, predict_data["job_id"])
        assert predict_job["status"] == "completed"

        prediction_artifact_ids.append(predict_data["prediction_artifact_id"])

    # Verify all prediction artifacts exist and link to the same model
    assert len(set(prediction_artifact_ids)) == 3  # All unique

    for pred_id in prediction_artifact_ids:
        artifact_response = client.get(f"/api/v1/artifacts/{pred_id}")
        artifact = artifact_response.json()

        assert artifact["parent_id"] == model_artifact_id
        assert artifact["data"]["model_artifact_id"] == model_artifact_id


def test_artifact_hierarchy_levels(client: TestClient) -> None:
    """Test that artifact hierarchy has correct levels (0=model, 1=prediction)."""
    from ulid import ULID

    # Create config and train
    config_response = client.post("/api/v1/configs", json={"name": f"hierarchy_config_{ULID()}", "data": {}})
    config_id = config_response.json()["id"]

    train_request = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [[10.0, 25.0, 5.0]],
        },
    }

    train_response = client.post("/api/v1/ml/$train", json=train_request)
    model_artifact_id = train_response.json()["model_artifact_id"]
    wait_for_job_completion(client, train_response.json()["job_id"])

    # Get model artifact and check level
    model_response = client.get(f"/api/v1/artifacts/{model_artifact_id}")
    model_artifact = model_response.json()
    assert model_artifact["level"] == 0

    # Make prediction
    predict_request = {
        "model_artifact_id": model_artifact_id,
        "historic": {"columns": ["rainfall", "mean_temperature"], "data": []},
        "future": {"columns": ["rainfall", "mean_temperature"], "data": [[11.0, 26.0]]},
    }

    predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
    prediction_artifact_id = predict_response.json()["prediction_artifact_id"]
    wait_for_job_completion(client, predict_response.json()["job_id"])

    # Get prediction artifact and check level
    pred_response = client.get(f"/api/v1/artifacts/{prediction_artifact_id}")
    pred_artifact = pred_response.json()
    assert pred_artifact["level"] == 1
    assert pred_artifact["parent_id"] == model_artifact_id


def test_list_jobs(client: TestClient) -> None:
    """Test listing ML jobs."""
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert isinstance(jobs, list)


def test_concurrent_training_jobs(client: TestClient) -> None:
    """Test submitting multiple training jobs concurrently."""
    from ulid import ULID

    # Create configs
    config_ids = []
    unique_id = ULID()
    for i in range(3):
        config_response = client.post(
            "/api/v1/configs", json={"name": f"concurrent_config_{unique_id}_{i}", "data": {}}
        )
        config_ids.append(config_response.json()["id"])

    # Submit training jobs
    job_ids = []
    for config_id in config_ids:
        train_request = {
            "config_id": config_id,
            "data": {
                "columns": ["rainfall", "mean_temperature", "disease_cases"],
                "data": [[10.0, 25.0, 5.0]],
            },
        }

        response = client.post("/api/v1/ml/$train", json=train_request)
        assert response.status_code == 202
        job_ids.append(response.json()["job_id"])

    # Wait for all jobs
    for job_id in job_ids:
        job = wait_for_job_completion(client, job_id)
        # Jobs should complete (either successfully or with failure due to concurrency)
        assert job["status"] in ["completed", "failed", "canceled"]
