"""Tests for artifact download and metadata endpoints."""

from typing import Any

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client using ML functional example."""
    from examples.ml_functional.main import app

    # The example app initializes the database on lifespan startup
    # TestClient handles this automatically
    with TestClient(app) as test_client:
        yield test_client


def wait_for_job_completion(client: TestClient, job_id: str, max_attempts: int = 10) -> dict[str, Any]:
    """Wait for job to complete."""
    import time
    from typing import cast

    for _ in range(max_attempts):
        response = client.get(f"/api/v1/jobs/{job_id}")
        job = cast(dict[str, Any], response.json())
        if job["status"] in ["completed", "failed"]:
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete")


def test_download_prediction_artifact_as_csv(client: TestClient):
    """Test downloading prediction artifact as CSV."""
    from ulid import ULID

    # Create config
    config_response = client.post("/api/v1/configs", json={"name": f"test_download_{ULID()}", "data": {}})
    assert config_response.status_code == 201
    config_id = config_response.json()["id"]

    # Train model
    train_request = {
        "config_id": config_id,
            "run_info": {"prediction_length": 3},
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [[10.0, 25.0, 5.0], [12.0, 26.0, 6.0], [8.0, 24.0, 4.0]],
        },
    }
    train_response = client.post("/api/v1/ml/$train", json=train_request)
    assert train_response.status_code == 202
    train_data = train_response.json()

    # Wait for training
    train_job = wait_for_job_completion(client, train_data["job_id"])
    assert train_job["status"] == "completed"
    model_artifact_id = train_data["artifact_id"]

    # Predict
    predict_request = {
        "artifact_id": model_artifact_id,
        "run_info": {"prediction_length": 3},
            "historic": {"columns": ["rainfall", "mean_temperature"], "data": []},
        "future": {"columns": ["rainfall", "mean_temperature"], "data": [[11.0, 25.5], [9.0, 24.5]]},
    }
    predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
    assert predict_response.status_code == 202
    predict_data = predict_response.json()

    # Wait for prediction
    predict_job = wait_for_job_completion(client, predict_data["job_id"])
    assert predict_job["status"] == "completed"
    prediction_artifact_id = predict_data["artifact_id"]

    # Test download endpoint - predictions are DataFrames with content_type application/vnd.chapkit.dataframe+json
    # which defaults to JSON serialization
    download_response = client.get(f"/api/v1/artifacts/{prediction_artifact_id}/$download")
    if download_response.status_code != 200:
        print(f"Download failed with: {download_response.json()}")
    assert download_response.status_code == 200

    # Should be JSON format (default for DataFrame)
    assert download_response.headers["content-type"] == "application/vnd.chapkit.dataframe+json"
    assert "attachment" in download_response.headers["content-disposition"]
    assert f"artifact_{prediction_artifact_id}" in download_response.headers["content-disposition"]

    # Content should be DataFrame JSON (orient="records" format - array of objects)
    import json

    predictions = json.loads(download_response.content)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert "sample_0" in predictions[0]


def test_get_metadata_from_training_artifact(client: TestClient):
    """Test getting metadata from training artifact."""
    from ulid import ULID

    # Create config
    config_response = client.post("/api/v1/configs", json={"name": f"test_metadata_{ULID()}", "data": {}})
    assert config_response.status_code == 201
    config_id = config_response.json()["id"]

    # Train model
    train_request = {
        "config_id": config_id,
            "run_info": {"prediction_length": 3},
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [[10.0, 25.0, 5.0], [12.0, 26.0, 6.0], [8.0, 24.0, 4.0]],
        },
    }
    train_response = client.post("/api/v1/ml/$train", json=train_request)
    assert train_response.status_code == 202
    train_data = train_response.json()

    # Wait for training
    train_job = wait_for_job_completion(client, train_data["job_id"])
    assert train_job["status"] == "completed"
    model_artifact_id = train_data["artifact_id"]

    # Test metadata endpoint
    metadata_response = client.get(f"/api/v1/artifacts/{model_artifact_id}/$metadata")
    assert metadata_response.status_code == 200

    metadata = metadata_response.json()

    # Check metadata structure
    assert "status" in metadata
    assert "config_id" in metadata
    assert "started_at" in metadata
    assert "completed_at" in metadata
    assert "duration_seconds" in metadata

    # Check values
    assert metadata["status"] == "success"
    assert metadata["config_id"] == config_id
    assert isinstance(metadata["duration_seconds"], (int, float))
    assert metadata["duration_seconds"] >= 0


def test_get_metadata_from_prediction_artifact(client: TestClient):
    """Test getting metadata from prediction artifact."""
    from ulid import ULID

    # Create config
    config_response = client.post("/api/v1/configs", json={"name": f"test_pred_metadata_{ULID()}", "data": {}})
    assert config_response.status_code == 201
    config_id = config_response.json()["id"]

    # Train model
    train_request = {
        "config_id": config_id,
            "run_info": {"prediction_length": 3},
        "data": {
            "columns": ["rainfall", "mean_temperature", "disease_cases"],
            "data": [[10.0, 25.0, 5.0], [12.0, 26.0, 6.0], [8.0, 24.0, 4.0]],
        },
    }
    train_response = client.post("/api/v1/ml/$train", json=train_request)
    assert train_response.status_code == 202
    train_data = train_response.json()

    # Wait for training
    train_job = wait_for_job_completion(client, train_data["job_id"])
    assert train_job["status"] == "completed"
    model_artifact_id = train_data["artifact_id"]

    # Predict
    predict_request = {
        "artifact_id": model_artifact_id,
        "run_info": {"prediction_length": 3},
            "historic": {"columns": ["rainfall", "mean_temperature"], "data": []},
        "future": {"columns": ["rainfall", "mean_temperature"], "data": [[11.0, 25.5]]},
    }
    predict_response = client.post("/api/v1/ml/$predict", json=predict_request)
    assert predict_response.status_code == 202
    predict_data = predict_response.json()

    # Wait for prediction
    predict_job = wait_for_job_completion(client, predict_data["job_id"])
    assert predict_job["status"] == "completed"
    prediction_artifact_id = predict_data["artifact_id"]

    # Test metadata endpoint
    metadata_response = client.get(f"/api/v1/artifacts/{prediction_artifact_id}/$metadata")
    assert metadata_response.status_code == 200

    metadata = metadata_response.json()

    # Check metadata structure
    assert "status" in metadata
    assert "config_id" in metadata
    assert "started_at" in metadata
    assert "completed_at" in metadata
    assert "duration_seconds" in metadata

    # Check values
    assert metadata["status"] == "success"
    assert metadata["config_id"] == config_id


def test_download_nonexistent_artifact(client: TestClient):
    """Test downloading non-existent artifact returns 404."""
    from ulid import ULID

    response = client.get(f"/api/v1/artifacts/{ULID()}/$download")
    assert response.status_code == 404


def test_get_metadata_nonexistent_artifact(client: TestClient):
    """Test getting metadata for non-existent artifact returns 404."""
    from ulid import ULID

    response = client.get(f"/api/v1/artifacts/{ULID()}/$metadata")
    assert response.status_code == 404
