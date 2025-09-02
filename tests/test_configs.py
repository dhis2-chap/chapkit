import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chapkit.runner import ChapRunner
from chapkit.service import ChapService
from chapkit.storage import JsonChapStorage
from chapkit.types import ChapConfig, ChapServiceInfo


class MockRunner(ChapRunner):
    pass


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        info = ChapServiceInfo(display_name="Test Service")
        runner = MockRunner(info, ChapConfig)
        storage = JsonChapStorage(Path(tmpdir) / "test.json", ChapConfig)
        service = ChapService(runner, storage)
        app = service.create_fastapi()
        yield TestClient(app)


def test_get_configs_empty(client):
    response = client.get("/api/v1/configs")
    assert response.status_code == 200
    assert response.json() == []


def test_add_and_get_config(client):
    # Add a config
    config_data = {"name": "test-config"}
    response = client.post("/api/v1/configs", json=config_data)
    assert response.status_code == 201
    response_json = response.json()
    assert response_json["name"] == "test-config"
    config_id = response_json["id"]

    # Get all configs
    response = client.get("/api/v1/configs")
    assert response.status_code == 200
    assert response.json() == [response_json]

    # Get the specific config
    response = client.get(f"/api/v1/configs/{config_id}")
    assert response.status_code == 200
    assert response.json() == response_json


def test_update_config(client):
    # Add a config
    config_data = {"name": "test-config"}
    response = client.post("/api/v1/configs", json=config_data)
    assert response.status_code == 201
    config_id = response.json()["id"]

    # Update the config
    updated_config_data = {"id": config_id, "name": "new-name"}
    response = client.put(f"/api/v1/configs/{config_id}", json=updated_config_data)
    assert response.status_code == 200
    assert response.json() == updated_config_data

    # Get the config to verify the update
    response = client.get(f"/api/v1/configs/{config_id}")
    assert response.status_code == 200
    assert response.json() == updated_config_data


def test_update_config_id_mismatch(client):
    # Add a config
    config_data = {"name": "test-config"}
    response = client.post("/api/v1/configs", json=config_data)
    assert response.status_code == 201
    config_id = response.json()["id"]

    # Try to update with a mismatched ID
    mismatched_id = "b6d3a2e0-5a6e-4b5f-8e9b-1e2e3e4e5e6e"
    updated_config_data = {"id": mismatched_id, "name": "new-name"}
    response = client.put(f"/api/v1/configs/{config_id}", json=updated_config_data)
    assert response.status_code == 400


def test_delete_config(client):
    # Add a config
    config_data = {"name": "test-config"}
    response = client.post("/api/v1/configs", json=config_data)
    config_id = response.json()["id"]

    # Delete the config
    response = client.delete(f"/api/v1/configs/{config_id}")
    assert response.status_code == 204

    # Verify it's gone
    response = client.get(f"/api/v1/configs/{config_id}")
    assert response.status_code == 404


def test_get_nonexistent_config(client):
    response = client.get("/api/v1/configs/b6d3a2e0-5a6e-4b5f-8e9b-1e2e3e4e5e6e")
    assert response.status_code == 404


def test_delete_nonexistent_config(client):
    response = client.delete("/api/v1/configs/b6d3a2e0-5a6e-4b5f-8e9b-1e2e3e4e5e6e")
    assert response.status_code == 404
