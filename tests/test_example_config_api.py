"""Tests for config_api example using TestClient.

Tests use FastAPI's TestClient instead of running a separate server.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from examples.config_api import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient for testing with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_landing_page(client: TestClient) -> None:
    """Test landing page returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "checks" in data
    assert "database" in data["checks"]


def test_info_endpoint(client: TestClient) -> None:
    """Test service info endpoint returns service metadata."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "Chapkit Config Service"
    assert data["summary"] == "Environment configuration CRUD example"
    assert data["author"] == "Morten Hansen"
    assert "seeded_configs" in data
    assert len(data["seeded_configs"]) == 3


def test_list_configs(client: TestClient) -> None:
    """Test listing all seeded configs."""
    response = client.get("/api/v1/configs")
    assert response.status_code == 200
    data = response.json()

    # Should be a list of 3 seeded configs
    assert isinstance(data, list)
    assert len(data) == 3

    # Check config names
    names = {config["name"] for config in data}
    assert names == {"production", "staging", "local"}

    # Verify structure
    for config in data:
        assert "id" in config
        assert "name" in config
        assert "data" in config
        assert "created_at" in config
        assert "updated_at" in config

        # Check data structure
        assert "debug" in config["data"]
        assert "api_host" in config["data"]
        assert "api_port" in config["data"]
        assert "max_connections" in config["data"]


def test_list_configs_with_pagination(client: TestClient) -> None:
    """Test listing configs with pagination."""
    response = client.get("/api/v1/configs", params={"page": 1, "size": 2})
    assert response.status_code == 200
    data = response.json()

    # Should return paginated response
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert "pages" in data

    assert len(data["items"]) == 2
    assert data["total"] == 3
    assert data["page"] == 1
    assert data["size"] == 2
    assert data["pages"] == 2


def test_get_config_by_id(client: TestClient) -> None:
    """Test retrieving config by ID."""
    # First get the list to obtain a valid ID
    list_response = client.get("/api/v1/configs")
    configs = list_response.json()
    config_id = configs[0]["id"]

    response = client.get(f"/api/v1/configs/{config_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["id"] == config_id
    assert "name" in data
    assert "data" in data


def test_get_config_by_invalid_ulid(client: TestClient) -> None:
    """Test retrieving config with invalid ULID format returns 400."""
    response = client.get("/api/v1/configs/not-a-valid-ulid")
    assert response.status_code == 400
    data = response.json()
    assert "invalid ulid" in data["detail"].lower()


def test_get_config_by_id_not_found(client: TestClient) -> None:
    """Test retrieving non-existent config by ID returns 404."""
    # Use a valid ULID format but non-existent ID
    response = client.get("/api/v1/configs/01K72P5N5KCRM6MD3BRE4P0999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_create_config(client: TestClient) -> None:
    """Test creating a new config."""
    new_config = {
        "name": "test-environment",
        "data": {"debug": True, "api_host": "localhost", "api_port": 9000, "max_connections": 50},
    }

    response = client.post("/api/v1/configs", json=new_config)
    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert data["name"] == "test-environment"
    assert data["data"]["debug"] is True
    assert data["data"]["api_port"] == 9000

    # Verify it was created by fetching it
    config_id = data["id"]
    get_response = client.get(f"/api/v1/configs/{config_id}")
    assert get_response.status_code == 200


def test_create_config_duplicate_name(client: TestClient) -> None:
    """Test creating config with duplicate name succeeds (no unique constraint)."""
    # First create a config
    first_config = {
        "name": "duplicate-test",
        "data": {"debug": False, "api_host": "0.0.0.0", "api_port": 8080, "max_connections": 1000},
    }
    response1 = client.post("/api/v1/configs", json=first_config)
    assert response1.status_code == 201
    first_id = response1.json()["id"]

    # Create another with the same name - should succeed
    duplicate_config = {
        "name": "duplicate-test",
        "data": {"debug": True, "api_host": "127.0.0.1", "api_port": 8000, "max_connections": 500},
    }

    response2 = client.post("/api/v1/configs", json=duplicate_config)
    assert response2.status_code == 201
    second_id = response2.json()["id"]

    # Verify they have different IDs but same name
    assert first_id != second_id
    assert response1.json()["name"] == response2.json()["name"] == "duplicate-test"


def test_update_config(client: TestClient) -> None:
    """Test updating an existing config."""
    # Create a config first
    new_config = {
        "name": "update-test",
        "data": {"debug": False, "api_host": "127.0.0.1", "api_port": 8080, "max_connections": 100},
    }
    create_response = client.post("/api/v1/configs", json=new_config)
    created = create_response.json()
    config_id = created["id"]

    # Update it
    updated_config = {
        "id": config_id,
        "name": "update-test",
        "data": {
            "debug": True,  # Changed
            "api_host": "127.0.0.1",
            "api_port": 9999,  # Changed
            "max_connections": 200,  # Changed
        },
    }

    response = client.put(f"/api/v1/configs/{config_id}", json=updated_config)
    assert response.status_code == 200
    data = response.json()

    assert data["id"] == config_id
    assert data["data"]["debug"] is True
    assert data["data"]["api_port"] == 9999
    assert data["data"]["max_connections"] == 200


def test_delete_config(client: TestClient) -> None:
    """Test deleting a config."""
    # Create a config first
    new_config = {
        "name": "delete-test",
        "data": {"debug": False, "api_host": "127.0.0.1", "api_port": 8080, "max_connections": 100},
    }
    create_response = client.post("/api/v1/configs", json=new_config)
    created = create_response.json()
    config_id = created["id"]

    # Delete it
    response = client.delete(f"/api/v1/configs/{config_id}")
    assert response.status_code == 204

    # Verify it's gone
    get_response = client.get(f"/api/v1/configs/{config_id}")
    assert get_response.status_code == 404


def test_delete_config_not_found(client: TestClient) -> None:
    """Test deleting non-existent config returns 404."""
    response = client.delete("/api/v1/configs/01K72P5N5KCRM6MD3BRE4P0999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
