"""Tests for artifact_api example using TestClient.

This example demonstrates a read-only artifact API with hierarchical data.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from examples.artifact_api import app


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


def test_info_endpoint(client: TestClient) -> None:
    """Test service info endpoint returns service metadata."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "Chapkit Artifact Service"
    assert data["summary"] == "Artifact CRUD and tree operations example"
    assert "hierarchy" in data
    assert data["hierarchy"]["name"] == "ml_experiment"
    assert "pipelines" in data
    assert len(data["pipelines"]) == 2


def test_list_artifacts(client: TestClient) -> None:
    """Test listing all seeded artifacts."""
    response = client.get("/api/v1/artifacts")
    assert response.status_code == 200
    data = response.json()

    # Should be a list of artifacts from 2 pipelines
    assert isinstance(data, list)
    assert len(data) > 0

    # Verify structure
    for artifact in data:
        assert "id" in artifact
        assert "data" in artifact
        assert "parent_id" in artifact
        assert "level" in artifact
        assert "created_at" in artifact
        assert "updated_at" in artifact


def test_list_artifacts_with_pagination(client: TestClient) -> None:
    """Test listing artifacts with pagination."""
    response = client.get("/api/v1/artifacts", params={"page": 1, "size": 3})
    assert response.status_code == 200
    data = response.json()

    # Should return paginated response
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert "pages" in data

    assert len(data["items"]) <= 3
    assert data["page"] == 1
    assert data["size"] == 3


def test_get_artifact_by_id(client: TestClient) -> None:
    """Test retrieving artifact by ID."""
    # First get the list to obtain a valid ID
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    artifact_id = artifacts[0]["id"]

    response = client.get(f"/api/v1/artifacts/{artifact_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["id"] == artifact_id
    assert "data" in data
    assert "level" in data


def test_get_artifact_by_id_not_found(client: TestClient) -> None:
    """Test retrieving non-existent artifact returns 404."""
    response = client.get("/api/v1/artifacts/01K72P5N5KCRM6MD3BRE4P0999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_get_artifact_tree(client: TestClient) -> None:
    """Test retrieving artifact tree structure with $tree operation."""
    # Get a root artifact (level 0)
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    root_artifact = next((a for a in artifacts if a["level"] == 0), None)
    assert root_artifact is not None

    root_id = root_artifact["id"]
    response = client.get(f"/api/v1/artifacts/{root_id}/$tree")
    assert response.status_code == 200
    data = response.json()

    # Verify tree structure
    assert data["id"] == root_id
    assert "data" in data
    assert "children" in data
    assert isinstance(data["children"], list)

    # Verify hierarchical structure
    if len(data["children"]) > 0:
        child = data["children"][0]
        assert "id" in child
        assert "data" in child
        assert "children" in child


def test_get_artifact_tree_not_found(client: TestClient) -> None:
    """Test $tree operation on non-existent artifact returns 404."""
    response = client.get("/api/v1/artifacts/01K72P5N5KCRM6MD3BRE4P0999/$tree")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_expand_artifact(client: TestClient) -> None:
    """Test expanding artifact with $expand operation returns hierarchy metadata without children."""
    # Get a root artifact (level 0)
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    root_artifact = next((a for a in artifacts if a["level"] == 0), None)
    assert root_artifact is not None

    root_id = root_artifact["id"]
    response = client.get(f"/api/v1/artifacts/{root_id}/$expand")
    assert response.status_code == 200
    data = response.json()

    # Verify expanded structure
    assert data["id"] == root_id
    assert "data" in data
    assert "level" in data
    assert "level_label" in data
    assert "hierarchy" in data

    # Verify hierarchy metadata is present
    assert data["hierarchy"] == "ml_experiment"
    assert data["level_label"] == "experiment"

    # Verify children is None (not included in expand)
    assert data["children"] is None


def test_expand_artifact_with_parent(client: TestClient) -> None:
    """Test expanding artifact with parent includes hierarchy metadata."""
    # Get a child artifact (level 1)
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    child_artifact = next((a for a in artifacts if a["level"] == 1), None)
    assert child_artifact is not None

    child_id = child_artifact["id"]
    response = client.get(f"/api/v1/artifacts/{child_id}/$expand")
    assert response.status_code == 200
    data = response.json()

    # Verify expanded structure
    assert data["id"] == child_id
    assert data["level"] == 1
    assert data["level_label"] == "stage"
    assert data["hierarchy"] == "ml_experiment"
    assert data["children"] is None


def test_expand_artifact_not_found(client: TestClient) -> None:
    """Test $expand operation on non-existent artifact returns 404."""
    response = client.get("/api/v1/artifacts/01K72P5N5KCRM6MD3BRE4P0999/$expand")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_artifact_with_non_json_payload(client: TestClient) -> None:
    """Test artifact with non-JSON payload (MockLinearModel) is serialized with metadata."""
    # Find the artifact with MockLinearModel (ID: 01K72P5N5KCRM6MD3BRE4P07NJ)
    response = client.get("/api/v1/artifacts/01K72P5N5KCRM6MD3BRE4P07NJ")
    assert response.status_code == 200
    data = response.json()

    # Verify the MockLinearModel is serialized with metadata fields
    assert "data" in data
    artifact_data = data["data"]
    assert isinstance(artifact_data, dict)
    assert artifact_data["_type"] == "MockLinearModel"
    assert artifact_data["_module"] == "examples.artifact_api"
    assert "MockLinearModel" in artifact_data["_repr"]
    assert "coefficients" in artifact_data["_repr"]
    assert "intercept" in artifact_data["_repr"]
    assert "_serialization_error" in artifact_data


def test_create_artifact_not_allowed(client: TestClient) -> None:
    """Test that creating artifacts is disabled (read-only API)."""
    new_artifact = {"data": {"name": "test", "value": 123}}

    response = client.post("/api/v1/artifacts", json=new_artifact)
    assert response.status_code == 405  # Method Not Allowed


def test_update_artifact_not_allowed(client: TestClient) -> None:
    """Test that updating artifacts is disabled (read-only API)."""
    # Get an existing artifact ID
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    artifact_id = artifacts[0]["id"]

    updated_artifact = {"id": artifact_id, "data": {"updated": True}}

    response = client.put(f"/api/v1/artifacts/{artifact_id}", json=updated_artifact)
    assert response.status_code == 405  # Method Not Allowed


def test_delete_artifact_not_allowed(client: TestClient) -> None:
    """Test that deleting artifacts is disabled (read-only API)."""
    # Get an existing artifact ID
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    artifact_id = artifacts[0]["id"]

    response = client.delete(f"/api/v1/artifacts/{artifact_id}")
    assert response.status_code == 405  # Method Not Allowed


def test_list_configs(client: TestClient) -> None:
    """Test listing configs endpoint exists."""
    response = client.get("/api/v1/configs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_experiment_alpha_tree_structure(client: TestClient) -> None:
    """Test experiment_alpha tree has correct hierarchical structure."""
    # Root artifact for experiment_alpha
    root_id = "01K72P5N5KCRM6MD3BRE4P07NB"
    response = client.get(f"/api/v1/artifacts/{root_id}/$tree")
    assert response.status_code == 200
    tree = response.json()

    # Verify root
    assert tree["id"] == root_id
    assert tree["data"]["name"] == "experiment_alpha"
    assert tree["level"] == 0
    assert len(tree["children"]) == 2  # Two stages

    # Verify stages
    for stage in tree["children"]:
        assert stage["level"] == 1
        assert "stage" in stage["data"]
        assert "artifacts" in stage or "children" in stage


def test_experiment_beta_tree_structure(client: TestClient) -> None:
    """Test experiment_beta tree has correct hierarchical structure."""
    # Root artifact for experiment_beta
    root_id = "01K72P5N5KCRM6MD3BRE4P07NK"
    response = client.get(f"/api/v1/artifacts/{root_id}/$tree")
    assert response.status_code == 200
    tree = response.json()

    # Verify root
    assert tree["id"] == root_id
    assert tree["data"]["name"] == "experiment_beta"
    assert tree["level"] == 0
    assert len(tree["children"]) == 2  # Two stages
