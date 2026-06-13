"""Integration tests for the artifact service: hierarchies, config linking, read-only API, non-JSON payloads."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.artifact_app import build_artifact_app

ALPHA_CONFIG_ID = "01K72PWT05GEXK1S24AVKAZ9VE"
ALPHA_TRAIN_ID = "01K72PWT05GEXK1S24AVKAZ9VF"
ALPHA_RESULT_ID = "01K72PWT05GEXK1S24AVKAZ9VH"
MODEL_ARTIFACT_ID = "01K72PWT05GEXK1S24AVKAZ9VQ"
BETA_TRAIN_ID = "01K72PWT05GEXK1S24AVKAZ9VM"


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient for testing with lifespan context."""
    with TestClient(build_artifact_app()) as test_client:
        yield test_client


def test_landing_page(client: TestClient) -> None:
    """Test landing page returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns a status (flaky_service check randomizes the overall state)."""
    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")


def test_info_endpoint(client: TestClient) -> None:
    """Test service info endpoint returns service metadata."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "Chapkit Artifact Service"
    assert "hierarchy" in data
    assert data["hierarchy"]["name"] == "training_pipeline"
    assert data["configs"] == ["experiment_alpha", "experiment_beta"]
    assert data["non_json_payload"] == "MockLinearModel"


def test_list_artifacts(client: TestClient) -> None:
    """Test listing all seeded artifacts."""
    response = client.get("/api/v1/artifacts")
    assert response.status_code == 200
    data = response.json()

    # Two experiments seed eight artifacts in total
    assert isinstance(data, list)
    assert len(data) == 8

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
    response = client.get(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}")
    assert response.status_code == 200
    data = response.json()

    assert data["id"] == ALPHA_TRAIN_ID
    assert data["data"]["stage"] == "train"
    assert data["data"]["dataset"] == "alpha_train.parquet"
    assert data["level"] == 0


def test_get_artifact_by_id_not_found(client: TestClient) -> None:
    """Test retrieving non-existent artifact returns 404."""
    response = client.get("/api/v1/artifacts/01K72P5N5KCRM6MD3BRE4P0999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_get_artifact_tree(client: TestClient) -> None:
    """Test retrieving artifact tree structure with $tree operation."""
    response = client.get(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}/$tree")
    assert response.status_code == 200
    data = response.json()

    # Verify tree structure
    assert data["id"] == ALPHA_TRAIN_ID
    assert "data" in data
    assert "children" in data
    assert isinstance(data["children"], list)

    # Verify hierarchical structure
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
    response = client.get(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}/$expand")
    assert response.status_code == 200
    data = response.json()

    # Verify expanded structure
    assert data["id"] == ALPHA_TRAIN_ID
    assert "data" in data
    assert data["level"] == 0
    assert data["level_label"] == "train"
    assert data["hierarchy"] == "training_pipeline"

    # Verify children is None (not included in expand)
    assert data["children"] is None


def test_expand_artifact_with_parent(client: TestClient) -> None:
    """Test expanding a child artifact includes hierarchy metadata."""
    list_response = client.get("/api/v1/artifacts")
    artifacts = list_response.json()
    child_artifact = next(a for a in artifacts if a["level"] == 1)

    response = client.get(f"/api/v1/artifacts/{child_artifact['id']}/$expand")
    assert response.status_code == 200
    data = response.json()

    # Verify expanded structure
    assert data["id"] == child_artifact["id"]
    assert data["level"] == 1
    assert data["level_label"] == "predict"
    assert data["hierarchy"] == "training_pipeline"
    assert data["children"] is None


def test_expand_artifact_not_found(client: TestClient) -> None:
    """Test $expand operation on non-existent artifact returns 404."""
    response = client.get("/api/v1/artifacts/01K72P5N5KCRM6MD3BRE4P0999/$expand")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_artifact_with_non_json_payload(client: TestClient) -> None:
    """Test artifact with non-JSON content (MockLinearModel) is serialized with metadata."""
    response = client.get(f"/api/v1/artifacts/{MODEL_ARTIFACT_ID}")
    assert response.status_code == 200
    data = response.json()

    artifact_data = data["data"]
    assert artifact_data["content_type"] == "application/x-pickle"
    assert artifact_data["metadata"] == {"kind": "model", "format": "pickle"}

    # The MockLinearModel content is serialized with fallback metadata fields
    content = artifact_data["content"]
    assert content["_type"] == "MockLinearModel"
    assert content["_module"] == "tests.fixtures.artifact_app"
    assert "MockLinearModel" in content["_repr"]
    assert "coefficients" in content["_repr"]
    assert "intercept" in content["_repr"]
    assert "_serialization_error" in content


def test_artifact_metadata_operation(client: TestClient) -> None:
    """Test $metadata operation returns JSON metadata without the pickled content."""
    response = client.get(f"/api/v1/artifacts/{MODEL_ARTIFACT_ID}/$metadata")
    assert response.status_code == 200
    assert response.json() == {"kind": "model", "format": "pickle"}


def test_get_linked_artifacts_for_config(client: TestClient) -> None:
    """Test $artifacts operation on a config returns its linked root artifacts."""
    response = client.get(f"/api/v1/configs/{ALPHA_CONFIG_ID}/$artifacts")
    assert response.status_code == 200
    artifacts = response.json()

    assert len(artifacts) == 1
    assert artifacts[0]["id"] == ALPHA_TRAIN_ID


def test_get_config_for_artifact(client: TestClient) -> None:
    """Test $config operation on a root artifact returns the linked config."""
    response = client.get(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}/$config")
    assert response.status_code == 200
    config = response.json()

    assert config["id"] == ALPHA_CONFIG_ID
    assert config["name"] == "experiment_alpha"
    assert config["data"]["model"] == "xgboost"


def test_get_config_for_nested_artifact(client: TestClient) -> None:
    """Test $config operation on a nested artifact traverses to the root's linked config."""
    response = client.get(f"/api/v1/artifacts/{ALPHA_RESULT_ID}/$config")
    assert response.status_code == 200
    config = response.json()

    assert config["name"] == "experiment_alpha"


def test_create_artifact_not_allowed(client: TestClient) -> None:
    """Test that creating artifacts is disabled (read-only API)."""
    new_artifact = {"data": {"name": "test", "value": 123}}

    response = client.post("/api/v1/artifacts", json=new_artifact)
    assert response.status_code == 405  # Method Not Allowed


def test_update_artifact_not_allowed(client: TestClient) -> None:
    """Test that updating artifacts is disabled (read-only API)."""
    updated_artifact = {"id": ALPHA_TRAIN_ID, "data": {"updated": True}}

    response = client.put(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}", json=updated_artifact)
    assert response.status_code == 405  # Method Not Allowed


def test_delete_artifact_not_allowed(client: TestClient) -> None:
    """Test that deleting artifacts is disabled (read-only API)."""
    response = client.delete(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}")
    assert response.status_code == 405  # Method Not Allowed


def test_list_configs(client: TestClient) -> None:
    """Test listing configs returns the seeded experiments."""
    response = client.get("/api/v1/configs")
    assert response.status_code == 200
    data = response.json()
    names = [config["name"] for config in data]
    assert "experiment_alpha" in names
    assert "experiment_beta" in names


def test_experiment_alpha_tree_structure(client: TestClient) -> None:
    """Test experiment_alpha tree has correct hierarchical structure."""
    response = client.get(f"/api/v1/artifacts/{ALPHA_TRAIN_ID}/$tree")
    assert response.status_code == 200
    tree = response.json()

    # Verify root
    assert tree["id"] == ALPHA_TRAIN_ID
    assert tree["data"]["stage"] == "train"
    assert tree["level"] == 0
    assert len(tree["children"]) == 2  # Two prediction runs

    # Verify predict nodes
    for predict_node in tree["children"]:
        assert predict_node["level"] == 1
        assert predict_node["data"]["stage"] == "predict"


def test_experiment_beta_tree_structure(client: TestClient) -> None:
    """Test experiment_beta tree has correct hierarchical structure."""
    response = client.get(f"/api/v1/artifacts/{BETA_TRAIN_ID}/$tree")
    assert response.status_code == 200
    tree = response.json()

    # Verify root
    assert tree["id"] == BETA_TRAIN_ID
    assert tree["data"]["stage"] == "train"
    assert tree["level"] == 0
    assert len(tree["children"]) == 1  # One prediction run
