"""Tests for ArtifactRouter error handling."""

import pickle
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ulid import ULID

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactOut, ArtifactRouter
from chapkit.data import DataFrame


def test_expand_artifact_not_found_returns_404() -> None:
    """Test that expand_artifact returns 404 when artifact not found."""
    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.expand_artifact = AsyncMock(return_value=None)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    artifact_id = str(ULID())
    response = client.get(f"/api/v1/artifacts/{artifact_id}/$expand")

    assert response.status_code == 404
    assert f"Artifact with id {artifact_id} not found" in response.text


def test_build_tree_not_found_returns_404() -> None:
    """Test that build_tree returns 404 when artifact not found."""
    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.build_tree = AsyncMock(return_value=None)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    artifact_id = str(ULID())
    response = client.get(f"/api/v1/artifacts/{artifact_id}/$tree")

    assert response.status_code == 404
    assert f"Artifact with id {artifact_id} not found" in response.text


def test_download_artifact_not_found_returns_404() -> None:
    """Test that download_artifact returns 404 when artifact not found."""
    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=None)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    artifact_id = str(ULID())
    response = client.get(f"/api/v1/artifacts/{artifact_id}/$download")

    assert response.status_code == 404
    assert f"Artifact with id {artifact_id} not found" in response.text


def test_download_artifact_with_non_dict_data() -> None:
    """Test that download returns 400 when artifact.data is not a dict."""
    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = "not a dict"

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 400
    assert "Artifact has no downloadable content" in response.text


def test_download_artifact_with_no_content_field() -> None:
    """Test that download returns 404 when artifact has no content field."""
    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {"type": "ml_training", "metadata": {}}

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 404
    assert "Artifact has no content" in response.text


def test_download_artifact_with_bytes_content() -> None:
    """Test downloading artifact with bytes content."""
    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "content": b"binary data here",
        "content_type": "application/octet-stream",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "attachment" in response.headers["content-disposition"]
    assert response.content == b"binary data here"


def test_download_artifact_with_dataframe_content() -> None:
    """Test downloading artifact with DataFrame content stored as dict."""
    df_data = DataFrame(
        columns=["col1", "col2"],
        data=[[1, 2], [3, 4]],
    )

    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "content": df_data.model_dump(),
        "content_type": "application/vnd.chapkit.dataframe+json",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.chapkit.dataframe+json"
    assert "attachment" in response.headers["content-disposition"]

    # Content should be JSON array format from DataFrame.to_json()
    import json

    content = json.loads(response.content)
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"col1": 1, "col2": 2}
    assert content[1] == {"col1": 3, "col2": 4}


def test_download_artifact_with_dataframe_as_csv() -> None:
    """Test downloading DataFrame artifact with CSV content type."""
    df_data = DataFrame(
        columns=["name", "age"],
        data=[["Alice", 30], ["Bob", 25]],
    )

    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "content": df_data,
        "content_type": "text/csv",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]

    # Verify CSV content
    csv_content = response.content.decode()
    assert "name,age" in csv_content
    assert "Alice,30" in csv_content
    assert "Bob,25" in csv_content


def test_download_artifact_with_pickled_bytes() -> None:
    """Test downloading artifact with pickled data stored as bytes."""
    # Use a simple pickleable object
    data = {"model_params": [1, 2, 3], "score": 0.95}
    pickled_bytes = pickle.dumps(data)

    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "content": pickled_bytes,
        "content_type": "application/x-pickle",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-pickle"
    assert "attachment" in response.headers["content-disposition"]

    # Verify data can be unpickled
    unpickled = pickle.loads(response.content)
    assert unpickled == data


def test_download_artifact_with_unsupported_type() -> None:
    """Test that downloading artifact with unsupported content type returns 400."""

    class UnsupportedType:
        """Type that cannot be serialized."""

        pass

    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "content": UnsupportedType(),
        "content_type": "application/x-custom",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 400
    assert "Cannot serialize content" in response.text


def test_download_artifact_with_dict_content() -> None:
    """Test downloading artifact with dict content."""
    content_dict = {"key1": "value1", "key2": 123, "nested": {"a": 1}}

    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "content": content_dict,
        "content_type": "application/json",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "attachment" in response.headers["content-disposition"]

    # Verify JSON can be parsed
    import json

    parsed = json.loads(response.content)
    assert parsed == content_dict


def test_get_metadata_not_found_returns_404() -> None:
    """Test that get_metadata returns 404 when artifact not found."""
    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=None)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    artifact_id = str(ULID())
    response = client.get(f"/api/v1/artifacts/{artifact_id}/$metadata")

    assert response.status_code == 404
    assert f"Artifact with id {artifact_id} not found" in response.text


def test_get_metadata_returns_metadata_only() -> None:
    """Test that get_metadata returns only metadata, excluding content."""
    metadata = {
        "status": "success",
        "config_id": "01CONFIG123",
        "started_at": "2025-10-18T10:00:00",
        "completed_at": "2025-10-18T10:05:00",
        "duration_seconds": 300.0,
    }

    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "type": "ml_training",
        "metadata": metadata,
        "content": b"large binary content that should not be included",
        "content_type": "application/x-pickle",
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$metadata")

    assert response.status_code == 200
    response_data = response.json()

    # Should return exactly the metadata
    assert response_data == metadata
    # Should not include content
    assert "content" not in response_data
    assert "content_type" not in response_data


def test_get_metadata_when_no_metadata_field() -> None:
    """Test that get_metadata returns empty dict when no metadata field."""
    mock_artifact = Mock(spec=ArtifactOut)
    mock_artifact.id = ULID()
    mock_artifact.data = {
        "type": "generic",
        "content": {"some": "data"},
    }

    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_by_id = AsyncMock(return_value=mock_artifact)

    def manager_factory() -> ArtifactManager:
        return mock_manager

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=manager_factory,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get(f"/api/v1/artifacts/{mock_artifact.id}/$metadata")

    assert response.status_code == 200
    assert response.json() == {}
