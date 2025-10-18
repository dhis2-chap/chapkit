"""Tests for ArtifactRouter error handling."""

from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from servicekit.artifact import ArtifactIn, ArtifactManager, ArtifactOut, ArtifactRouter
from ulid import ULID


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
