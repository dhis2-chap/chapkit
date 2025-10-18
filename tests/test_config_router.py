"""Tests for ConfigRouter artifact linking operations."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from servicekit.artifact import ArtifactOut
from ulid import ULID

from chapkit.config import BaseConfig, ConfigIn, ConfigManager, ConfigOut, ConfigRouter


def test_link_artifact_success() -> None:
    """Test successful artifact linking to config."""
    mock_manager = Mock(spec=ConfigManager)
    mock_manager.link_artifact = AsyncMock()

    def manager_factory() -> ConfigManager:
        return mock_manager

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=manager_factory,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
        enable_artifact_operations=True,
    )
    app.include_router(router)

    client = TestClient(app)

    config_id = str(ULID())
    artifact_id = str(ULID())
    response = client.post(
        f"/api/v1/configs/{config_id}/$link-artifact",
        json={"artifact_id": artifact_id},
    )

    assert response.status_code == 204
    mock_manager.link_artifact.assert_called_once()


def test_link_artifact_invalid_artifact_returns_400() -> None:
    """Test that linking invalid artifact returns 400."""
    mock_manager = Mock(spec=ConfigManager)
    mock_manager.link_artifact = AsyncMock(side_effect=ValueError("Artifact not found"))

    def manager_factory() -> ConfigManager:
        return mock_manager

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=manager_factory,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
        enable_artifact_operations=True,
    )
    app.include_router(router)

    client = TestClient(app)

    config_id = str(ULID())
    artifact_id = str(ULID())
    response = client.post(
        f"/api/v1/configs/{config_id}/$link-artifact",
        json={"artifact_id": artifact_id},
    )

    assert response.status_code == 400
    assert "Artifact not found" in response.text


def test_unlink_artifact_success() -> None:
    """Test successful artifact unlinking from config."""
    mock_manager = Mock(spec=ConfigManager)
    mock_manager.unlink_artifact = AsyncMock()

    def manager_factory() -> ConfigManager:
        return mock_manager

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=manager_factory,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
        enable_artifact_operations=True,
    )
    app.include_router(router)

    client = TestClient(app)

    config_id = str(ULID())
    artifact_id = str(ULID())
    response = client.post(
        f"/api/v1/configs/{config_id}/$unlink-artifact",
        json={"artifact_id": artifact_id},
    )

    assert response.status_code == 204
    mock_manager.unlink_artifact.assert_called_once()


def test_unlink_artifact_error_returns_400() -> None:
    """Test that unlinking artifact with error returns 400."""
    mock_manager = Mock(spec=ConfigManager)
    mock_manager.unlink_artifact = AsyncMock(side_effect=Exception("Unlink failed"))

    def manager_factory() -> ConfigManager:
        return mock_manager

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=manager_factory,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
        enable_artifact_operations=True,
    )
    app.include_router(router)

    client = TestClient(app)

    config_id = str(ULID())
    artifact_id = str(ULID())
    response = client.post(
        f"/api/v1/configs/{config_id}/$unlink-artifact",
        json={"artifact_id": artifact_id},
    )

    assert response.status_code == 400
    assert "Unlink failed" in response.text


def test_get_linked_artifacts_success() -> None:
    """Test retrieving linked artifacts for a config."""
    mock_artifacts = [
        ArtifactOut(
            id=ULID(),
            data={},
            parent_id=None,
            level=0,
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            updated_at=datetime(2024, 1, 1, 0, 0, 0),
        ),
        ArtifactOut(
            id=ULID(),
            data={},
            parent_id=None,
            level=0,
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            updated_at=datetime(2024, 1, 1, 0, 0, 0),
        ),
    ]

    mock_manager = Mock(spec=ConfigManager)
    mock_manager.get_linked_artifacts = AsyncMock(return_value=mock_artifacts)

    def manager_factory() -> ConfigManager:
        return mock_manager

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=manager_factory,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
        enable_artifact_operations=True,
    )
    app.include_router(router)

    client = TestClient(app)

    config_id = str(ULID())
    response = client.get(f"/api/v1/configs/{config_id}/$artifacts")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    mock_manager.get_linked_artifacts.assert_called_once()


def test_artifact_operations_disabled_by_default() -> None:
    """Test that artifact operations are not registered when disabled."""
    mock_manager = Mock(spec=ConfigManager)

    def manager_factory() -> ConfigManager:
        return mock_manager

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=manager_factory,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
        enable_artifact_operations=False,  # Disabled
    )
    app.include_router(router)

    client = TestClient(app)

    config_id = str(ULID())

    # All artifact operations should return 404
    response = client.post(
        f"/api/v1/configs/{config_id}/$link-artifact",
        json={"artifact_id": str(ULID())},
    )
    assert response.status_code == 404

    response = client.post(
        f"/api/v1/configs/{config_id}/$unlink-artifact",
        json={"artifact_id": str(ULID())},
    )
    assert response.status_code == 404

    response = client.get(f"/api/v1/configs/{config_id}/$artifacts")
    assert response.status_code == 404
