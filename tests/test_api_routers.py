"""Tests for concrete API routers built on CrudRouter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ulid import ULID

from chapkit import ArtifactIn, ArtifactOut, ArtifactTreeNode, BaseConfig, ConfigIn, ConfigOut
from chapkit.api import ArtifactRouter, ConfigRouter
from tests._stubs import ArtifactManagerStub, ConfigManagerStub, singleton_factory


class ExampleConfig(BaseConfig):
    """Sample config payload for tests."""

    enabled: bool


@pytest.fixture
def config_app() -> tuple[TestClient, ConfigOut[ExampleConfig]]:
    from servicekit.api.middleware import add_error_handlers

    now = datetime.now(tz=timezone.utc)
    record = ConfigOut[ExampleConfig](
        id=ULID(),
        name="feature-toggle",
        data=ExampleConfig(enabled=True),
        created_at=now,
        updated_at=now,
    )
    manager = ConfigManagerStub[ExampleConfig](items={"feature-toggle": record})
    router = ConfigRouter.create(
        prefix="/config",
        tags=["Config"],
        manager_factory=singleton_factory(manager),
        entity_in_type=ConfigIn[ExampleConfig],
        entity_out_type=ConfigOut[ExampleConfig],
        enable_artifact_operations=True,
    )
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(router)
    return TestClient(app), record


@pytest.fixture
def artifact_app() -> tuple[TestClient, ArtifactTreeNode]:
    from servicekit.api.middleware import add_error_handlers

    now = datetime.now(tz=timezone.utc)
    root_id = ULID()
    root = ArtifactTreeNode(
        id=root_id,
        data={"name": "root"},
        parent_id=None,
        level=0,
        created_at=now,
        updated_at=now,
        children=[],
    )
    manager = ArtifactManagerStub(trees={root_id: root})
    router = ArtifactRouter.create(
        prefix="/artifacts",
        tags=["Artifacts"],
        manager_factory=singleton_factory(manager),
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(router)
    return TestClient(app), root


def test_config_router_list_returns_records(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    client, record = config_app

    response = client.get("/config/")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    first = payload[0]
    assert first["id"] == str(record.id)
    assert first["name"] == "feature-toggle"
    assert first["data"] == {"enabled": True}
    assert "created_at" in first
    assert "updated_at" in first


def test_config_router_find_by_id_returns_record(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    client, record = config_app

    response = client.get(f"/config/{record.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(record.id)
    assert payload["name"] == "feature-toggle"
    assert payload["data"] == {"enabled": True}


def test_config_router_find_by_id_returns_404(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    client, _ = config_app

    response = client.get(f"/config/{ULID()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_artifact_router_tree_returns_node(artifact_app: tuple[TestClient, ArtifactTreeNode]) -> None:
    client, root = artifact_app

    response = client.get(f"/artifacts/{root.id}/$tree")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(root.id)
    assert payload["data"] == {"name": "root"}
    assert payload["children"] == []


def test_artifact_router_tree_returns_404_when_missing(artifact_app: tuple[TestClient, ArtifactTreeNode]) -> None:
    client, _ = artifact_app

    response = client.get(f"/artifacts/{ULID()}/$tree")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_artifact_router_tree_rejects_invalid_ulid(artifact_app: tuple[TestClient, ArtifactTreeNode]) -> None:
    client, _ = artifact_app

    response = client.get("/artifacts/not-a-ulid/$tree")

    assert response.status_code == 400
    assert "Invalid ULID" in response.json()["detail"]


def test_config_router_link_artifact_success(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    """Test successful artifact linking to a config."""
    client, record = config_app

    artifact_id = ULID()
    response = client.post(
        f"/config/{record.id}/$link-artifact",
        json={"artifact_id": str(artifact_id)},
    )

    assert response.status_code == 204


def test_config_router_link_artifact_with_error(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    """Test artifact linking with validation error."""
    client, record = config_app
    artifact_id = ULID()

    # Inject error into manager
    from tests._stubs import ConfigManagerStub

    manager = ConfigManagerStub[ExampleConfig](items={"feature-toggle": record})
    manager.set_link_error("Cannot link: artifact is not a root node")

    # Create new app with error-injected manager
    from servicekit.api.middleware import add_error_handlers

    router = ConfigRouter.create(
        prefix="/config",
        tags=["Config"],
        manager_factory=singleton_factory(manager),
        entity_in_type=ConfigIn[ExampleConfig],
        entity_out_type=ConfigOut[ExampleConfig],
        enable_artifact_operations=True,
    )
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        f"/config/{record.id}/$link-artifact",
        json={"artifact_id": str(artifact_id)},
    )

    assert response.status_code == 400
    assert "Cannot link" in response.json()["detail"]


def test_config_router_unlink_artifact_success(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    """Test successful artifact unlinking from a config."""
    client, _ = config_app

    artifact_id = ULID()
    response = client.post(
        "/config/anyid/$unlink-artifact",
        json={"artifact_id": str(artifact_id)},
    )

    assert response.status_code == 204


def test_config_router_unlink_artifact_with_error(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    """Test artifact unlinking with error."""
    client, record = config_app
    artifact_id = ULID()

    # Inject error into manager
    from tests._stubs import ConfigManagerStub

    manager = ConfigManagerStub[ExampleConfig](items={"feature-toggle": record})
    manager.set_link_error("Artifact not found")

    # Create new app with error-injected manager
    from servicekit.api.middleware import add_error_handlers

    router = ConfigRouter.create(
        prefix="/config",
        tags=["Config"],
        manager_factory=singleton_factory(manager),
        entity_in_type=ConfigIn[ExampleConfig],
        entity_out_type=ConfigOut[ExampleConfig],
        enable_artifact_operations=True,
    )
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        f"/config/{record.id}/$unlink-artifact",
        json={"artifact_id": str(artifact_id)},
    )

    assert response.status_code == 400
    assert "Artifact not found" in response.json()["detail"]


def test_config_router_get_linked_artifacts(config_app: tuple[TestClient, ConfigOut[ExampleConfig]]) -> None:
    """Test retrieving linked artifacts for a config."""
    client, record = config_app
    artifact_id = ULID()

    # Create manager with pre-linked artifacts
    now = datetime.now(tz=timezone.utc)
    artifact = ArtifactOut(
        id=artifact_id,
        data={"name": "test-artifact"},
        parent_id=None,
        level=0,
        created_at=now,
        updated_at=now,
    )

    from tests._stubs import ConfigManagerStub

    manager = ConfigManagerStub[ExampleConfig](
        items={"feature-toggle": record},
        linked_artifacts={record.id: [artifact]},
    )

    # Create new app with manager that has linked artifacts
    from servicekit.api.middleware import add_error_handlers

    router = ConfigRouter.create(
        prefix="/config",
        tags=["Config"],
        manager_factory=singleton_factory(manager),
        entity_in_type=ConfigIn[ExampleConfig],
        entity_out_type=ConfigOut[ExampleConfig],
        enable_artifact_operations=True,
    )
    app = FastAPI()
    add_error_handlers(app)
    app.include_router(router)
    client = TestClient(app)

    response = client.get(f"/config/{record.id}/$artifacts")

    assert response.status_code == 200
    artifacts = response.json()
    assert len(artifacts) == 1
    assert artifacts[0]["id"] == str(artifact_id)
    assert artifacts[0]["data"] == {"name": "test-artifact"}


@pytest.mark.skip(reason="Config access endpoint not yet implemented in ArtifactRouter")
def test_artifact_router_get_config_returns_config() -> None:
    """Test retrieving config for an artifact."""
    now = datetime.now(tz=timezone.utc)
    artifact_id = ULID()
    config_id = ULID()

    # Create managers with linked config
    artifact = ArtifactTreeNode(
        id=artifact_id,
        data={"name": "test-artifact"},
        parent_id=None,
        level=0,
        created_at=now,
        updated_at=now,
        children=[],
    )
    config = ConfigOut[ExampleConfig](
        id=config_id,
        name="test-config",
        data=ExampleConfig(enabled=True),
        created_at=now,
        updated_at=now,
    )

    from tests._stubs import ArtifactManagerStub, ConfigManagerStub

    artifact_manager = ArtifactManagerStub(trees={artifact_id: artifact})
    config_manager = ConfigManagerStub[ExampleConfig](items={"test-config": config})

    # Create app with both managers
    from servicekit.api.middleware import add_error_handlers

    artifact_router = ArtifactRouter.create(
        prefix="/artifacts",
        tags=["Artifacts"],
        manager_factory=singleton_factory(artifact_manager),
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
        enable_config_access=True,
    )

    app = FastAPI()
    add_error_handlers(app)
    app.include_router(artifact_router)

    # Override config manager dependency
    from chapkit.api.dependencies import get_config_manager

    app.dependency_overrides[get_config_manager] = singleton_factory(config_manager)

    client = TestClient(app)

    response = client.get(f"/artifacts/{artifact_id}/$config")

    assert response.status_code == 200
    config_data = response.json()
    assert config_data["id"] == str(config_id)
    assert config_data["name"] == "test-config"
    assert config_data["data"] == {"enabled": True}
