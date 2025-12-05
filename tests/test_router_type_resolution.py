"""Regression tests for router type annotation resolution (PEP 563 compatibility)."""

from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chapkit.artifact import ArtifactManager
from chapkit.artifact.router import ArtifactRouter
from chapkit.artifact.schemas import ArtifactIn, ArtifactOut
from chapkit.config import BaseConfig, ConfigIn, ConfigManager, ConfigOut
from chapkit.config.router import ConfigRouter


def test_config_router_type_resolution() -> None:
    """Verify ConfigRouter resolves types at runtime.

    This test fails if `from __future__ import annotations` is added
    to config/router.py, as FastAPI cannot resolve string annotations
    for response_model and generic type parameters.
    """
    mock_manager = Mock(spec=ConfigManager)
    mock_manager.find_all = AsyncMock(return_value=[])

    app = FastAPI()
    router = ConfigRouter.create(
        prefix="/api/v1/configs",
        tags=["Configs"],
        manager_factory=lambda: mock_manager,
        entity_in_type=ConfigIn[BaseConfig],
        entity_out_type=ConfigOut[BaseConfig],
    )
    app.include_router(router)

    client = TestClient(app)

    # Trigger type resolution via request
    response = client.get("/api/v1/configs")
    assert response.status_code == 200

    # Verify OpenAPI schema generation works (requires type resolution)
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200


def test_artifact_router_type_resolution() -> None:
    """Verify ArtifactRouter resolves types at runtime.

    This test fails if `from __future__ import annotations` is added
    to artifact/router.py.
    """
    mock_manager = Mock(spec=ArtifactManager)
    mock_manager.find_all = AsyncMock(return_value=[])

    app = FastAPI()
    router = ArtifactRouter.create(
        prefix="/api/v1/artifacts",
        tags=["Artifacts"],
        manager_factory=lambda: mock_manager,
        entity_in_type=ArtifactIn,
        entity_out_type=ArtifactOut,
    )
    app.include_router(router)

    client = TestClient(app)

    response = client.get("/api/v1/artifacts")
    assert response.status_code == 200

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
