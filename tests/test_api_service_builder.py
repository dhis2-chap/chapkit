"""Tests for ServiceBuilder functionality."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from servicekit import Database, SqliteDatabaseBuilder
from servicekit.api.routers.health import HealthState

from chapkit import ArtifactHierarchy, BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo


class ExampleConfig(BaseConfig):
    """Example configuration schema for tests."""

    prediction_periods: int = 3
    enabled: bool
    value: int


@pytest.fixture
async def test_database() -> AsyncGenerator[Database, None]:
    """Provide a test database instance."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    yield db
    await db.dispose()


@pytest.fixture
def service_info() -> ServiceInfo:
    """Provide basic service info for tests."""
    return ServiceInfo(
        id="test-service",
        display_name="Test Service",
        version="1.0.0",
        summary="Test service for unit tests",
    )


def test_service_builder_creates_basic_app(service_info: ServiceInfo) -> None:
    """Test that ServiceBuilder creates a minimal FastAPI app."""
    app = ServiceBuilder.create(info=service_info)

    assert isinstance(app, FastAPI)
    assert app.title == "Test Service"
    assert app.version == "1.0.0"


def test_service_builder_with_health_endpoint(service_info: ServiceInfo) -> None:
    """Test that with_health() adds health endpoint."""
    app = ServiceBuilder(info=service_info).with_health().build()

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_service_builder_with_custom_health_checks(service_info: ServiceInfo) -> None:
    """Test custom health checks are registered."""
    check_called = False

    async def custom_check() -> tuple[HealthState, str | None]:
        nonlocal check_called
        check_called = True
        return (HealthState.HEALTHY, None)

    app = (
        ServiceBuilder(info=service_info)
        .with_health(checks={"custom": custom_check}, include_database_check=False)
        .build()
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "custom" in data["checks"]
    assert check_called


def test_service_builder_with_config(service_info: ServiceInfo) -> None:
    """Test that with_config() adds config endpoints."""
    app = ServiceBuilder(info=service_info).with_config(ExampleConfig).build()

    with TestClient(app) as client:
        response = client.get("/api/v1/configs/")

        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_service_builder_with_artifacts(service_info: ServiceInfo) -> None:
    """Test that with_artifacts() adds artifact endpoints."""
    hierarchy = ArtifactHierarchy(
        name="test_hierarchy",
        level_labels={0: "root", 1: "child"},
    )

    app = ServiceBuilder(info=service_info).with_artifacts(hierarchy=hierarchy).build()

    with TestClient(app) as client:
        response = client.get("/api/v1/artifacts/")

        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_service_builder_config_artifact_linking(service_info: ServiceInfo) -> None:
    """Test config-artifact linking integration."""
    hierarchy = ArtifactHierarchy(
        name="test_hierarchy",
        level_labels={0: "root"},
    )

    app = (
        ServiceBuilder(info=service_info)
        .with_config(ExampleConfig)
        .with_artifacts(hierarchy=hierarchy, enable_config_linking=True)
        .build()
    )

    assert app is not None


def test_service_builder_validation_fails_without_config(service_info: ServiceInfo) -> None:
    """Test that enabling config linking without config schema raises error."""
    hierarchy = ArtifactHierarchy(
        name="test_hierarchy",
        level_labels={0: "root"},
    )

    with pytest.raises(ValueError, match="config schema"):
        ServiceBuilder(info=service_info).with_artifacts(
            hierarchy=hierarchy,
            enable_config_linking=True,
        ).build()


def test_service_builder_invalid_health_check_name(service_info: ServiceInfo) -> None:
    """Test that invalid health check names are rejected."""

    async def check() -> tuple[HealthState, str | None]:
        return (HealthState.HEALTHY, None)

    with pytest.raises(ValueError, match="invalid characters"):
        ServiceBuilder(info=service_info).with_health(checks={"invalid name!": check}).build()


@pytest.mark.asyncio
async def test_service_builder_with_database_instance(
    service_info: ServiceInfo,
    test_database: Database,
) -> None:
    """Test injecting a pre-configured database instance."""
    app = ServiceBuilder(info=service_info).with_database(test_database).with_config(ExampleConfig).build()

    # Test that the app uses the injected database
    with TestClient(app) as client:
        response = client.get("/api/v1/configs/")

        assert response.status_code == 200


def test_service_builder_with_database_invalid_type(service_info: ServiceInfo) -> None:
    """Test that with_database() rejects invalid types."""
    with pytest.raises(TypeError, match="Expected str, Database, or None"):
        ServiceBuilder(info=service_info).with_database(123).build()  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_service_builder_custom_router_integration(service_info: ServiceInfo) -> None:
    """Test including custom routers."""
    from fastapi import APIRouter

    custom_router = APIRouter(prefix="/custom", tags=["custom"])

    @custom_router.get("/test")
    async def custom_endpoint() -> dict[str, str]:
        return {"message": "custom"}

    app = ServiceBuilder(info=service_info).include_router(custom_router).build()

    client = TestClient(app)
    response = client.get("/custom/test")

    assert response.status_code == 200
    assert response.json() == {"message": "custom"}


def test_service_builder_info_endpoint(service_info: ServiceInfo) -> None:
    """Test that /api/v1/info endpoint is created."""
    app = ServiceBuilder.create(info=service_info)

    client = TestClient(app)
    response = client.get("/api/v1/info")

    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "Test Service"
    assert data["version"] == "1.0.0"


def test_service_builder_permissions(service_info: ServiceInfo) -> None:
    """Test that permissions restrict operations."""
    app = ServiceBuilder(info=service_info).with_config(ExampleConfig, allow_create=False, allow_delete=False).build()

    with TestClient(app) as client:
        # GET should work (read is allowed)
        response = client.get("/api/v1/configs/")
        assert response.status_code == 200

        # POST should fail (create is disabled)
        response = client.post("/api/v1/configs/", json={"name": "test", "data": {"enabled": True, "value": 42}})
        assert response.status_code == 405  # Method not allowed


def test_service_builder_fluent_api(service_info: ServiceInfo) -> None:
    """Test fluent API chaining."""
    hierarchy = ArtifactHierarchy(name="test", level_labels={0: "root"})

    app = (
        ServiceBuilder(info=service_info)
        .with_health()
        .with_config(ExampleConfig)
        .with_artifacts(hierarchy=hierarchy)
        .build()
    )

    assert isinstance(app, FastAPI)
    assert app.title == "Test Service"


def test_service_builder_startup_hook(service_info: ServiceInfo) -> None:
    """Test that startup hooks are executed."""
    hook_called = False

    async def startup_hook(app: FastAPI) -> None:
        nonlocal hook_called
        hook_called = True

    app = ServiceBuilder(info=service_info).on_startup(startup_hook).build()

    # Startup hooks run during lifespan
    with TestClient(app):
        pass

    assert hook_called


def test_service_builder_shutdown_hook(service_info: ServiceInfo) -> None:
    """Test that shutdown hooks are executed."""
    hook_called = False

    async def shutdown_hook(app: FastAPI) -> None:
        nonlocal hook_called
        hook_called = True

    app = ServiceBuilder(info=service_info).on_shutdown(shutdown_hook).build()

    # Shutdown hooks run during lifespan cleanup
    with TestClient(app):
        pass

    assert hook_called


def test_service_builder_preserves_summary_as_description(service_info: ServiceInfo) -> None:
    """Test that summary is preserved as description when description is missing."""
    info = ServiceInfo(id="test", display_name="Test", summary="Test summary")
    builder = ServiceBuilder(info=info)

    assert builder.info.description == "Test summary"


def test_service_builder_landing_page(service_info: ServiceInfo) -> None:
    """Test that with_landing_page() adds root endpoint."""
    app = ServiceBuilder(info=service_info).with_landing_page().build()

    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        # Check that the page contains JavaScript to fetch service info
        assert "fetch('/api/v1/info')" in response.text
        assert "/docs" in response.text  # Navigation link to API docs

        # Verify that /api/v1/info returns the correct data
        info_response = client.get("/api/v1/info")
        assert info_response.status_code == 200
        data = info_response.json()
        assert data["display_name"] == "Test Service"
        assert data["version"] == "1.0.0"
        assert data["summary"] == "Test service for unit tests"


def test_service_builder_without_landing_page(service_info: ServiceInfo) -> None:
    """Test that root endpoint is not added by default."""
    app = ServiceBuilder(info=service_info).build()

    with TestClient(app) as client:
        response = client.get("/")

        # Should return 404 when landing page is not enabled
        assert response.status_code == 404


def test_service_builder_with_system(service_info: ServiceInfo) -> None:
    """Test that with_system() adds system endpoint."""
    app = ServiceBuilder(info=service_info).with_system().build()

    with TestClient(app) as client:
        response = client.get("/api/v1/system/")

        assert response.status_code == 200
        data = response.json()
        assert "current_time" in data
        assert "timezone" in data
        assert "python_version" in data
        assert "platform" in data
        assert "hostname" in data


def test_service_builder_landing_page_with_custom_fields() -> None:
    """Test that landing page displays custom ServiceInfo fields."""
    from pydantic import EmailStr

    class CustomServiceInfo(ServiceInfo):
        """Extended service info with custom fields."""

        author: str
        contact_email: EmailStr
        custom_field: dict[str, object]

    info = CustomServiceInfo(
        id="custom-service",
        display_name="Custom Service",
        version="2.0.0",
        summary="Test with custom fields",
        author="Jane Doe",
        contact_email="jane@example.com",
        custom_field={"key": "value", "count": 42},
    )

    app = ServiceBuilder(info=info).with_landing_page().build()

    with TestClient(app) as client:
        # Check landing page HTML loads properly
        response = client.get("/")
        assert response.status_code == 200
        assert "fetch('/api/v1/info')" in response.text

        # Check that /api/v1/info includes custom fields
        info_response = client.get("/api/v1/info")
        assert info_response.status_code == 200
        data = info_response.json()
        assert data["display_name"] == "Custom Service"
        assert data["version"] == "2.0.0"
        assert data["summary"] == "Test with custom fields"
        assert data["author"] == "Jane Doe"
        assert data["contact_email"] == "jane@example.com"
        assert data["custom_field"] == {"key": "value", "count": 42}


def test_service_builder_with_monitoring(service_info: ServiceInfo) -> None:
    """Test that with_monitoring() adds metrics endpoint."""
    app = ServiceBuilder(info=service_info).with_monitoring().build()

    with TestClient(app) as client:
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        # Check for Prometheus format markers
        assert b"# HELP" in response.content or b"# TYPE" in response.content or len(response.content) > 0


def test_service_builder_with_monitoring_custom_prefix(service_info: ServiceInfo) -> None:
    """Test that monitoring can use custom prefix."""
    app = ServiceBuilder(info=service_info).with_monitoring(prefix="/api/v1/metrics").build()

    with TestClient(app) as client:
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200


def test_service_builder_with_all_features(service_info: ServiceInfo) -> None:
    """Integration test with all features enabled."""
    hierarchy = ArtifactHierarchy(name="test", level_labels={0: "root"})

    async def health_check() -> tuple[HealthState, str | None]:
        return (HealthState.HEALTHY, None)

    async def startup(app: FastAPI) -> None:
        pass

    async def shutdown(app: FastAPI) -> None:
        pass

    app = (
        ServiceBuilder(info=service_info)
        .with_landing_page()
        .with_health(checks={"test": health_check})
        .with_config(ExampleConfig)
        .with_artifacts(hierarchy=hierarchy, enable_config_linking=True)
        .with_monitoring()
        .on_startup(startup)
        .on_shutdown(shutdown)
        .build()
    )

    with TestClient(app) as client:
        # Test all endpoints work
        # Note: Root app mount catches trailing slash redirects, so use exact paths
        assert client.get("/").status_code == 200
        assert client.get("/api/v1/info").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/configs").status_code == 200
        assert client.get("/api/v1/artifacts").status_code == 200
        assert client.get("/metrics").status_code == 200
