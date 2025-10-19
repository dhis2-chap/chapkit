"""Integration tests for artifact API with non-JSON-serializable data."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from servicekit import SqliteDatabaseBuilder
from servicekit.api.dependencies import get_session, set_database
from servicekit.api.middleware import add_error_handlers
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit.artifact import ArtifactIn, ArtifactManager, ArtifactOut, ArtifactRepository, ArtifactRouter


def get_artifact_manager(session: AsyncSession = Depends(get_session)) -> ArtifactManager:
    """Create an artifact manager instance for dependency injection."""
    return ArtifactManager(ArtifactRepository(session))


class NonSerializableObject:
    """Custom object that cannot be JSON-serialized."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"NonSerializableObject({self.value!r})"


class TestArtifactAPIWithNonSerializableData:
    """Test that the API handles non-serializable artifact data gracefully."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_artifact_with_custom_object(self) -> None:
        """API should handle artifacts with non-serializable data without crashing."""
        # Setup
        app = FastAPI()
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()
        set_database(db)

        try:
            # Create router
            artifact_router = ArtifactRouter.create(
                prefix="/api/v1/artifacts",
                tags=["Artifacts"],
                manager_factory=get_artifact_manager,
                entity_in_type=ArtifactIn,
                entity_out_type=ArtifactOut,
            )
            app.include_router(artifact_router)
            add_error_handlers(app)

            # Create an artifact directly via the database (bypassing API validation)
            async with db.session() as session:
                repo = ArtifactRepository(session)
                manager = ArtifactManager(repo)

                # Save artifact with non-serializable data
                custom_obj = NonSerializableObject("test_value")
                artifact_in = ArtifactIn(data=custom_obj)
                saved_artifact = await manager.save(artifact_in)
                artifact_id = saved_artifact.id

            # Now try to retrieve via API - should not crash
            with TestClient(app) as client:
                response = client.get(f"/api/v1/artifacts/{artifact_id}")

                # Should return 200 OK (not crash with 500)
                assert response.status_code == 200

                # Should return metadata instead of the actual object
                data = response.json()
                assert data["id"] == str(artifact_id)
                assert isinstance(data["data"], dict)
                assert data["data"]["_type"] == "NonSerializableObject"
                assert data["data"]["_module"] == __name__
                assert "NonSerializableObject('test_value')" in data["data"]["_repr"]
                assert "_serialization_error" in data["data"]

        finally:
            await db.dispose()

    @pytest.mark.asyncio
    async def test_list_artifacts_with_mixed_data(self) -> None:
        """API should handle listing artifacts with both serializable and non-serializable data."""
        # Setup
        app = FastAPI()
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()
        set_database(db)

        try:
            # Create router
            artifact_router = ArtifactRouter.create(
                prefix="/api/v1/artifacts",
                tags=["Artifacts"],
                manager_factory=get_artifact_manager,
                entity_in_type=ArtifactIn,
                entity_out_type=ArtifactOut,
            )
            app.include_router(artifact_router)
            add_error_handlers(app)

            # Create artifacts with different data types
            async with db.session() as session:
                repo = ArtifactRepository(session)
                manager = ArtifactManager(repo)

                # Artifact 1: JSON-serializable data
                await manager.save(ArtifactIn(data={"type": "json", "value": 123}))

                # Artifact 2: Non-serializable data
                custom_obj = NonSerializableObject("test")
                await manager.save(ArtifactIn(data=custom_obj))

                # Artifact 3: Another JSON-serializable
                await manager.save(ArtifactIn(data=["list", "of", "values"]))

            # List all artifacts - should not crash
            with TestClient(app) as client:
                response = client.get("/api/v1/artifacts")

                # Should return 200 OK
                assert response.status_code == 200

                artifacts = response.json()
                assert len(artifacts) == 3

                # First artifact - JSON data unchanged
                assert artifacts[0]["data"] == {"type": "json", "value": 123}

                # Second artifact - metadata returned
                assert artifacts[1]["data"]["_type"] == "NonSerializableObject"

                # Third artifact - JSON data unchanged
                assert artifacts[2]["data"] == ["list", "of", "values"]

        finally:
            await db.dispose()

    @pytest.mark.asyncio
    async def test_tree_operation_with_non_serializable_data(self) -> None:
        """Tree operation should handle non-serializable data in nested artifacts."""
        # Setup
        app = FastAPI()
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()
        set_database(db)

        try:
            # Create router
            artifact_router = ArtifactRouter.create(
                prefix="/api/v1/artifacts",
                tags=["Artifacts"],
                manager_factory=get_artifact_manager,
                entity_in_type=ArtifactIn,
                entity_out_type=ArtifactOut,
            )
            app.include_router(artifact_router)
            add_error_handlers(app)

            # Create a tree with non-serializable data
            root_id = None
            async with db.session() as session:
                repo = ArtifactRepository(session)
                manager = ArtifactManager(repo)

                # Root with non-serializable data
                custom_obj = NonSerializableObject("root")
                root = await manager.save(ArtifactIn(data=custom_obj))
                root_id = root.id

                # Child with JSON data
                await manager.save(ArtifactIn(data={"child": "data"}, parent_id=root_id))

            # Get tree - should not crash
            with TestClient(app) as client:
                response = client.get(f"/api/v1/artifacts/{root_id}/$tree")

                # Should return 200 OK
                assert response.status_code == 200

                tree = response.json()

                # Root should have metadata
                assert tree["data"]["_type"] == "NonSerializableObject"

                # Child should have JSON data
                assert tree["children"][0]["data"] == {"child": "data"}

        finally:
            await db.dispose()
