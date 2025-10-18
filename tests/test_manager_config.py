"""Service-layer tests for config management."""

from __future__ import annotations

from conftest import DemoConfig
from ulid import ULID

from chapkit import (
    Artifact,
    ArtifactRepository,
    Config,
    ConfigIn,
    ConfigManager,
    ConfigOut,
    ConfigRepository,
    SqliteDatabaseBuilder,
)


async def test_config_manager_deserializes_dict_payloads() -> None:
    """ConfigManager should convert raw dict payloads to the configured schema."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ConfigRepository(session)
        manager = ConfigManager[DemoConfig](repo, DemoConfig)

        assert await manager.find_by_name("missing") is None

        raw = Config(name="raw", data={"x": 1, "y": 2, "z": 3, "tags": ["dict"]})
        await repo.save(raw)
        await repo.commit()
        await repo.refresh_many([raw])

        output = manager._to_output_schema(raw)
        assert isinstance(output, ConfigOut)
        assert isinstance(output.data, DemoConfig)
        assert output.data.x == 1
        assert output.data.tags == ["dict"]

        found = await manager.find_by_name("raw")
        assert found is not None
        assert isinstance(found.data, DemoConfig)
        assert found.data.model_dump() == output.data.model_dump()

    await db.dispose()


async def test_config_manager_save() -> None:
    """Test saving a config through the manager."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ConfigRepository(session)
        manager = ConfigManager[DemoConfig](repo, DemoConfig)

        config_in = ConfigIn[DemoConfig](name="test_save", data=DemoConfig(x=10, y=20, z=30, tags=["test"]))
        saved = await manager.save(config_in)

        assert saved.name == "test_save"
        assert saved.data.x == 10

    await db.dispose()


async def test_config_manager_link_artifact() -> None:
    """Test linking a config to a root artifact."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)
        manager = ConfigManager[DemoConfig](config_repo, DemoConfig)

        # Create config
        config_in = ConfigIn[DemoConfig](name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=[]))
        saved_config = await manager.save(config_in)

        # Create root artifact
        root_artifact = Artifact(parent_id=None, data={"type": "root"})
        await artifact_repo.save(root_artifact)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root_artifact])

        # Link them
        await manager.link_artifact(saved_config.id, root_artifact.id)

        # Verify link
        linked_artifacts = await manager.get_linked_artifacts(saved_config.id)
        assert len(linked_artifacts) == 1
        assert linked_artifacts[0].id == root_artifact.id

    await db.dispose()


async def test_config_manager_unlink_artifact() -> None:
    """Test unlinking an artifact from a config."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)
        manager = ConfigManager[DemoConfig](config_repo, DemoConfig)

        # Create config and artifact
        config_in = ConfigIn[DemoConfig](name="test_unlink", data=DemoConfig(x=1, y=2, z=3, tags=[]))
        saved_config = await manager.save(config_in)

        root_artifact = Artifact(parent_id=None, data={"type": "root"})
        await artifact_repo.save(root_artifact)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root_artifact])

        # Link and then unlink
        await manager.link_artifact(saved_config.id, root_artifact.id)
        await manager.unlink_artifact(root_artifact.id)

        # Verify unlinked
        linked_artifacts = await manager.get_linked_artifacts(saved_config.id)
        assert len(linked_artifacts) == 0

    await db.dispose()


async def test_config_manager_get_config_for_artifact() -> None:
    """Test getting config for an artifact by walking up the tree."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)
        manager = ConfigManager[DemoConfig](config_repo, DemoConfig)

        # Create config
        config_in = ConfigIn[DemoConfig](name="tree_config", data=DemoConfig(x=100, y=200, z=300, tags=["tree"]))
        saved_config = await manager.save(config_in)

        # Create artifact tree: root -> child
        root_artifact = Artifact(parent_id=None, data={"level": "root"})
        await artifact_repo.save(root_artifact)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root_artifact])

        child_artifact = Artifact(parent_id=root_artifact.id, data={"level": "child"})
        await artifact_repo.save(child_artifact)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([child_artifact])

        # Link config to root
        await manager.link_artifact(saved_config.id, root_artifact.id)

        # Get config via child artifact (should walk up to root)
        config_from_child = await manager.get_config_for_artifact(child_artifact.id, artifact_repo)
        assert config_from_child is not None
        assert config_from_child.id == saved_config.id
        assert config_from_child.name == "tree_config"

        # Test with non-existent artifact
        config_from_missing = await manager.get_config_for_artifact(ULID(), artifact_repo)
        assert config_from_missing is None

    await db.dispose()
