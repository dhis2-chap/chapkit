"""Tests for Config-Artifact linking functionality."""

from __future__ import annotations

import pytest
from servicekit import SqliteDatabaseBuilder
from servicekit.artifact import Artifact, ArtifactRepository

from chapkit import Config, ConfigManager, ConfigRepository

from .conftest import DemoConfig


async def test_link_artifact_creates_link() -> None:
    """ConfigRepository.link_artifact should create a link between config and root artifact."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        # Create a config
        config = Config(name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        # Create a root artifact
        artifact = Artifact(data={"name": "root"}, level=0)
        await artifact_repo.save(artifact)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([artifact])

        # Link them
        await config_repo.link_artifact(config.id, artifact.id)
        await config_repo.commit()

        # Verify link exists
        found_config = await config_repo.find_by_root_artifact_id(artifact.id)
        assert found_config is not None
        assert found_config.id == config.id

    await db.dispose()


async def test_link_artifact_rejects_non_root_artifacts() -> None:
    """ConfigRepository.link_artifact should raise ValueError if artifact has parent."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        # Create a config
        config = Config(name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        # Create root and child artifacts
        root = Artifact(data={"name": "root"}, level=0)
        await artifact_repo.save(root)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root])

        child = Artifact(data={"name": "child"}, parent_id=root.id, level=1)
        await artifact_repo.save(child)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([child])

        # Attempt to link child should fail
        with pytest.raises(ValueError, match="not a root artifact"):
            await config_repo.link_artifact(config.id, child.id)

    await db.dispose()


async def test_unlink_artifact_removes_link() -> None:
    """ConfigRepository.unlink_artifact should remove the link."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        # Create and link
        config = Config(name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        artifact = Artifact(data={"name": "root"}, level=0)
        await artifact_repo.save(artifact)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([artifact])

        await config_repo.link_artifact(config.id, artifact.id)
        await config_repo.commit()

        # Verify link exists
        assert await config_repo.find_by_root_artifact_id(artifact.id) is not None

        # Unlink
        await config_repo.unlink_artifact(artifact.id)
        await config_repo.commit()

        # Verify link removed
        assert await config_repo.find_by_root_artifact_id(artifact.id) is None

    await db.dispose()


async def test_find_artifacts_for_config_returns_linked_artifacts() -> None:
    """ConfigRepository.find_artifacts_for_config should return all linked root artifacts."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        # Create a config
        config = Config(name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        # Create multiple root artifacts
        artifact1 = Artifact(data={"name": "root1"}, level=0)
        artifact2 = Artifact(data={"name": "root2"}, level=0)
        await artifact_repo.save_all([artifact1, artifact2])
        await artifact_repo.commit()
        await artifact_repo.refresh_many([artifact1, artifact2])

        # Link both to the same config
        await config_repo.link_artifact(config.id, artifact1.id)
        await config_repo.link_artifact(config.id, artifact2.id)
        await config_repo.commit()

        # Find all linked artifacts
        linked = await config_repo.find_artifacts_for_config(config.id)
        assert len(linked) == 2
        assert {a.id for a in linked} == {artifact1.id, artifact2.id}

    await db.dispose()


async def test_get_root_artifact_walks_up_tree() -> None:
    """ArtifactRepository.get_root_artifact should walk up the tree to find root."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ArtifactRepository(session)

        # Create tree: root -> child -> grandchild
        root = Artifact(data={"name": "root"}, level=0)
        await repo.save(root)
        await repo.commit()
        await repo.refresh_many([root])

        child = Artifact(data={"name": "child"}, parent_id=root.id, level=1)
        await repo.save(child)
        await repo.commit()
        await repo.refresh_many([child])

        grandchild = Artifact(data={"name": "grandchild"}, parent_id=child.id, level=2)
        await repo.save(grandchild)
        await repo.commit()
        await repo.refresh_many([grandchild])

        # Get root from grandchild
        found_root = await repo.get_root_artifact(grandchild.id)
        assert found_root is not None
        assert found_root.id == root.id

        # Get root from child
        found_root = await repo.get_root_artifact(child.id)
        assert found_root is not None
        assert found_root.id == root.id

        # Get root from root
        found_root = await repo.get_root_artifact(root.id)
        assert found_root is not None
        assert found_root.id == root.id

    await db.dispose()


async def test_config_manager_get_config_for_artifact() -> None:
    """ConfigManager.get_config_for_artifact should walk up tree and return config."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)
        manager = ConfigManager[DemoConfig](config_repo, DemoConfig)

        # Create config and artifacts
        config = Config(name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        root = Artifact(data={"name": "root"}, level=0)
        await artifact_repo.save(root)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root])

        child = Artifact(data={"name": "child"}, parent_id=root.id, level=1)
        await artifact_repo.save(child)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([child])

        # Link config to root
        await config_repo.link_artifact(config.id, root.id)
        await config_repo.commit()

        # Get config from child (should walk up to root)
        found_config = await manager.get_config_for_artifact(child.id, artifact_repo)
        assert found_config is not None
        assert found_config.id == config.id
        assert found_config.data is not None
        assert found_config.data.x == 1

    await db.dispose()


async def test_cascade_delete_config_deletes_artifacts() -> None:
    """Deleting a config should cascade delete linked artifacts."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        # Create config and artifacts
        config = Config(name="test_config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        root = Artifact(data={"name": "root"}, level=0)
        await artifact_repo.save(root)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root])

        child = Artifact(data={"name": "child"}, parent_id=root.id, level=1)
        await artifact_repo.save(child)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([child])

        # Link config to root
        await config_repo.link_artifact(config.id, root.id)
        await config_repo.commit()

        # Delete config
        await config_repo.delete_by_id(config.id)
        await config_repo.commit()

        # Verify artifacts are deleted (cascade from config -> config_artifact -> artifact)
        assert await artifact_repo.find_by_id(root.id) is None
        assert await artifact_repo.find_by_id(child.id) is None

    await db.dispose()
