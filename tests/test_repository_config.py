"""Config repository tests."""

from __future__ import annotations

import pytest
from servicekit import SqliteDatabaseBuilder
from servicekit.artifact import Artifact, ArtifactRepository
from ulid import ULID

from chapkit import Config, ConfigRepository

from .conftest import DemoConfig


async def test_config_repository_find_by_name_round_trip() -> None:
    """ConfigRepository.find_by_name should return matching rows and None otherwise."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ConfigRepository(session)

        assert await repo.find_by_name("missing") is None

        created = Config(name="feature", data=DemoConfig(x=1, y=2, z=3, tags=["feature"]))
        await repo.save(created)
        await repo.commit()
        await repo.refresh_many([created])

        found = await repo.find_by_name("feature")
        assert found is not None
        assert found.id == created.id
        assert found.name == "feature"
        assert found.data == {"x": 1, "y": 2, "z": 3, "tags": ["feature"]}

    await db.dispose()


async def test_link_artifact_not_found_raises_error() -> None:
    """ConfigRepository.link_artifact should raise ValueError when artifact not found."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ConfigRepository(session)

        config = Config(name="test-config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await repo.save(config)
        await repo.commit()

        non_existent_artifact_id = ULID()
        with pytest.raises(ValueError, match=f"Artifact {non_existent_artifact_id} not found"):
            await repo.link_artifact(config.id, non_existent_artifact_id)

    await db.dispose()


async def test_link_artifact_non_root_raises_error() -> None:
    """ConfigRepository.link_artifact should raise ValueError when artifact is not a root."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        config = Config(name="test-config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)

        # Create parent and child artifacts
        parent = Artifact(data={"type": "parent"}, level=0)
        await artifact_repo.save(parent)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([parent])

        child = Artifact(data={"type": "child"}, level=1, parent_id=parent.id)
        await artifact_repo.save(child)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([child])

        with pytest.raises(ValueError, match=f"Artifact {child.id} is not a root artifact"):
            await config_repo.link_artifact(config.id, child.id)

    await db.dispose()


async def test_delete_by_id_cascades_to_artifacts() -> None:
    """ConfigRepository.delete_by_id should cascade delete linked artifacts and subtrees."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        # Create config
        config = Config(name="test-config", data=DemoConfig(x=1, y=2, z=3, tags=["test"]))
        await config_repo.save(config)
        await config_repo.commit()
        await config_repo.refresh_many([config])

        # Create root artifact with children
        root = Artifact(data={"type": "root"}, level=0)
        await artifact_repo.save(root)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([root])

        child1 = Artifact(data={"type": "child1"}, level=1, parent_id=root.id)
        child2 = Artifact(data={"type": "child2"}, level=1, parent_id=root.id)
        await artifact_repo.save(child1)
        await artifact_repo.save(child2)
        await artifact_repo.commit()
        await artifact_repo.refresh_many([child1, child2])

        # Link root artifact to config
        await config_repo.link_artifact(config.id, root.id)
        await config_repo.commit()

        # Delete config should cascade delete all artifacts
        await config_repo.delete_by_id(config.id)
        await config_repo.commit()

        # Verify config is deleted
        assert await config_repo.find_by_id(config.id) is None

        # Verify artifacts are deleted
        assert await artifact_repo.find_by_id(root.id) is None
        assert await artifact_repo.find_by_id(child1.id) is None
        assert await artifact_repo.find_by_id(child2.id) is None

    await db.dispose()
