"""Artifact repository tests."""

from __future__ import annotations

from servicekit import SqliteDatabaseBuilder

from chapkit import Artifact, ArtifactRepository


async def test_artifact_repository_find_by_id_eager_loads_children() -> None:
    """find_by_id should eagerly load the artifact children collection."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ArtifactRepository(session)

        root = Artifact(data={"name": "root"}, level=0)
        await repo.save(root)
        await repo.commit()
        await repo.refresh_many([root])

        child_a = Artifact(data={"name": "child_a"}, parent_id=root.id, level=1)
        child_b = Artifact(data={"name": "child_b"}, parent_id=root.id, level=1)
        await repo.save_all([child_a, child_b])
        await repo.commit()
        await repo.refresh_many([child_a, child_b])

        fetched = await repo.find_by_id(root.id)
        assert fetched is not None
        children = await fetched.awaitable_attrs.children
        assert {child.data["name"] for child in children} == {"child_a", "child_b"}

    await db.dispose()


async def test_artifact_repository_find_subtree_returns_full_hierarchy() -> None:
    """find_subtree should return the start node and all descendants."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ArtifactRepository(session)

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

        subtree = list(await repo.find_subtree(root.id))
        ids = {artifact.id for artifact in subtree}
        assert ids == {root.id, child.id, grandchild.id}
        lookup = {artifact.id: artifact for artifact in subtree}
        assert lookup[child.id].parent_id == root.id
        assert lookup[grandchild.id].parent_id == child.id

    await db.dispose()
