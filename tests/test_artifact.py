import pandas as pd
import pytest
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import (
    Artifact,
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactOut,
    ArtifactRepository,
)
from chapkit.data import DataFrame


class TestArtifactRepository:
    """Tests for the ArtifactRepository class."""

    async def test_find_by_id_with_children(self) -> None:
        """Test that find_by_id returns artifact with parent_id set."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)

            # Create parent
            parent = Artifact(data={"type": "parent", "name": "root"}, level=0)
            await repo.save(parent)
            await repo.commit()
            await repo.refresh_many([parent])

            # Create child with parent_id
            child = Artifact(data={"type": "child", "name": "child1"}, parent_id=parent.id, level=1)
            await repo.save(child)
            await repo.commit()
            await repo.refresh_many([child])

            # Find child by ID
            found = await repo.find_by_id(child.id)

            assert found is not None
            assert found.id == child.id
            assert found.parent_id == parent.id
            assert found.data == {"type": "child", "name": "child1"}
            assert found.level == 1

        await db.dispose()

    async def test_find_subtree_single_node(self) -> None:
        """Test finding subtree with a single node (no children)."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)

            # Create single artifact
            artifact = Artifact(data={"type": "leaf", "name": "single"}, level=0)
            await repo.save(artifact)
            await repo.commit()
            await repo.refresh_many([artifact])

            # Find subtree
            subtree = await repo.find_subtree(artifact.id)

            assert len(list(subtree)) == 1
            assert list(subtree)[0].id == artifact.id

        await db.dispose()

    async def test_find_subtree_with_children(self) -> None:
        """Test finding subtree with parent and children."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)

            # Create parent
            parent = Artifact(data={"type": "parent"}, level=0)
            await repo.save(parent)
            await repo.commit()
            await repo.refresh_many([parent])

            # Create children
            child1 = Artifact(data={"type": "child1"}, parent_id=parent.id, level=1)
            child2 = Artifact(data={"type": "child2"}, parent_id=parent.id, level=1)
            await repo.save_all([child1, child2])
            await repo.commit()

            # Find subtree
            subtree = list(await repo.find_subtree(parent.id))

            assert len(subtree) == 3
            ids = {artifact.id for artifact in subtree}
            assert parent.id in ids
            assert child1.id in ids
            assert child2.id in ids

        await db.dispose()

    async def test_find_subtree_with_grandchildren(self) -> None:
        """Test finding subtree with multiple levels (grandchildren)."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)

            # Create parent
            parent = Artifact(data={"level": "root"}, level=0)
            await repo.save(parent)
            await repo.commit()
            await repo.refresh_many([parent])

            # Create children
            child1 = Artifact(data={"level": "child1"}, parent_id=parent.id, level=1)
            child2 = Artifact(data={"level": "child2"}, parent_id=parent.id, level=1)
            await repo.save_all([child1, child2])
            await repo.commit()
            await repo.refresh_many([child1, child2])

            # Create grandchildren
            grandchild1 = Artifact(data={"level": "grandchild1"}, parent_id=child1.id, level=2)
            grandchild2 = Artifact(data={"level": "grandchild2"}, parent_id=child1.id, level=2)
            grandchild3 = Artifact(data={"level": "grandchild3"}, parent_id=child2.id, level=2)
            await repo.save_all([grandchild1, grandchild2, grandchild3])
            await repo.commit()

            # Find subtree from root
            subtree = list(await repo.find_subtree(parent.id))

            assert len(subtree) == 6  # parent + 2 children + 3 grandchildren
            ids = {artifact.id for artifact in subtree}
            assert parent.id in ids
            assert child1.id in ids
            assert child2.id in ids
            assert grandchild1.id in ids
            assert grandchild2.id in ids
            assert grandchild3.id in ids

        await db.dispose()

    async def test_find_subtree_from_middle_node(self) -> None:
        """Test finding subtree starting from a middle node."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)

            # Create parent
            parent = Artifact(data={"level": "root"}, level=0)
            await repo.save(parent)
            await repo.commit()
            await repo.refresh_many([parent])

            # Create children
            child1 = Artifact(data={"level": "child1"}, parent_id=parent.id, level=1)
            child2 = Artifact(data={"level": "child2"}, parent_id=parent.id, level=1)
            await repo.save_all([child1, child2])
            await repo.commit()
            await repo.refresh_many([child1, child2])

            # Create grandchildren under child1
            grandchild1 = Artifact(data={"level": "grandchild1"}, parent_id=child1.id, level=2)
            grandchild2 = Artifact(data={"level": "grandchild2"}, parent_id=child1.id, level=2)
            await repo.save_all([grandchild1, grandchild2])
            await repo.commit()

            # Find subtree from child1 (not root)
            subtree = list(await repo.find_subtree(child1.id))

            # Should only include child1 and its descendants, not parent or child2
            assert len(subtree) == 3  # child1 + 2 grandchildren
            ids = {artifact.id for artifact in subtree}
            assert child1.id in ids
            assert grandchild1.id in ids
            assert grandchild2.id in ids
            assert parent.id not in ids
            assert child2.id not in ids

        await db.dispose()


class TestArtifactManager:
    """Tests for the ArtifactManager class."""

    async def test_save_artifact(self) -> None:
        """Test saving an artifact through manager."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Save artifact
            artifact_in = ArtifactIn(data={"type": "test", "value": 123})
            result = await manager.save(artifact_in)

            assert isinstance(result, ArtifactOut)
            assert result.id is not None
            assert result.data == {"type": "test", "value": 123}
            assert result.level == 0

        await db.dispose()


class HookAwareArtifactManager(ArtifactManager):
    def __init__(self, repo: ArtifactRepository) -> None:
        super().__init__(repo)
        self.calls: list[str] = []

    def _record(self, event: str, entity: Artifact) -> None:
        self.calls.append(f"{event}:{entity.id}")

    async def pre_save(self, entity: Artifact, data: ArtifactIn) -> None:
        self._record("pre_save", entity)
        await super().pre_save(entity, data)

    async def post_save(self, entity: Artifact) -> None:
        self._record("post_save", entity)
        await super().post_save(entity)

    async def pre_update(self, entity: Artifact, data: ArtifactIn, old_values: dict[str, object]) -> None:
        self._record("pre_update", entity)
        await super().pre_update(entity, data, old_values)

    async def post_update(self, entity: Artifact, changes: dict[str, tuple[object, object]]) -> None:
        self._record("post_update", entity)
        await super().post_update(entity, changes)

    async def pre_delete(self, entity: Artifact) -> None:
        self._record("pre_delete", entity)
        await super().pre_delete(entity)

    async def post_delete(self, entity: Artifact) -> None:
        self._record("post_delete", entity)
        await super().post_delete(entity)


class TestBaseManagerLifecycle:
    async def test_hooks_invoke_during_crud(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = HookAwareArtifactManager(repo)

            saved = await manager.save(ArtifactIn(data={"name": "one"}))
            assert [call.split(":")[0] for call in manager.calls] == ["pre_save", "post_save"]
            manager.calls.clear()

            await manager.save(ArtifactIn(id=saved.id, data={"name": "one-updated"}))
            assert [call.split(":")[0] for call in manager.calls] == ["pre_update", "post_update"]
            manager.calls.clear()

            await manager.delete_by_id(saved.id)
            assert [call.split(":")[0] for call in manager.calls] == ["pre_delete", "post_delete"]

        await db.dispose()

    async def test_delete_by_id_handles_missing(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = HookAwareArtifactManager(repo)

            await manager.delete_by_id(ULID())

        await db.dispose()

    async def test_save_all_empty_input(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = HookAwareArtifactManager(repo)

            results = await manager.save_all([])
            assert results == []

        await db.dispose()

    async def test_delete_all_no_entities(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = HookAwareArtifactManager(repo)

            await manager.delete_all()
            await manager.delete_all_by_id([])

        await db.dispose()

    async def test_compute_level_handles_none_parent(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            level = await manager._compute_level(None)
            assert level == 0

        await db.dispose()

    async def test_save_artifact_with_parent(self) -> None:
        """Test saving an artifact with a parent relationship."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Save parent
            parent_in = ArtifactIn(data={"type": "parent"})
            parent_out = await manager.save(parent_in)

            # Save child
            assert parent_out.id is not None
            assert parent_out.level == 0
            child_in = ArtifactIn(data={"type": "child"}, parent_id=parent_out.id)
            child_out = await manager.save(child_in)

            assert child_out.id is not None
            assert child_out.parent_id == parent_out.id
            assert child_out.level == 1

        await db.dispose()

    async def test_save_all_assigns_levels(self) -> None:
        """Test save_all computes level based on parent relationships."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            parent = await manager.save(ArtifactIn(data={"type": "parent"}))
            assert parent.id is not None

            children = [ArtifactIn(data={"type": "child", "index": i}, parent_id=parent.id) for i in range(2)]

            results = await manager.save_all(children)

            assert len(results) == 2
            assert all(child.level == 1 for child in results)

        await db.dispose()

    async def test_save_all_updates_existing_entities(self) -> None:
        """save_all should respect hooks when updating existing artifacts."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            roots = await manager.save_all(
                [
                    ArtifactIn(data={"name": "root_a"}),
                    ArtifactIn(data={"name": "root_b"}),
                ]
            )
            root_a, root_b = roots
            assert root_a.id is not None and root_b.id is not None

            children = await manager.save_all(
                [
                    ArtifactIn(data={"name": "child0"}, parent_id=root_a.id),
                    ArtifactIn(data={"name": "child1"}, parent_id=root_a.id),
                ]
            )

            moved = await manager.save_all(
                [
                    ArtifactIn(id=children[0].id, data={"name": "child0"}, parent_id=root_b.id),
                    ArtifactIn(id=children[1].id, data={"name": "child1"}, parent_id=root_b.id),
                ]
            )

            assert all(child.level == 1 for child in moved)

        await db.dispose()

    async def test_update_parent_recomputes_levels(self) -> None:
        """Moving an artifact under a new parent updates levels for it and descendants."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            root_a = await manager.save(ArtifactIn(data={"name": "root_a"}))
            root_b = await manager.save(ArtifactIn(data={"name": "root_b"}))
            assert root_a.id is not None and root_b.id is not None

            child = await manager.save(ArtifactIn(data={"name": "child"}, parent_id=root_a.id))
            grandchild = await manager.save(ArtifactIn(data={"name": "grandchild"}, parent_id=child.id))

            assert child.level == 1
            assert grandchild.level == 2

            # Move child (and implicitly grandchild) under root_b
            updated_child = await manager.save(ArtifactIn(id=child.id, data={"name": "child"}, parent_id=root_b.id))

            assert updated_child.level == 1

            subtree = await manager.find_subtree(updated_child.id)
            levels = {artifact.id: artifact.level for artifact in subtree}
            assert levels[updated_child.id] == 1
            assert levels[grandchild.id] == 2

        await db.dispose()

    async def test_update_without_parent_change_preserves_levels(self) -> None:
        """Updating artifact data without changing parent leaves levels untouched."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            parent = await manager.save(ArtifactIn(data={"name": "root"}))
            assert parent.id is not None

            child = await manager.save(ArtifactIn(data={"name": "child"}, parent_id=parent.id))
            assert child.level == 1

            renamed = await manager.save(ArtifactIn(id=child.id, data={"name": "child-renamed"}, parent_id=parent.id))

            assert renamed.level == 1

        await db.dispose()

    async def test_find_subtree_single_artifact(self) -> None:
        """Test finding subtree through manager with single artifact."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Create artifact
            artifact_in = ArtifactIn(data={"name": "single"})
            saved = await manager.save(artifact_in)

            # Find subtree
            assert saved.id is not None
            subtree = await manager.find_subtree(saved.id)

            assert len(subtree) == 1
            assert isinstance(subtree[0], ArtifactOut)
            assert subtree[0].id == saved.id
            assert subtree[0].data == {"name": "single"}
            assert subtree[0].level == 0

        await db.dispose()

    async def test_find_subtree_with_hierarchy(self) -> None:
        """Test finding subtree through manager with hierarchical data."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            hierarchy = ArtifactHierarchy(name="ml", level_labels={0: "root", 1: "child", 2: "grandchild"})
            manager = ArtifactManager(repo, hierarchy=hierarchy)

            # Create parent
            parent_in = ArtifactIn(data={"level": "root"})
            parent_out = await manager.save(parent_in)

            # Create children
            assert parent_out.id is not None
            child1_in = ArtifactIn(data={"level": "child1"}, parent_id=parent_out.id)
            child2_in = ArtifactIn(data={"level": "child2"}, parent_id=parent_out.id)
            child1_out = await manager.save(child1_in)
            child2_out = await manager.save(child2_in)

            # Create grandchild
            assert child1_out.id is not None
            grandchild_in = ArtifactIn(data={"level": "grandchild"}, parent_id=child1_out.id)
            grandchild_out = await manager.save(grandchild_in)

            # Find subtree from root
            subtree = await manager.find_subtree(parent_out.id)

            assert len(subtree) == 4
            assert all(isinstance(artifact, ArtifactOut) for artifact in subtree)

            ids = {artifact.id for artifact in subtree}
            assert parent_out.id in ids
            assert child1_out.id in ids
            assert child2_out.id in ids
            assert grandchild_out.id in ids
            for artifact in subtree:
                if artifact.id == parent_out.id:
                    assert artifact.level == 0
                    assert artifact.level_label == "root"
                    assert artifact.hierarchy == "ml"
                elif artifact.id in {child1_out.id, child2_out.id}:
                    assert artifact.level == 1
                    assert artifact.level_label == "child"
                else:
                    assert artifact.level == 2
                    assert artifact.level_label == "grandchild"

        await db.dispose()

    async def test_build_tree_returns_nested_structure(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            hierarchy = ArtifactHierarchy(name="ml", level_labels={0: "root", 1: "child", 2: "grandchild"})
            manager = ArtifactManager(repo, hierarchy=hierarchy)

            root = await manager.save(ArtifactIn(data={"name": "root"}))
            assert root.id is not None
            child = await manager.save(ArtifactIn(data={"name": "child"}, parent_id=root.id))
            grandchild = await manager.save(ArtifactIn(data={"name": "grandchild"}, parent_id=child.id))

            tree = await manager.build_tree(root.id)
            assert tree is not None
            assert tree.id == root.id
            assert tree.level_label == "root"
            assert tree.children is not None
            assert len(tree.children) == 1
            child_node = tree.children[0]
            assert child_node.id == child.id
            assert child_node.level_label == "child"
            assert child_node.children is not None
            assert len(child_node.children) == 1
            grandchild_node = child_node.children[0]
            assert grandchild_node.id == grandchild.id
            assert grandchild_node.level_label == "grandchild"
            assert grandchild_node.children == []

        await db.dispose()

    async def test_build_tree_returns_none_for_missing_root(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            result = await manager.build_tree(ULID())
            assert result is None

        await db.dispose()

    async def test_manager_without_hierarchy_has_null_labels(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            root = await manager.save(ArtifactIn(data={"name": "root"}))
            tree = await manager.build_tree(root.id)
            assert tree is not None
            assert tree.level_label is None
            assert tree.hierarchy is None

        await db.dispose()

    async def test_build_tree_handles_parent_missing_in_db(self) -> None:
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            root = Artifact(data={"name": "orphan"}, parent_id=None, level=0)
            await repo.save(root)
            await repo.commit()

            child = Artifact(data={"name": "stray"}, parent_id=root.id, level=1)
            await repo.save(child)
            await repo.commit()

            # Delete the parent so the child references a missing parent record
            parent_entity = await repo.find_by_id(root.id)
            assert parent_entity is not None
            await repo.delete(parent_entity)
            await repo.commit()

            level = await manager._compute_level(child.parent_id)
            assert level == 0

        await db.dispose()

    async def test_find_subtree_returns_output_schemas(self) -> None:
        """Test that find_subtree returns ArtifactOut schemas."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Create parent and child
            parent_in = ArtifactIn(data={"name": "parent"})
            parent_out = await manager.save(parent_in)

            assert parent_out.id is not None
            child_in = ArtifactIn(data={"name": "child"}, parent_id=parent_out.id)
            await manager.save(child_in)

            # Find subtree
            subtree = await manager.find_subtree(parent_out.id)

            assert len(subtree) == 2
            assert all(isinstance(artifact, ArtifactOut) for artifact in subtree)
            assert all(artifact.id is not None for artifact in subtree)
            levels = {artifact.level for artifact in subtree}
            assert levels == {0, 1}

        await db.dispose()

    async def test_find_by_id(self) -> None:
        """Test finding artifact by ID through manager."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Create artifact
            artifact_in = ArtifactIn(data={"key": "value"})
            saved = await manager.save(artifact_in)

            # Find by ID
            assert saved.id is not None
            found = await manager.find_by_id(saved.id)

            assert found is not None
            assert isinstance(found, ArtifactOut)
            assert found.id == saved.id
            assert found.data == {"key": "value"}
            assert found.level == 0

            # Non-existent ID should return None
            random_id = ULID()
            not_found = await manager.find_by_id(random_id)
            assert not_found is None

        await db.dispose()

    async def test_delete_artifact(self) -> None:
        """Test deleting an artifact through manager."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Create artifact
            artifact_in = ArtifactIn(data={"to": "delete"})
            saved = await manager.save(artifact_in)

            # Verify it exists
            assert await manager.count() == 1

            # Delete it
            assert saved.id is not None
            await manager.delete_by_id(saved.id)

            # Verify it's gone
            assert await manager.count() == 0

        await db.dispose()

    async def test_output_schema_includes_timestamps(self) -> None:
        """Test that ArtifactOut schemas include created_at and updated_at timestamps."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            repo = ArtifactRepository(session)
            manager = ArtifactManager(repo)

            # Create artifact
            artifact_in = ArtifactIn(data={"test": "timestamps"})
            result = await manager.save(artifact_in)

            # Verify timestamps exist
            assert result.created_at is not None
            assert result.updated_at is not None
            assert result.id is not None
            assert result.level == 0

        await db.dispose()


def test_dataframe_round_trip() -> None:
    """DataFrame should round-trip DataFrame data."""
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    wrapper = DataFrame.from_pandas(df)

    assert wrapper.columns == ["col1", "col2"]
    assert wrapper.data == [[1, "a"], [2, "b"]]

    reconstructed = wrapper.to_pandas()
    pd.testing.assert_frame_equal(reconstructed, df)


def test_dataframe_requires_dataframe() -> None:
    """DataFrame.from_pandas should validate input type."""
    with pytest.raises(TypeError):
        DataFrame.from_pandas({"not": "dataframe"})  # pyright: ignore[reportArgumentType]
