"""Service-layer tests for artifact management."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

from ulid import ULID

from chapkit import Artifact, ArtifactHierarchy, ArtifactManager, ArtifactRepository


def test_artifact_manager_should_assign_field_controls_level() -> None:
    """_should_assign_field should skip assigning level when the value is None."""
    manager = ArtifactManager(cast(ArtifactRepository, SimpleNamespace()))

    assert manager._should_assign_field("data", {"foo": "bar"}) is True
    assert manager._should_assign_field("level", 2) is True
    assert manager._should_assign_field("level", None) is False


def test_artifact_manager_to_tree_node_applies_hierarchy_metadata() -> None:
    """_to_tree_node should attach hierarchy metadata when configured."""
    hierarchy = ArtifactHierarchy(name="ml", level_labels={0: "root"})
    manager = ArtifactManager(cast(ArtifactRepository, SimpleNamespace()), hierarchy=hierarchy)

    now = datetime.now(timezone.utc)
    entity = SimpleNamespace(
        id=ULID(),
        data={"name": "root"},
        parent_id=None,
        level=0,
        created_at=now,
        updated_at=now,
    )

    node = manager._to_tree_node(cast(Artifact, entity))

    assert node.level_label == "root"
    assert node.hierarchy == "ml"
    assert node.children is None


class StaticArtifactRepository:
    """Minimal repository stub returning a predefined subtree."""

    def __init__(self, nodes: list[SimpleNamespace]) -> None:
        self._nodes = nodes

    async def find_subtree(self, start_id: ULID) -> list[Artifact]:
        return [cast(Artifact, node) for node in self._nodes]


async def test_artifact_manager_build_tree_with_unordered_results() -> None:
    """build_tree should assemble a nested tree even when repository order is arbitrary."""
    now = datetime.now(timezone.utc)
    root_id = ULID()
    child_id = ULID()
    grandchild_id = ULID()

    def make_entity(identifier: ULID, parent: ULID | None, level: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=identifier,
            parent_id=parent,
            level=level,
            data={"name": str(identifier)},
            created_at=now,
            updated_at=now,
        )

    nodes = [
        make_entity(child_id, root_id, 1),
        make_entity(grandchild_id, child_id, 2),
        make_entity(root_id, None, 0),
    ]

    repo = StaticArtifactRepository(nodes)
    manager = ArtifactManager(cast(ArtifactRepository, repo))

    tree = await manager.build_tree(root_id)

    assert tree is not None
    assert tree.id == root_id
    assert tree.children is not None
    assert len(tree.children) == 1
    child_node = tree.children[0]
    assert child_node.id == child_id
    assert child_node.children is not None
    assert len(child_node.children) == 1
    grandchild_node = child_node.children[0]
    assert grandchild_node.id == grandchild_id
    assert grandchild_node.children == []
