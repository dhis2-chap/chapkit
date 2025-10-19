"""Artifact manager for hierarchical data with parent-child relationships."""

from __future__ import annotations

from collections import deque

from servicekit.manager import BaseManager
from ulid import ULID

from .models import Artifact
from .repository import ArtifactRepository
from .schemas import ArtifactHierarchy, ArtifactIn, ArtifactOut, ArtifactTreeNode


class ArtifactManager(BaseManager[Artifact, ArtifactIn, ArtifactOut, ULID]):
    """Manager for Artifact entities with hierarchical tree operations."""

    def __init__(
        self,
        repo: ArtifactRepository,
        hierarchy: ArtifactHierarchy | None = None,
    ) -> None:
        """Initialize artifact manager with repository and optional hierarchy."""
        super().__init__(repo, Artifact, ArtifactOut)
        self.repository: ArtifactRepository = repo
        self.hierarchy = hierarchy

    # Public API ------------------------------------------------------

    async def find_subtree(self, start_id: ULID) -> list[ArtifactTreeNode]:
        """Find all artifacts in the subtree rooted at the given ID."""
        artifacts = await self.repository.find_subtree(start_id)
        return [self._to_tree_node(artifact) for artifact in artifacts]

    async def expand_artifact(self, artifact_id: ULID) -> ArtifactTreeNode | None:
        """Expand a single artifact with hierarchy metadata but without children."""
        artifact = await self.repository.find_by_id(artifact_id)
        if artifact is None:
            return None

        node = self._to_tree_node(artifact)
        node.children = None

        return node

    async def build_tree(self, start_id: ULID) -> ArtifactTreeNode | None:
        """Build a hierarchical tree structure rooted at the given artifact ID."""
        artifacts = await self.find_subtree(start_id)
        if not artifacts:
            return None

        node_map: dict[ULID, ArtifactTreeNode] = {}
        for node in artifacts:
            node.children = []
            node_map[node.id] = node

        for node in artifacts:
            if node.parent_id is None:
                continue
            parent = node_map.get(node.parent_id)
            if parent is None:
                continue
            if parent.children is None:
                parent.children = []
            parent.children.append(node)

        # Keep children as [] for leaf nodes (semantic: "loaded but empty")
        # Only expand_artifact sets children=None (semantic: "not loaded")

        root = node_map.get(start_id)

        return root

    # Lifecycle overrides --------------------------------------------

    def _should_assign_field(self, field: str, value: object) -> bool:
        """Prevent assigning None to level field during updates."""
        if field == "level" and value is None:
            return False
        return super()._should_assign_field(field, value)

    async def pre_save(self, entity: Artifact, data: ArtifactIn) -> None:
        """Compute and set artifact level before saving."""
        entity.level = await self._compute_level(entity.parent_id)

    async def pre_update(self, entity: Artifact, data: ArtifactIn, old_values: dict[str, object]) -> None:
        """Recalculate artifact level and cascade updates to descendants if parent changed."""
        previous_level = old_values.get("level", entity.level)
        entity.level = await self._compute_level(entity.parent_id)
        parent_changed = old_values.get("parent_id") != entity.parent_id
        if parent_changed or previous_level != entity.level:
            await self._recalculate_descendants(entity)

    # Helper utilities ------------------------------------------------

    async def _compute_level(self, parent_id: ULID | None) -> int:
        """Compute the level of an artifact based on its parent."""
        if parent_id is None:
            return 0
        parent = await self.repository.find_by_id(parent_id)
        if parent is None:
            return 0  # pragma: no cover
        return parent.level + 1

    async def _recalculate_descendants(self, entity: Artifact) -> None:
        """Recalculate levels for all descendants of an artifact."""
        subtree = await self.repository.find_subtree(entity.id)
        by_parent: dict[ULID | None, list[Artifact]] = {}
        for node in subtree:
            by_parent.setdefault(node.parent_id, []).append(node)

        queue: deque[Artifact] = deque([entity])
        while queue:
            current = queue.popleft()
            for child in by_parent.get(current.id, []):
                child.level = current.level + 1
                queue.append(child)

    def _to_tree_node(self, entity: Artifact) -> ArtifactTreeNode:
        """Convert artifact entity to tree node with hierarchy metadata."""
        base = super()._to_output_schema(entity)
        node = ArtifactTreeNode.from_artifact(base)
        if self.hierarchy is not None:
            meta = self.hierarchy.describe(node.level)
            hierarchy_value = meta.get(self.hierarchy.hierarchy_key)
            if hierarchy_value is not None:
                node.hierarchy = str(hierarchy_value)
            label_value = meta.get(self.hierarchy.label_key)
            if label_value is not None:
                node.level_label = str(label_value)

        return node
