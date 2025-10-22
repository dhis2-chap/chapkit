"""Pydantic schemas for hierarchical artifacts with tree structures."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Self

from pydantic import BaseModel, Field
from servicekit.schemas import EntityIn, EntityOut
from servicekit.types import JsonSafe
from ulid import ULID


class ArtifactIn(EntityIn):
    """Input schema for creating or updating artifacts."""

    data: Any
    parent_id: ULID | None = None
    level: int | None = None


class ArtifactOut(EntityOut):
    """Output schema for artifact entities."""

    data: JsonSafe
    parent_id: ULID | None = None
    level: int


class ArtifactTreeNode(ArtifactOut):
    """Artifact node with tree structure metadata."""

    level_label: str | None = None
    hierarchy: str | None = None
    children: list["ArtifactTreeNode"] | None = None

    @classmethod
    def from_artifact(cls, artifact: ArtifactOut) -> Self:
        """Create a tree node from an artifact output schema."""
        return cls.model_validate(artifact.model_dump())


class ArtifactHierarchy(BaseModel):
    """Configuration for artifact hierarchy with level labels."""

    name: str = Field(..., description="Human readable name of this hierarchy")
    level_labels: Mapping[int, str] = Field(
        default_factory=dict,
        description="Mapping of numeric levels to labels (0 -> 'train', etc.)",
    )

    model_config = {"frozen": True}

    hierarchy_key: ClassVar[str] = "hierarchy"
    depth_key: ClassVar[str] = "level_depth"
    label_key: ClassVar[str] = "level_label"

    def label_for(self, level: int) -> str:
        """Get the label for a given level or return default."""
        return self.level_labels.get(level, f"level_{level}")

    def describe(self, level: int) -> dict[str, Any]:
        """Get hierarchy metadata dict for a given level."""
        return {
            self.hierarchy_key: self.name,
            self.depth_key: level,
            self.label_key: self.label_for(level),
        }
