"""Pydantic schemas for hierarchical artifacts with tree structures."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Mapping, Self

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


# Typed artifact data schemas


class BaseArtifactData[MetadataT: BaseModel](BaseModel):
    """Base class for all artifact data types with typed metadata."""

    type: str = Field(description="Discriminator field for artifact type")
    metadata: MetadataT = Field(description="Strongly-typed JSON-serializable metadata")
    content: JsonSafe = Field(description="Content as Python object (bytes, DataFrame, models, etc.)")
    content_type: str | None = Field(default=None, description="MIME type for download responses")
    content_size: int | None = Field(default=None, description="Size of content in bytes")

    model_config = {"extra": "forbid"}


class MLMetadata(BaseModel):
    """Metadata for ML artifacts (training and prediction)."""

    status: Literal["success", "failed"] = Field(description="Job execution status")
    config_id: str = Field(description="ID of the config used for this operation")
    started_at: str = Field(description="ISO 8601 timestamp when operation started")
    completed_at: str = Field(description="ISO 8601 timestamp when operation completed")
    duration_seconds: float = Field(description="Operation duration in seconds")


class GenericMetadata(BaseModel):
    """Free-form metadata for generic artifacts."""

    model_config = {"extra": "allow"}


class MLTrainingArtifactData(BaseArtifactData[MLMetadata]):
    """Schema for ML training artifact data with trained model."""

    type: Literal["ml_training"] = Field(default="ml_training", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    metadata: MLMetadata


class MLPredictionArtifactData(BaseArtifactData[MLMetadata]):
    """Schema for ML prediction artifact data with results."""

    type: Literal["ml_prediction"] = Field(default="ml_prediction", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    metadata: MLMetadata


class GenericArtifactData(BaseArtifactData[GenericMetadata]):
    """Schema for generic artifact data with free-form metadata."""

    type: Literal["generic"] = Field(default="generic", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    metadata: GenericMetadata


ArtifactData = Annotated[
    MLTrainingArtifactData | MLPredictionArtifactData | GenericArtifactData,
    Field(discriminator="type"),
]
"""Discriminated union type for all artifact data types."""


def validate_artifact_data(data: dict[str, Any]) -> BaseArtifactData:
    """Validate artifact data against appropriate schema based on type field."""
    artifact_type = data.get("type", "generic")

    schema_map: dict[str, type[BaseArtifactData]] = {
        "ml_training": MLTrainingArtifactData,
        "ml_prediction": MLPredictionArtifactData,
        "generic": GenericArtifactData,
    }

    schema = schema_map.get(artifact_type, GenericArtifactData)
    return schema.model_validate(data)
