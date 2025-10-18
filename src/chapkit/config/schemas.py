"""Config schemas for key-value configuration with JSON data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo
from servicekit.schemas import EntityIn, EntityOut
from ulid import ULID


class BaseConfig(BaseModel):
    """Base class for configuration schemas with arbitrary extra fields allowed."""

    model_config = {"extra": "allow"}


class ConfigIn[DataT: BaseConfig](EntityIn):
    """Input schema for creating or updating configurations."""

    name: str
    data: DataT


class ConfigOut[DataT: BaseConfig](EntityOut):
    """Output schema for configuration entities."""

    name: str
    data: DataT

    model_config = {"ser_json_timedelta": "float", "ser_json_bytes": "base64"}

    @field_validator("data", mode="before")
    @classmethod
    def convert_dict_to_model(cls, v: Any, info: ValidationInfo) -> Any:
        """Convert dict to BaseConfig model if data_cls is provided in validation context."""
        if isinstance(v, BaseConfig):
            return v
        if isinstance(v, dict):
            if info.context and "data_cls" in info.context:
                data_cls = info.context["data_cls"]
                return data_cls.model_validate(v)
        return v

    @field_serializer("data", when_used="json")
    def serialize_data(self, value: DataT) -> dict[str, Any]:
        """Serialize BaseConfig data to JSON dict."""
        if isinstance(value, BaseConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
            return value.model_dump(mode="json")
        return value


class LinkArtifactRequest(BaseModel):
    """Request schema for linking an artifact to a config."""

    artifact_id: ULID


class UnlinkArtifactRequest(BaseModel):
    """Request schema for unlinking an artifact from a config."""

    artifact_id: ULID
