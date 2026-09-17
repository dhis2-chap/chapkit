"""Config schemas for key-value configuration with JSON data."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, field_serializer, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo
from servicekit.schemas import EntityIn, EntityOut
from ulid import ULID


class BaseConfig(BaseModel):
    """Base class for configuration schemas with arbitrary extra fields allowed."""

    model_config = {"extra": "allow"}

    # Reserved parameters (CHAP-interpreted). chap-core does not send prediction_periods
    # when it creates a config, so it must have a default; the horizon itself comes from
    # the future frame at predict time.
    prediction_periods: int = 3
    additional_continuous_covariates: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def hoist_user_option_values(cls, data: object) -> object:
        """Accept chap-core's nested user_option_values payload as flat fields.

        chap-core creates configs as {"name": ..., "user_option_values": {...},
        "additional_continuous_covariates": [...]}. Without this hook the nested dict
        would be stored as an opaque extra field, every declared tunable would keep
        its default, and dump_config_yaml(format="chap_core") would re-nest it one
        level too deep. Flat keys win on conflict, so payloads that already post the
        fields flat (chapkit test, hand-written clients) are unchanged. Keys other
        than user_option_values are left as they are.
        """
        if not isinstance(data, dict):
            return data
        payload = cast(dict[str, object], data)
        nested = payload.get("user_option_values")
        if not isinstance(nested, dict):
            return payload
        hoisted: dict[str, object] = {k: v for k, v in payload.items() if k != "user_option_values"}
        for key, value in cast(dict[str, object], nested).items():
            if key not in hoisted:
                hoisted[key] = value
        return hoisted


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
