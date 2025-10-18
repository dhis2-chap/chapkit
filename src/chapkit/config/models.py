"""Config ORM models for key-value configuration storage and artifact linking."""

from __future__ import annotations

from typing import Any

from servicekit.models import Base, Entity
from servicekit.types import ULIDType
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON
from ulid import ULID

from .schemas import BaseConfig


class Config(Entity):
    """ORM model for configuration with JSON data storage."""

    __tablename__ = "configs"

    name: Mapped[str] = mapped_column(index=True)
    _data_json: Mapped[dict[str, Any]] = mapped_column("data", JSON, nullable=False)

    @property
    def data(self) -> dict[str, Any]:
        """Return JSON data as dict."""
        return self._data_json

    @data.setter
    def data(self, value: BaseConfig | dict[str, Any]) -> None:
        """Serialize Pydantic model to JSON or store dict directly."""
        if isinstance(value, dict):
            self._data_json = value
        elif hasattr(value, "model_dump") and callable(value.model_dump):
            # BaseConfig or other Pydantic model
            self._data_json = value.model_dump(mode="json")
        else:
            raise TypeError(f"data must be a BaseConfig subclass or dict, got {type(value)}")


class ConfigArtifact(Base):
    """Junction table linking Configs to root Artifacts."""

    __tablename__ = "config_artifacts"

    config_id: Mapped[ULID] = mapped_column(
        ULIDType,
        ForeignKey("configs.id", ondelete="CASCADE"),
        primary_key=True,
    )

    artifact_id: Mapped[ULID] = mapped_column(
        ULIDType,
        ForeignKey("artifacts.id", ondelete="CASCADE"),
        primary_key=True,
        unique=True,
    )

    __table_args__ = (UniqueConstraint("artifact_id", name="uq_artifact_id"),)
