"""Artifact ORM model for hierarchical data storage."""

from __future__ import annotations

from typing import Any

from servicekit.models import Entity
from servicekit.types import ULIDType
from sqlalchemy import ForeignKey, PickleType
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ulid import ULID


class Artifact(Entity):
    """ORM model for hierarchical artifacts with parent-child relationships."""

    __tablename__ = "artifacts"

    parent_id: Mapped[ULID | None] = mapped_column(
        ULIDType,
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    parent: Mapped[Artifact | None] = relationship(
        remote_side="Artifact.id",
        back_populates="children",
    )

    children: Mapped[list[Artifact]] = relationship(
        back_populates="parent",
    )

    data: Mapped[Any] = mapped_column(PickleType(protocol=4), nullable=False)
    level: Mapped[int] = mapped_column(default=0, nullable=False, index=True)
