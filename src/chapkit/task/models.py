"""Task ORM model for reusable command templates."""

from __future__ import annotations

from servicekit.models import Entity
from sqlalchemy import Boolean
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Text


class Task(Entity):
    """ORM model for reusable task templates containing commands to execute."""

    __tablename__ = "tasks"

    command: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(Text, nullable=False, default="shell", server_default="shell")
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
