"""Custom models demonstrating user's own database tables."""

from servicekit.models import Entity
from servicekit.types import ULIDType
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from ulid import ULID


class User(Entity):
    """User account model with email and name."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True, index=True)
    name: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)


class Order(Entity):
    """Order model demonstrating foreign key to User."""

    __tablename__ = "orders"

    user_id: Mapped[ULID] = mapped_column(
        ULIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    total_amount: Mapped[float]
    status: Mapped[str] = mapped_column(default="pending")
