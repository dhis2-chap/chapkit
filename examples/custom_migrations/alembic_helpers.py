"""Migration helpers for custom tables following chapkit's pattern."""

from typing import Any

import servicekit.types
import sqlalchemy as sa


def create_users_table(op: Any) -> None:
    """Create users table for user accounts."""
    op.create_table(
        "users",
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)


def drop_users_table(op: Any) -> None:
    """Drop users table."""
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")


def create_orders_table(op: Any) -> None:
    """Create orders table with user relationship."""
    op.create_table(
        "orders",
        sa.Column("user_id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("total_amount", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_orders_user_id"), "orders", ["user_id"], unique=False)


def drop_orders_table(op: Any) -> None:
    """Drop orders table."""
    op.drop_index(op.f("ix_orders_user_id"), table_name="orders")
    op.drop_table("orders")
