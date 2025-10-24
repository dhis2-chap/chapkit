"""Reusable Alembic migration helpers for chapkit tables.

This module provides helper functions for creating and dropping chapkit's database tables
in Alembic migrations. Using helpers instead of raw Alembic operations provides:

- Reusability across migrations
- Consistent table definitions
- Clear documentation
- Easier maintenance

Users can create their own helper modules following this pattern for custom tables.

Example:
    # In your migration file
    from chapkit.alembic_helpers import create_configs_table, drop_configs_table

    def upgrade() -> None:
        create_configs_table(op)

    def downgrade() -> None:
        drop_configs_table(op)

Creating Your Own Helpers:
    Follow the same pattern for your custom tables:

    # myapp/alembic_helpers.py
    def create_users_table(op: Any) -> None:
        '''Create users table.'''
        op.create_table(
            'users',
            sa.Column('email', sa.String(), nullable=False),
            sa.Column('name', sa.String(), nullable=False),
            sa.Column('id', servicekit.types.ULIDType(length=26), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
            sa.Column('tags', sa.JSON(), nullable=False, server_default='[]'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)

    def drop_users_table(op: Any) -> None:
        '''Drop users table.'''
        op.drop_index(op.f('ix_users_email'), table_name='users')
        op.drop_table('users')

See examples/custom_migrations/ for a complete working example.
"""

from typing import Any

import servicekit.types
import sqlalchemy as sa


def create_artifacts_table(op: Any) -> None:
    """Create artifacts table for hierarchical artifact storage."""
    op.create_table(
        "artifacts",
        sa.Column("parent_id", servicekit.types.ULIDType(length=26), nullable=True),
        sa.Column("data", sa.PickleType(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.ForeignKeyConstraint(["parent_id"], ["artifacts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_artifacts_level"), "artifacts", ["level"], unique=False)
    op.create_index(op.f("ix_artifacts_parent_id"), "artifacts", ["parent_id"], unique=False)


def drop_artifacts_table(op: Any) -> None:
    """Drop artifacts table."""
    op.drop_index(op.f("ix_artifacts_parent_id"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_level"), table_name="artifacts")
    op.drop_table("artifacts")


def create_configs_table(op: Any) -> None:
    """Create configs table for configuration storage."""
    op.create_table(
        "configs",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_configs_name"), "configs", ["name"], unique=False)


def drop_configs_table(op: Any) -> None:
    """Drop configs table."""
    op.drop_index(op.f("ix_configs_name"), table_name="configs")
    op.drop_table("configs")


def create_config_artifacts_table(op: Any) -> None:
    """Create config_artifacts junction table linking configs to artifacts."""
    op.create_table(
        "config_artifacts",
        sa.Column("config_id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("artifact_id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["config_id"], ["configs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("config_id", "artifact_id"),
        sa.UniqueConstraint("artifact_id"),
        sa.UniqueConstraint("artifact_id", name="uq_artifact_id"),
    )


def drop_config_artifacts_table(op: Any) -> None:
    """Drop config_artifacts junction table."""
    op.drop_table("config_artifacts")


def create_tasks_table(op: Any) -> None:
    """Create tasks table for task execution infrastructure."""
    op.create_table(
        "tasks",
        sa.Column("command", sa.Text(), nullable=False),
        sa.Column("task_type", sa.Text(), nullable=False, server_default="shell"),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("id", servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.PrimaryKeyConstraint("id"),
    )


def drop_tasks_table(op: Any) -> None:
    """Drop tasks table."""
    op.drop_table("tasks")
