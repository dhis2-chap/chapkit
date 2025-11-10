"""Initial database schema migration."""

from alembic import op

from chapkit.alembic_helpers import (
    create_artifacts_table,
    create_config_artifacts_table,
    create_configs_table,
    drop_artifacts_table,
    drop_config_artifacts_table,
    drop_configs_table,
)

# revision identifiers, used by Alembic.
revision = "4d869b5fb06e"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply database schema changes."""
    # Chapkit domain tables
    create_artifacts_table(op)
    create_configs_table(op)
    create_config_artifacts_table(op)


def downgrade() -> None:
    """Revert database schema changes."""
    # Drop in reverse order
    drop_config_artifacts_table(op)
    drop_configs_table(op)
    drop_artifacts_table(op)
