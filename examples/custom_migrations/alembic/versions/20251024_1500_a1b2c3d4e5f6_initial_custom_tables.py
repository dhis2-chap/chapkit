"""Initial custom tables using helper pattern."""

# Import your custom helpers
from alembic import op
from alembic_helpers import create_orders_table, create_users_table, drop_orders_table, drop_users_table

# OPTIONAL: Import chapkit's helpers to create chapkit tables
# Uncomment these lines to include chapkit's Config, Artifact, Task tables:
#
# from chapkit.alembic_helpers import (
#     create_artifacts_table,
#     create_config_artifacts_table,
#     create_configs_table,
#     create_tasks_table,
#     drop_artifacts_table,
#     drop_config_artifacts_table,
#     drop_configs_table,
#     drop_tasks_table,
# )

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply database schema changes."""
    # Create chapkit tables (if imported above)
    # create_artifacts_table(op)
    # create_configs_table(op)
    # create_config_artifacts_table(op)
    # create_tasks_table(op)

    # Create custom tables
    create_users_table(op)
    create_orders_table(op)


def downgrade() -> None:
    """Revert database schema changes."""
    # Drop in reverse order
    drop_orders_table(op)
    drop_users_table(op)

    # Drop chapkit tables (if imported above)
    # drop_tasks_table(op)
    # drop_config_artifacts_table(op)
    # drop_configs_table(op)
    # drop_artifacts_table(op)
