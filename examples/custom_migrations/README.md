# Custom Migrations Example

This example demonstrates how to:
- Create custom database models (User, Order)
- Set up your own Alembic migrations
- Reuse chapkit's alembic_helpers pattern
- Integrate custom models with servicekit's infrastructure

## Project Structure

```
custom_migrations/
├── main.py                  # FastAPI service with custom models
├── models.py                # User and Order ORM models
├── alembic_helpers.py       # Migration helper functions
├── alembic.ini              # Alembic configuration
├── alembic/
│   ├── env.py              # Imports servicekit.Base + custom models
│   ├── script.py.mako      # Migration template
│   └── versions/
│       └── 20251024_1500_..._initial_custom_tables.py
└── README.md               # This file
```

## Models

### User Model

```python
class User(Entity):
    """User account model with email and name."""
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True, index=True)
    name: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)
```

Inherits from `servicekit.models.Entity`, which provides:
- `id`: ULID primary key
- `created_at`, `updated_at`: Timestamps
- `tags`: JSON array for tagging

### Order Model

```python
class Order(Entity):
    """Order model with foreign key to User."""
    __tablename__ = "orders"

    user_id: Mapped[ULID] = mapped_column(
        ULIDType,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    total_amount: Mapped[float]
    status: Mapped[str] = mapped_column(default="pending")
```

Demonstrates:
- Foreign key relationships with CASCADE delete
- Using `ULIDType` for foreign keys
- Default values

## Setup

### 1. Install Dependencies

Already installed if you have chapkit:

```bash
cd examples/custom_migrations
```

### 2. Apply Migrations

```bash
# View migration
cat alembic/versions/20251024_1500_a1b2c3d4e5f6_initial_custom_tables.py

# Apply migration
uv run alembic upgrade head

# Verify tables were created
sqlite3 custom_migrations.db "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
# Output: alembic_version, orders, users

# View schema
sqlite3 custom_migrations.db ".schema users"
sqlite3 custom_migrations.db ".schema orders"
```

### 3. Run Service

```bash
fastapi dev main.py
```

Visit http://localhost:8000/docs to see the Swagger UI.

## Key Integration Points

### 1. Alembic env.py imports servicekit.Base

```python
from servicekit import Base
from models import User, Order  # Import custom models

target_metadata = Base.metadata  # Includes all imported models
```

**Important**: This example creates only custom tables (User, Order). Chapkit tables are **not** created.

**To include chapkit tables**, you have two options:

**Option 1: Use chapkit's alembic_helpers (Recommended)**

Uncomment the helper imports in your migration file:
```python
# In alembic/versions/20251024_1500_a1b2c3d4e5f6_initial_custom_tables.py
from chapkit.alembic_helpers import (
    create_artifacts_table,
    create_configs_table,
    create_config_artifacts_table,
    create_tasks_table,
)

def upgrade() -> None:
    create_artifacts_table(op)
    create_configs_table(op)
    create_config_artifacts_table(op)
    create_tasks_table(op)
    create_users_table(op)
    create_orders_table(op)
```

**Option 2: Import models and autogenerate**

Import chapkit models in `alembic/env.py` and regenerate migrations:
```python
from chapkit.config.models import Config, ConfigArtifact
from chapkit.artifact.models import Artifact
from chapkit.task.models import Task
```

**Option 1 is recommended** because it:
- Follows the reusable helper pattern
- No need to regenerate migrations
- Easier to maintain and understand

### 2. Migration Helpers Pattern

This example demonstrates **two levels of helpers**:

**Your custom helpers** (`alembic_helpers.py`):
```python
def create_users_table(op: Any) -> None:
    """Create users table for user accounts."""
    op.create_table(
        "users",
        sa.Column("email", sa.String(), nullable=False),
        # ... columns
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)
```

**Chapkit's helpers** (imported from `chapkit.alembic_helpers`):
```python
from chapkit.alembic_helpers import create_configs_table, create_artifacts_table
```

Benefits:
- Reusable across migrations
- Mix your helpers with chapkit's helpers
- Clear documentation
- Consistent patterns
- Easy to maintain

### 3. Using Helpers in Migrations

```python
from alembic_helpers import create_users_table, drop_users_table
from chapkit.alembic_helpers import create_configs_table, drop_configs_table

def upgrade() -> None:
    # Chapkit tables
    create_configs_table(op)
    # Your tables
    create_users_table(op)

def downgrade() -> None:
    drop_users_table(op)
    drop_configs_table(op)
```

## Generating New Migrations

### Using Autogenerate

```bash
# Add a new field to User model
# models.py: phone: Mapped[str] = mapped_column(nullable=True)

# Generate migration
uv run alembic revision --autogenerate -m "add phone to user"

# Format
uv run ruff format alembic/versions/

# Apply
uv run alembic upgrade head
```

### Manual Migration

```bash
# Create empty migration
uv run alembic revision -m "add custom index"

# Edit the file in alembic/versions/
# Add your SQL operations

# Apply
uv run alembic upgrade head
```

## Working with Data

### Create User

```bash
curl -X POST http://localhost:8000/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "name": "John Doe"}'
```

### Create Order

```bash
curl -X POST http://localhost:8000/api/v1/orders \
  -H "Content-Type: application/json" \
  -d '{"user_id": "01HZERO...", "total_amount": 99.99, "status": "pending"}'
```

## Migration Commands

```bash
# View migration history
uv run alembic history

# View current version
uv run alembic current

# Upgrade to specific revision
uv run alembic upgrade <revision>

# Downgrade one step
uv run alembic downgrade -1

# Downgrade to base (empty database)
uv run alembic downgrade base

# View SQL without executing
uv run alembic upgrade head --sql
```

## Testing Migrations

```bash
# Test upgrade
uv run alembic upgrade head

# Test downgrade
uv run alembic downgrade base

# Test upgrade again
uv run alembic upgrade head

# Verify schema
sqlite3 custom_migrations.db ".schema"
```

## Cleanup

```bash
# Remove database
rm custom_migrations.db

# Regenerate from migrations
uv run alembic upgrade head
```

## Next Steps

- Add more models (Product, Category, etc.)
- Create relationships between models
- Add custom repositories and managers
- Implement business logic
- Add API endpoints for CRUD operations

See the [Database Migrations Guide](../../docs/guides/database-migrations.md) for more details.
