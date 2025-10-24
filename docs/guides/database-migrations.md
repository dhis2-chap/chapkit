# Database Migrations

Chapkit uses Alembic for database schema migrations, integrated with servicekit's async SQLAlchemy infrastructure. This guide covers the migration workflow and how to extend chapkit with your own custom database models.

## Quick Start

```bash
# Generate a new migration
make migrate MSG='add user table'

# Apply migrations
make upgrade

# Rollback last migration
make downgrade
```

Migrations are automatically applied when your application starts via `Database.init()`.

---

## Architecture Overview

### How It Works

1. **Base Metadata Registry**: All ORM models inherit from `servicekit.models.Base` or `servicekit.models.Entity`
2. **Model Registration**: Models with `__tablename__` are automatically registered with `Base.metadata`
3. **Alembic Auto-detection**: Alembic's `env.py` uses `Base.metadata` as `target_metadata` to discover all tables
4. **Migration Generation**: Alembic compares ORM models to current database schema and generates SQL operations

### Chapkit's Tables

Chapkit provides these domain models:
- **configs** - Configuration key-value storage
- **config_artifacts** - Junction table linking configs to artifacts
- **artifacts** - Hierarchical artifact storage
- **tasks** - Task execution infrastructure

All inherit from `servicekit.models.Entity` which provides: `id` (ULID), `created_at`, `updated_at`, `tags`.

---

## Basic Workflow

### 1. Modify Your Models

```python
# src/myapp/models.py
from servicekit.models import Entity
from sqlalchemy.orm import Mapped, mapped_column

class User(Entity):
    """User model with email and name."""
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True, index=True)
    name: Mapped[str]
```

**Important**: Models must inherit from `servicekit.models.Entity` or `Base` to be detected by Alembic.

### 2. Generate Migration

```bash
make migrate MSG='add user table'
```

This creates a timestamped migration file in `alembic/versions/`:
```
alembic/versions/20251024_1430_a1b2c3d4e5f6_add_user_table.py
```

### 3. Review Migration

```python
def upgrade() -> None:
    """Apply database schema changes."""
    op.create_table(
        'users',
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('id', servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('tags', sa.JSON(), nullable=False, server_default='[]'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)

def downgrade() -> None:
    """Revert database schema changes."""
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
```

### 4. Apply Migration

```bash
# Apply manually
make upgrade

# Or restart app (auto-applies on startup)
fastapi dev main.py
```

### 5. Verify Schema

```bash
# Check migration history
uv run alembic history

# Check current version
uv run alembic current

# View database schema (SQLite example)
sqlite3 app.db ".schema users"
```

---

## Using Your Own Alembic Setup

For larger projects, you may want to maintain your own Alembic configuration separate from chapkit while still reusing chapkit's infrastructure.

### Project Structure

```
myproject/
├── src/
│   └── myapp/
│       ├── models.py          # Your custom models
│       └── alembic_helpers.py # Optional: your migration helpers
├── alembic/                   # Your alembic directory
│   ├── env.py                 # Your env.py imports servicekit.Base
│   └── versions/              # Your migrations
├── alembic.ini                # Your alembic config
└── main.py
```

### Setup Steps

#### 1. Initialize Your Alembic

```bash
uv run alembic init alembic
```

#### 2. Configure alembic.ini

```ini
[alembic]
script_location = alembic
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s
timezone = UTC
sqlalchemy.url = sqlite+aiosqlite:///./app.db
```

#### 3. Update env.py

**Key Change**: Import `Base` from servicekit to include all models (both chapkit's and yours):

```python
"""Alembic environment configuration for async migrations."""
import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from servicekit import Base  # Import Base from servicekit
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your custom models so they register with Base.metadata
from myapp.models import User, Order  # Your models here

# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata from servicekit.Base (includes all models)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the provided connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Create event loop in thread to avoid conflicts with existing async code
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_async_migrations())
    finally:
        loop.close()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Critical Points**:
- Import `Base` from `servicekit`
- Import all your custom models at the top
- Optionally import chapkit's models if you want those tables too
- Set `target_metadata = Base.metadata`

**Note**: Only **imported** models are included in migrations. If you don't import chapkit's models (Config, Artifact, Task), those tables won't be created. The example in `examples/custom_migrations/` demonstrates a standalone setup with only custom tables.

#### 4. Create Your Models

```python
# src/myapp/models.py
"""Custom application models."""
from servicekit.models import Entity
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from servicekit.types import ULIDType
from ulid import ULID

class User(Entity):
    """User account model."""
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True, index=True)
    name: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)

class Order(Entity):
    """Order model with user relationship."""
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

#### 5. Generate Migrations

```bash
# Using alembic directly
uv run alembic revision --autogenerate -m "add user and order tables"

# Format the generated file
uv run ruff format alembic/versions/
```

#### 6. Optional: Reuse Chapkit's Helper Pattern

You can create your own migration helpers following chapkit's pattern:

```python
# src/myapp/alembic_helpers.py
"""Migration helpers for myapp tables."""
from typing import Any
import sqlalchemy as sa
import servicekit.types


def create_users_table(op: Any) -> None:
    """Create users table."""
    op.create_table(
        'users',
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('id', servicekit.types.ULIDType(length=26), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('tags', sa.JSON(), nullable=False, server_default='[]'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)


def drop_users_table(op: Any) -> None:
    """Drop users table."""
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
```

Then use in migrations, optionally mixing with chapkit's helpers:

```python
"""Add user and order tables."""
from alembic import op
from myapp.alembic_helpers import create_users_table, drop_users_table

# OPTIONAL: Import chapkit's helpers if you want those tables too
from chapkit.alembic_helpers import (
    create_configs_table,
    create_artifacts_table,
    drop_configs_table,
    drop_artifacts_table,
)

revision = 'a1b2c3d4e5f6'
down_revision = None

def upgrade() -> None:
    """Apply database schema changes."""
    # Create chapkit tables (optional)
    create_configs_table(op)
    create_artifacts_table(op)

    # Create your custom tables
    create_users_table(op)

def downgrade() -> None:
    """Revert database schema changes."""
    drop_users_table(op)
    drop_artifacts_table(op)
    drop_configs_table(op)
```

**Benefits of Helpers**:
- Reusable across migrations and projects
- Mix chapkit's helpers with your own
- Consistent table definitions
- Easier to maintain
- Clear upgrade/downgrade operations
- No need to regenerate migrations when adding chapkit tables

---

## Model Import Requirements

**Critical**: Models must be imported before Alembic runs for auto-detection to work.

### Where to Import

**Option 1: In alembic/env.py** (Recommended)
```python
from servicekit import Base
from myapp.models import User, Order, Product  # Explicit imports
```

**Option 2: In your app module**
```python
# src/myapp/__init__.py
from .models import User, Order, Product

__all__ = ["User", "Order", "Product"]
```

Then import in env.py:
```python
import myapp  # Imports __init__.py which imports models
```

### Common Mistake

```python
# BAD: Models not imported
from servicekit import Base
target_metadata = Base.metadata  # Won't include your models!
```

```python
# GOOD: Models imported before metadata is used
from servicekit import Base
from myapp.models import User, Order
target_metadata = Base.metadata  # Now includes User and Order
```

---

## Mixing Chapkit and Custom Migrations

You can combine chapkit's tables with your custom tables in migrations.

**Recommended Approach: Reuse Helpers**

Import chapkit's helpers directly in your migrations:

```python
from alembic import op
from chapkit.alembic_helpers import (
    create_configs_table,
    create_artifacts_table,
    create_tasks_table,
)
from myapp.alembic_helpers import create_users_table

def upgrade() -> None:
    # Create chapkit tables
    create_configs_table(op)
    create_artifacts_table(op)
    create_tasks_table(op)

    # Create your tables
    create_users_table(op)
```

**Benefits**:
- ✅ No need to regenerate migrations
- ✅ Reuse tested helper functions
- ✅ Explicit control over table creation order
- ✅ Mix and match as needed
- ✅ Clear and maintainable

**Alternative: Single Alembic with All Models**

Import all models in `env.py` and use autogenerate:

```python
# Import chapkit models
from chapkit.config.models import Config
from chapkit.artifact.models import Artifact

# Import your models
from myapp.models import User, Order

# All models registered in Base.metadata
target_metadata = Base.metadata
```

Then: `uv run alembic revision --autogenerate -m "create all tables"`

**Choose the helper approach when**:
- You want explicit control
- You're following the helper pattern
- You don't need autogenerate for these tables

**Choose autogenerate when**:
- Models change frequently
- You prefer automatic detection
- You want Alembic to track differences

---

## Troubleshooting

### Models Not Detected

**Problem**: `make migrate` doesn't generate changes for your new model.

**Cause**: Model not imported or not inheriting from Base/Entity.

**Solution**:
```python
# 1. Verify model inherits from Entity or Base
from servicekit.models import Entity

class User(Entity):  # Must inherit from Entity or Base
    __tablename__ = "users"
    # ...

# 2. Import model in alembic/env.py
from myapp.models import User  # Add this line

# 3. Verify import works
python -c "from myapp.models import User; print(User.__tablename__)"
```

### Migration Conflicts

**Problem**: "Multiple heads" or "Can't determine base revision".

**Cause**: Branched revision history (two migrations with same parent).

**Solution**:
```bash
# View all revisions
uv run alembic branches

# Merge branches
uv run alembic merge -m "merge branches" head1 head2

# Apply merged migration
make upgrade
```

### Import Errors During Migration

**Problem**: Migration fails with import errors.

**Cause**: Model imports fail due to circular dependencies or missing packages.

**Solution**:
- Use relative imports in models
- Avoid importing app-level code in models
- Only import SQLAlchemy and servicekit in model files

### Autogenerate Not Detecting Changes

**Problem**: Changed a column but autogenerate ignores it.

**Cause**: Alembic doesn't always detect all changes automatically.

**Solution**:
```bash
# Create empty migration
uv run alembic revision -m "update user table"

# Manually add changes
def upgrade() -> None:
    op.alter_column('users', 'email', new_column_name='email_address')
```

### Database Locked (SQLite)

**Problem**: "Database is locked" during migration.

**Cause**: Another process has the database open.

**Solution**:
```bash
# Stop all running apps
# Then apply migration
make upgrade
```

For production, use PostgreSQL instead of SQLite to avoid locking issues.

---

## Production Considerations

### Backup Before Migrating

```bash
# SQLite backup
cp app.db app.db.backup

# PostgreSQL backup
pg_dump -U user -d database > backup.sql
```

### Test Migrations

```bash
# Apply migration
make upgrade

# Verify schema
sqlite3 app.db ".schema"

# Test rollback
make downgrade

# Reapply
make upgrade
```

### Migration in CI/CD

```bash
# In your deployment script
uv run alembic upgrade head
```

### Multiple Environments

Use environment variables for database URLs:

```python
# alembic/env.py
import os

config.set_main_option(
    "sqlalchemy.url",
    os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
)
```

---

## Complete Example

See `examples/custom_migrations/` for a working example with:
- Custom User and Order models
- Separate alembic setup
- Reusable migration helpers
- Integration with chapkit's infrastructure

---

## Next Steps

- **Database Guide**: Learn about servicekit's Database and SqliteDatabase classes
- **Models Guide**: Deep dive into Entity and custom ORM patterns
- **Testing**: Test migrations in CI/CD pipelines
