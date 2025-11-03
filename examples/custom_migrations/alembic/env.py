"""Alembic environment configuration for custom migrations."""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from servicekit import Base
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Add parent directory to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import custom models so they register with Base.metadata
# This ensures Alembic detects them during autogenerate

# OPTIONAL: Import chapkit's models if you want those tables too
# Uncomment these lines to include chapkit's Config, Artifact, and Task tables:
#
# from chapkit.config.models import Config, ConfigArtifact
# from chapkit.artifact.models import Artifact
# from chapkit.task.models import Task
#
# With these imports, your database will have BOTH chapkit tables AND custom tables

# Alembic Config object
config = context.config

# Configure logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata includes both servicekit models and custom models
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
