"""Tests for database migrations with file-based SQLite."""

import tempfile
from pathlib import Path

from servicekit import SqliteDatabaseBuilder

from chapkit.artifact.models import Artifact
from chapkit.config.models import Config
from chapkit.task.models import Task

# Path to chapkit's alembic directory
ALEMBIC_DIR = Path(__file__).parent.parent / "alembic"


class TestMigrations:
    """Test that migrations work correctly with file-based databases."""

    async def test_file_database_applies_migrations_with_tags_column(self) -> None:
        """Test that file-based database applies migrations including tags column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create database - this will run migrations
            db = (
                SqliteDatabaseBuilder.from_file(str(db_path))
                .with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR)
                .build()
            )
            await db.init()

            try:
                # Verify we can create entities with tags
                async with db.session() as session:
                    # Test Config with tags
                    config = Config(
                        name="test-config",
                        data={"key": "value"},
                        tags=["config-tag", "production"],
                    )
                    session.add(config)
                    await session.commit()
                    await session.refresh(config)

                    assert config.tags == ["config-tag", "production"]

                # Verify we can query entities with tags
                async with db.session() as session:
                    from sqlalchemy import select

                    result = await session.execute(select(Config))
                    loaded_config = result.scalar_one()
                    assert loaded_config.tags == ["config-tag", "production"]

                # Test Task with tags
                async with db.session() as session:
                    task = Task(
                        command="echo test",
                        task_type="shell",
                        tags=["task-tag", "daily"],
                    )
                    session.add(task)
                    await session.commit()
                    await session.refresh(task)

                    assert task.tags == ["task-tag", "daily"]

                # Test Artifact with tags
                async with db.session() as session:
                    artifact = Artifact(
                        data={"result": 42},
                        level=0,
                        tags=["artifact-tag", "ml-output"],
                    )
                    session.add(artifact)
                    await session.commit()
                    await session.refresh(artifact)

                    assert artifact.tags == ["artifact-tag", "ml-output"]

            finally:
                await db.dispose()

    async def test_file_database_default_empty_list_for_tags(self) -> None:
        """Test that tags defaults to empty list when not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            db = (
                SqliteDatabaseBuilder.from_file(str(db_path))
                .with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR)
                .build()
            )
            await db.init()

            try:
                async with db.session() as session:
                    config = Config(name="test", data={"x": 1})
                    session.add(config)
                    await session.commit()
                    await session.refresh(config)

                    # Should default to empty list
                    assert config.tags == []

            finally:
                await db.dispose()

    async def test_migration_schema_includes_tags_column(self) -> None:
        """Test that the migration schema actually includes tags column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            db = (
                SqliteDatabaseBuilder.from_file(str(db_path))
                .with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR)
                .build()
            )
            await db.init()

            try:
                # Check that tags column exists in the schema
                async with db.session() as session:
                    from sqlalchemy import text

                    # Check configs table
                    result = await session.execute(text("PRAGMA table_info(configs)"))
                    columns = {row[1] for row in result}
                    assert "tags" in columns, "configs table missing tags column"

                    # Check tasks table
                    result = await session.execute(text("PRAGMA table_info(tasks)"))
                    columns = {row[1] for row in result}
                    assert "tags" in columns, "tasks table missing tags column"

                    # Check artifacts table
                    result = await session.execute(text("PRAGMA table_info(artifacts)"))
                    columns = {row[1] for row in result}
                    assert "tags" in columns, "artifacts table missing tags column"

            finally:
                await db.dispose()

    async def test_alembic_helpers_drop_functions(self) -> None:
        """Test that alembic helper drop functions work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            db = (
                SqliteDatabaseBuilder.from_file(str(db_path))
                .with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR)
                .build()
            )
            await db.init()

            try:
                # Import alembic helpers
                from chapkit import alembic_helpers

                # Create a mock op object to test drop functions
                class MockOp:
                    """Mock Alembic operations object."""

                    def __init__(self) -> None:
                        self.dropped_indices: list[str] = []
                        self.dropped_tables: list[str] = []

                    def f(self, name: str) -> str:
                        """Mock f() method that returns the input name."""
                        return name

                    def drop_index(self, index_name: str, table_name: str) -> None:
                        """Record dropped index."""
                        self.dropped_indices.append(index_name)

                    def drop_table(self, table_name: str) -> None:
                        """Record dropped table."""
                        self.dropped_tables.append(table_name)

                # Test drop_artifacts_table
                mock_op = MockOp()
                alembic_helpers.drop_artifacts_table(mock_op)
                assert "artifacts" in mock_op.dropped_tables
                assert len(mock_op.dropped_indices) == 2

                # Test drop_configs_table
                mock_op = MockOp()
                alembic_helpers.drop_configs_table(mock_op)
                assert "configs" in mock_op.dropped_tables
                assert len(mock_op.dropped_indices) == 1

                # Test drop_config_artifacts_table
                mock_op = MockOp()
                alembic_helpers.drop_config_artifacts_table(mock_op)
                assert "config_artifacts" in mock_op.dropped_tables

                # Test drop_tasks_table
                mock_op = MockOp()
                alembic_helpers.drop_tasks_table(mock_op)
                assert "tasks" in mock_op.dropped_tables

            finally:
                await db.dispose()
