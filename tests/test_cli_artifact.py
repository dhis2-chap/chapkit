"""Tests for artifact CLI commands."""

import asyncio
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from servicekit import SqliteDatabaseBuilder
from typer.testing import CliRunner
from ulid import ULID

from chapkit.artifact import Artifact
from chapkit.cli.cli import app

runner = CliRunner()

ALEMBIC_DIR = Path(__file__).parent.parent / "src" / "chapkit" / "alembic"


async def _create_db_with_zip_artifact(tmp_path: Path) -> tuple[Path, ULID]:
    """Create a database with a ZIP artifact for testing."""
    db_path = tmp_path / "test.db"
    db = SqliteDatabaseBuilder.from_file(str(db_path)).with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR).build()
    await db.init()

    # Create test ZIP content
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("test.txt", "Hello, World!")
        zf.writestr("subdir/nested.txt", "Nested content")
    zip_bytes = zip_buffer.getvalue()

    artifact_id = ULID()
    async with db.session() as session:
        artifact = Artifact(
            id=artifact_id,
            data={
                "type": "ml_training_workspace",
                "content": zip_bytes,
                "content_type": "application/zip",
                "content_size": len(zip_bytes),
                "metadata": {"status": "success"},
            },
            level=0,
        )
        session.add(artifact)
        await session.commit()

    await db.dispose()
    return db_path, artifact_id


async def _create_db_with_non_zip_artifact(tmp_path: Path) -> tuple[Path, ULID]:
    """Create a database with a non-ZIP artifact for testing."""
    db_path = tmp_path / "test.db"
    db = SqliteDatabaseBuilder.from_file(str(db_path)).with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR).build()
    await db.init()

    artifact_id = ULID()
    async with db.session() as session:
        artifact = Artifact(
            id=artifact_id,
            data={
                "type": "ml_prediction",
                "content": {"columns": ["a"], "data": [[1]]},
                "content_type": "application/json",
                "metadata": {"status": "success"},
            },
            level=0,
        )
        session.add(artifact)
        await session.commit()

    await db.dispose()
    return db_path, artifact_id


async def _create_empty_db(tmp_path: Path) -> Path:
    """Create an empty database for testing."""
    db_path = tmp_path / "empty.db"
    db = SqliteDatabaseBuilder.from_file(str(db_path)).with_migrations(enabled=True, alembic_dir=ALEMBIC_DIR).build()
    await db.init()
    await db.dispose()
    return db_path


@pytest.fixture
def db_with_zip_artifact(tmp_path: Path) -> tuple[Path, ULID]:
    """Create a database with a ZIP artifact for testing."""
    return asyncio.run(_create_db_with_zip_artifact(tmp_path))


@pytest.fixture
def db_with_non_zip_artifact(tmp_path: Path) -> tuple[Path, ULID]:
    """Create a database with a non-ZIP artifact for testing."""
    return asyncio.run(_create_db_with_non_zip_artifact(tmp_path))


@pytest.fixture
def empty_db(tmp_path: Path) -> Path:
    """Create an empty database for testing."""
    return asyncio.run(_create_empty_db(tmp_path))


class TestArtifactList:
    """Tests for artifact list command."""

    def test_list_empty_database(self, empty_db: Path) -> None:
        """Test listing artifacts from an empty database."""
        result = runner.invoke(app, ["artifact", "list", "--database", str(empty_db)])
        assert result.exit_code == 0
        assert "No artifacts found" in result.output

    def test_list_with_artifacts(self, db_with_zip_artifact: tuple[Path, ULID]) -> None:
        """Test listing artifacts shows correct information."""
        db_path, artifact_id = db_with_zip_artifact
        result = runner.invoke(app, ["artifact", "list", "--database", str(db_path)])
        assert result.exit_code == 0
        assert "ml_training_workspace" in result.output
        assert "zip" in result.output.lower()

    def test_list_filter_by_type(self, db_with_zip_artifact: tuple[Path, ULID]) -> None:
        """Test filtering artifacts by type."""
        db_path, _ = db_with_zip_artifact
        result = runner.invoke(
            app, ["artifact", "list", "--database", str(db_path), "--type", "ml_training_workspace"]
        )
        assert result.exit_code == 0
        assert "ml_training_workspace" in result.output

    def test_list_filter_by_nonexistent_type(self, db_with_zip_artifact: tuple[Path, ULID]) -> None:
        """Test filtering by non-existent type returns no results."""
        db_path, _ = db_with_zip_artifact
        result = runner.invoke(app, ["artifact", "list", "--database", str(db_path), "--type", "nonexistent"])
        assert result.exit_code == 0
        assert "No artifacts found" in result.output

    def test_list_missing_source(self) -> None:
        """Test list command requires either --database or --url."""
        result = runner.invoke(app, ["artifact", "list"])
        assert result.exit_code == 1
        assert "Must provide either --database or --url" in result.output

    def test_list_both_sources(self, tmp_path: Path) -> None:
        """Test list command rejects both --database and --url."""
        result = runner.invoke(
            app, ["artifact", "list", "--database", str(tmp_path / "db.db"), "--url", "http://localhost"]
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_list_database_not_found(self) -> None:
        """Test list command with non-existent database."""
        result = runner.invoke(app, ["artifact", "list", "--database", "/nonexistent/path.db"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestArtifactExtract:
    """Tests for artifact extract command."""

    def test_extract_success(self, db_with_zip_artifact: tuple[Path, ULID], tmp_path: Path) -> None:
        """Test successful artifact extraction."""
        db_path, artifact_id = db_with_zip_artifact
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(artifact_id),
                "--database",
                str(db_path),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Success" in result.output
        assert (output_dir / "test.txt").exists()
        assert (output_dir / "subdir" / "nested.txt").exists()
        assert (output_dir / "test.txt").read_text() == "Hello, World!"

    def test_extract_artifact_not_found(self, empty_db: Path, tmp_path: Path) -> None:
        """Test extraction of non-existent artifact."""
        output_dir = tmp_path / "output"
        fake_id = ULID()

        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(fake_id),
                "--database",
                str(empty_db),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_extract_invalid_artifact_id(self) -> None:
        """Test extraction with invalid artifact ID."""
        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                "not-a-valid-ulid",
                "--database",
                "/some/path.db",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid artifact ID" in result.output

    def test_extract_database_not_found(self) -> None:
        """Test extraction with missing database file."""
        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(ULID()),
                "--database",
                "/nonexistent/path.db",
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_extract_non_zip_artifact(self, db_with_non_zip_artifact: tuple[Path, ULID], tmp_path: Path) -> None:
        """Test extraction of non-ZIP artifact fails gracefully."""
        db_path, artifact_id = db_with_non_zip_artifact
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(artifact_id),
                "--database",
                str(db_path),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "not a ZIP file" in result.output

    def test_extract_output_exists_without_force(
        self, db_with_zip_artifact: tuple[Path, ULID], tmp_path: Path
    ) -> None:
        """Test extraction fails if output directory exists without --force."""
        db_path, artifact_id = db_with_zip_artifact
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(artifact_id),
                "--database",
                str(db_path),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_extract_output_exists_with_force(
        self, db_with_zip_artifact: tuple[Path, ULID], tmp_path: Path
    ) -> None:
        """Test extraction succeeds if output directory exists with --force."""
        db_path, artifact_id = db_with_zip_artifact
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(artifact_id),
                "--database",
                str(db_path),
                "--output",
                str(output_dir),
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert "Success" in result.output

    def test_extract_missing_source(self) -> None:
        """Test extract command requires either --database or --url."""
        result = runner.invoke(app, ["artifact", "extract", str(ULID())])
        assert result.exit_code == 1
        assert "Must provide either --database or --url" in result.output

    def test_extract_both_sources(self, tmp_path: Path) -> None:
        """Test extract command rejects both --database and --url."""
        result = runner.invoke(
            app,
            [
                "artifact",
                "extract",
                str(ULID()),
                "--database",
                str(tmp_path / "db.db"),
                "--url",
                "http://localhost",
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


class TestArtifactHelp:
    """Tests for artifact command help."""

    def test_artifact_help(self) -> None:
        """Test artifact command shows help."""
        result = runner.invoke(app, ["artifact", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "extract" in result.output

    def test_artifact_list_help(self) -> None:
        """Test artifact list command shows help."""
        result = runner.invoke(app, ["artifact", "list", "--help"])
        assert result.exit_code == 0
        assert "--database" in result.output
        assert "--url" in result.output
        assert "--type" in result.output

    def test_artifact_extract_help(self) -> None:
        """Test artifact extract command shows help."""
        result = runner.invoke(app, ["artifact", "extract", "--help"])
        assert result.exit_code == 0
        assert "--database" in result.output
        assert "--url" in result.output
        assert "--output" in result.output
        assert "--force" in result.output
