"""Artifact subcommands for chapkit CLI."""

import asyncio
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

import httpx
import typer
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit.artifact import ArtifactRepository

artifact_app = typer.Typer(
    name="artifact",
    help="Artifact management commands",
    no_args_is_help=True,
)


async def _fetch_from_database(
    database_path: Path,
    artifact_id: ULID,
) -> tuple[bool, str, bytes | None]:
    """Fetch artifact content from local database."""
    db = SqliteDatabaseBuilder.from_file(str(database_path)).with_migrations(enabled=False).build()
    await db.init()

    try:
        async with db.session() as session:
            repository = ArtifactRepository(session)
            artifact = await repository.find_by_id(artifact_id)

            if artifact is None:
                return False, f"Artifact {artifact_id} not found", None

            data = artifact.data
            if not isinstance(data, dict):
                return False, f"Artifact {artifact_id} has invalid data format", None

            content = data.get("content")
            content_type = data.get("content_type")

            if content is None:
                return False, f"Artifact {artifact_id} has no content", None

            if content_type != "application/zip":
                return (
                    False,
                    f"Artifact {artifact_id} is not a ZIP file (content_type: {content_type or 'unknown'})",
                    None,
                )

            if not isinstance(content, bytes):
                return False, f"Artifact {artifact_id} content is not bytes", None

            return True, "", content
    finally:
        await db.dispose()


def _fetch_from_url(
    base_url: str,
    artifact_id: ULID,
) -> tuple[bool, str, bytes | None]:
    """Fetch artifact content from remote service via HTTP."""
    download_url = f"{base_url.rstrip('/')}/api/v1/artifacts/{artifact_id}/$download"

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(download_url)

            if response.status_code == 404:
                return False, f"Artifact {artifact_id} not found", None

            if response.status_code != 200:
                return False, f"HTTP error {response.status_code}: {response.text[:200]}", None

            content_type = response.headers.get("content-type", "")
            if "application/zip" not in content_type:
                return (
                    False,
                    f"Artifact {artifact_id} is not a ZIP file (content_type: {content_type or 'unknown'})",
                    None,
                )

            return True, "", response.content

    except httpx.ConnectError:
        return False, f"Connection error: Could not connect to {base_url}", None
    except httpx.TimeoutException:
        return False, f"Timeout error: Request to {base_url} timed out", None
    except httpx.HTTPError as e:
        return False, f"HTTP error: {e}", None


def _save_zip_file(content: bytes, output_file: Path) -> tuple[bool, str]:
    """Save ZIP content to a file."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(content)
        size = len(content)
        return True, f"Saved {_format_size(size)} to {output_file}"
    except OSError as e:
        return False, f"Failed to write file: {e}"


def _extract_zip_content(content: bytes, output_dir: Path) -> tuple[bool, str]:
    """Extract ZIP content to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_buffer = BytesIO(content)

    try:
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                return False, f"Corrupted file in ZIP: {bad_file}"

            zf.extractall(output_dir)
            file_count = len(zf.namelist())

        return True, f"Extracted {file_count} files to {output_dir}"

    except zipfile.BadZipFile as e:
        return False, f"Invalid ZIP file: {e}"


def _format_size(size: int | None) -> str:
    """Format byte size to human-readable string."""
    if size is None:
        return "-"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


def _format_timestamp(timestamp: str | None) -> str:
    """Format ISO timestamp to short display format."""
    if timestamp is None or timestamp == "-":
        return "-"
    try:
        # Parse ISO format and format as YYYY-MM-DD HH:MM
        from datetime import datetime

        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return "-"


async def _list_from_database(database_path: Path) -> tuple[bool, str, list[dict]]:
    """List artifacts from local database."""
    db = SqliteDatabaseBuilder.from_file(str(database_path)).with_migrations(enabled=False).build()
    await db.init()

    try:
        async with db.session() as session:
            repository = ArtifactRepository(session)
            artifacts = await repository.find_all()

            result = []
            for artifact in artifacts:
                data = artifact.data if isinstance(artifact.data, dict) else {}
                metadata = data.get("metadata", {})
                config_id = metadata.get("config_id") if isinstance(metadata, dict) else None
                result.append(
                    {
                        "id": str(artifact.id),
                        "type": data.get("type", "-"),
                        "content_type": data.get("content_type", "-"),
                        "size": data.get("content_size"),
                        "level": artifact.level,
                        "parent_id": str(artifact.parent_id) if artifact.parent_id else None,
                        "created_at": artifact.created_at.isoformat() if artifact.created_at else "-",
                        "config_id": config_id,
                    }
                )

            return True, "", result
    finally:
        await db.dispose()


def _list_from_url(base_url: str) -> tuple[bool, str, list[dict]]:
    """List artifacts from remote service via HTTP."""
    list_url = f"{base_url.rstrip('/')}/api/v1/artifacts"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(list_url)

            if response.status_code != 200:
                return False, f"HTTP error {response.status_code}: {response.text[:200]}", []

            artifacts = response.json()
            result = []
            for artifact in artifacts:
                data = artifact.get("data", {}) if isinstance(artifact.get("data"), dict) else {}
                metadata = data.get("metadata", {})
                config_id = metadata.get("config_id") if isinstance(metadata, dict) else None
                result.append(
                    {
                        "id": artifact.get("id", "-"),
                        "type": data.get("type", "-"),
                        "content_type": data.get("content_type", "-"),
                        "size": data.get("content_size"),
                        "level": artifact.get("level", 0),
                        "parent_id": artifact.get("parent_id"),
                        "created_at": artifact.get("created_at", "-"),
                        "config_id": config_id,
                    }
                )

            return True, "", result

    except httpx.ConnectError:
        return False, f"Connection error: Could not connect to {base_url}", []
    except httpx.TimeoutException:
        return False, f"Timeout error: Request to {base_url} timed out", []
    except httpx.HTTPError as e:
        return False, f"HTTP error: {e}", []


def list_command(
    database: Annotated[
        Path | None,
        typer.Option(
            "--database",
            "-d",
            help="Path to SQLite database file",
        ),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            "-u",
            help="Base URL of running chapkit service (e.g., http://localhost:8000)",
        ),
    ] = None,
    artifact_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by artifact type (e.g., ml_training_workspace, ml_prediction)",
        ),
    ] = None,
) -> None:
    """List all artifacts with their type and metadata."""
    # Validate mutual exclusivity
    if database is None and url is None:
        typer.echo(
            "Error: Must provide either --database or --url",
            err=True,
        )
        raise typer.Exit(code=1)

    if database is not None and url is not None:
        typer.echo(
            "Error: Cannot use both --database and --url (mutually exclusive)",
            err=True,
        )
        raise typer.Exit(code=1)

    # Validate database exists (if local mode)
    if database is not None and not database.exists():
        typer.echo(f"Error: Database file not found: {database}", err=True)
        raise typer.Exit(code=1)

    # Fetch artifacts
    if database is not None:
        success, error, artifacts = asyncio.run(_list_from_database(database))
    else:
        assert url is not None  # Guaranteed by mutual exclusivity check above
        success, error, artifacts = _list_from_url(url)

    if not success:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1)

    # Filter by type if specified
    if artifact_type:
        artifacts = [a for a in artifacts if a["type"] == artifact_type]

    if not artifacts:
        typer.echo("No artifacts found")
        return

    # Print header
    typer.echo(f"{'ID':<32} {'TYPE':<25} {'SIZE':<10} {'CONFIG':<28} {'CREATED'}")
    typer.echo("-" * 115)

    # Print artifacts with hierarchy indentation
    for artifact in artifacts:
        level = artifact["level"]
        indent = "  " * level  # 2 spaces per level

        # Full ULID with indentation (no truncation - ULIDs are always 26 chars)
        artifact_id = indent + artifact["id"]

        # Truncate type if needed
        artifact_type_str = artifact["type"]
        if len(artifact_type_str) > 23:
            artifact_type_str = artifact_type_str[:21] + ".."

        size = _format_size(artifact["size"])

        # Format config_id (full ULID, no truncation)
        config_id = artifact.get("config_id") or "-"

        # Format timestamp
        created = _format_timestamp(artifact.get("created_at"))

        typer.echo(f"{artifact_id:<32} {artifact_type_str:<25} {size:<10} {config_id:<28} {created}")


def download_command(
    artifact_id: Annotated[
        str,
        typer.Argument(help="Artifact ID (ULID) to download"),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path (default: ./<artifact_id>.zip or ./<artifact_id>/ with --extract)",
        ),
    ] = None,
    database: Annotated[
        Path | None,
        typer.Option(
            "--database",
            "-d",
            help="Path to SQLite database file",
        ),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            "-u",
            help="Base URL of running chapkit service (e.g., http://localhost:8000)",
        ),
    ] = None,
    extract: Annotated[
        bool,
        typer.Option(
            "--extract",
            "-x",
            help="Extract ZIP contents to a directory instead of saving as file",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing output file or directory",
        ),
    ] = False,
) -> None:
    """Download a ZIP artifact (optionally extract it)."""
    # Validate mutual exclusivity
    if database is None and url is None:
        typer.echo(
            "Error: Must provide either --database or --url",
            err=True,
        )
        raise typer.Exit(code=1)

    if database is not None and url is not None:
        typer.echo(
            "Error: Cannot use both --database and --url (mutually exclusive)",
            err=True,
        )
        raise typer.Exit(code=1)

    # Parse artifact ID
    try:
        ulid = ULID.from_str(artifact_id)
    except ValueError as e:
        typer.echo(f"Error: Invalid artifact ID '{artifact_id}': {e}", err=True)
        raise typer.Exit(code=1)

    # Validate database exists (if local mode)
    if database is not None and not database.exists():
        typer.echo(f"Error: Database file not found: {database}", err=True)
        raise typer.Exit(code=1)

    # Set default output path based on mode
    if extract:
        output_path = output or Path(artifact_id)
    else:
        output_path = output or Path(f"{artifact_id}.zip")

    # Check if output exists
    if output_path.exists():
        if not force:
            typer.echo(
                f"Error: Output already exists: {output_path}\nUse --force to overwrite",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Warning: Overwriting existing: {output_path}")

    typer.echo(f"Downloading artifact {artifact_id}...")

    if database is not None:
        typer.echo(f"Database: {database.absolute()}")
    else:
        typer.echo(f"URL: {url}")

    typer.echo(f"Output: {output_path.absolute()}")
    typer.echo()

    # Fetch content
    if database is not None:
        success, error, content = asyncio.run(_fetch_from_database(database, ulid))
    else:
        assert url is not None  # Guaranteed by mutual exclusivity check above
        success, error, content = _fetch_from_url(url, ulid)

    if not success:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1)

    assert content is not None  # Guaranteed by success=True

    # Save or extract
    if extract:
        success, message = _extract_zip_content(content, output_path)
    else:
        success, message = _save_zip_file(content, output_path)

    if success:
        typer.echo(f"Success: {message}")
    else:
        typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(code=1)


# Register commands
artifact_app.command(name="list", help="List all artifacts with their type and metadata")(list_command)
artifact_app.command(name="download", help="Download a ZIP artifact (optionally extract it)")(download_command)
