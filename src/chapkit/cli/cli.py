"""Main CLI application for chapkit."""

import tomllib
from pathlib import Path

import typer

from chapkit import __version__
from chapkit.cli.artifact import artifact_app
from chapkit.cli.init import init_command

# Get servicekit version
try:
    from importlib.metadata import version as _get_version

    _servicekit_version = _get_version("servicekit")
except Exception:
    _servicekit_version = "unknown"


def _find_chapkit_project() -> Path | None:
    """Find chapkit project root by looking for pyproject.toml with chapkit dependency."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists():
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                if any("chapkit" in dep for dep in deps):
                    return parent
            except Exception:
                pass
    return None


app = typer.Typer(
    name="chapkit",
    help="Chapkit CLI for ML service management and scaffolding",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> None:
    """Chapkit CLI for ML service management and scaffolding."""
    if version:
        typer.echo(f"chapkit version {__version__}")
        typer.echo(f"servicekit version {_servicekit_version}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# Register subcommands
# Only show 'init' command when NOT inside a chapkit project
if _find_chapkit_project() is None:
    app.command(name="init", help="Initialize a new chapkit ML service project")(init_command)
app.add_typer(artifact_app, name="artifact")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
