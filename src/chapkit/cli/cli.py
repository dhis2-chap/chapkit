"""Main CLI application for chapkit."""

import typer

from chapkit import __version__
from chapkit.cli.init import init_command

# Get servicekit version
try:
    from importlib.metadata import version as _get_version

    _servicekit_version = _get_version("servicekit")
except Exception:
    _servicekit_version = "unknown"

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
app.command(name="init", help="Initialize a new chapkit ML service project")(init_command)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
