"""Main CLI application for chapkit."""

import typer

from chapkit.cli.init import init_command

app = typer.Typer(
    name="chapkit",
    help="Chapkit CLI for ML service management and scaffolding",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    """Chapkit CLI for ML service management and scaffolding."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# Register subcommands
app.command(name="init", help="Initialize a new chapkit ML service project")(init_command)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
