"""Run an MLproject directory as a chapkit service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from chapkit.cli.mlproject import (
    MLProject,
    MLProjectError,
    build_config_schema,
    find_mlproject,
    parse_mlproject,
    slugify,
    translate_command,
)


def _parse_param_overrides(raw: list[str] | None) -> dict[str, str]:
    """Parse repeated --param NAME=FILENAME flags into a mapping."""
    overrides: dict[str, str] = {}
    if not raw:
        return overrides
    for entry in raw:
        if "=" not in entry:
            raise typer.BadParameter(f"--param expects NAME=FILENAME, got: {entry}")
        name, _, value = entry.partition("=")
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise typer.BadParameter(f"--param expects NAME=FILENAME, got: {entry}")
        overrides[name] = value
    return overrides


def _warn_about_env(mlproject: MLProject) -> None:
    """Print warnings for environment fields chapkit does not auto-handle."""
    if not mlproject.env_hints:
        return
    typer.echo("", err=True)
    typer.echo("WARNING: chapkit run does not auto-activate environments.", err=True)
    for field_name, value in mlproject.env_hints.items():
        typer.echo(f"  - {field_name}: {value}", err=True)
    typer.echo(
        "Activate the right runtime (R/renv, conda, Docker image, etc.) "
        "before launching chapkit run, or invoke chapkit run from inside it.",
        err=True,
    )
    typer.echo("", err=True)


def run_command(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory containing the MLproject file (default: current directory).",
        ),
    ] = Path("."),
    host: Annotated[
        str,
        typer.Option(help="Host interface to bind to."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help="Port to listen on."),
    ] = 8000,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help=(
                "Override the filename substituted for an MLproject parameter, "
                "e.g. --param dataset=data.csv. Repeatable."
            ),
        ),
    ] = None,
) -> None:
    """Run an MLproject directory as a chapkit service."""
    project_dir = path.resolve()
    try:
        mlproject_file = find_mlproject(project_dir)
        mlproject = parse_mlproject(mlproject_file)
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    overrides = _parse_param_overrides(param)

    try:
        train_command = translate_command(mlproject.entry_points["train"].command, overrides)
        predict_command = translate_command(mlproject.entry_points["predict"].command, overrides)
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    _warn_about_env(mlproject)

    typer.echo(f"Starting chapkit service for MLproject '{mlproject.name}'")
    typer.echo(f"  source: {mlproject_file}")
    typer.echo(f"  train:   {train_command}")
    typer.echo(f"  predict: {predict_command}")

    os.chdir(project_dir)

    # Lazy imports: avoid a circular import triggered by chapkit.__init__ loading the CLI.
    from chapkit import BaseConfig
    from chapkit.api import MLServiceBuilder, ServiceInfo, run_app
    from chapkit.artifact import ArtifactHierarchy
    from chapkit.ml import ShellModelRunner

    config_schema = build_config_schema(mlproject)
    # Match `chapkit migrate`'s default: emit config.yml in chap-core's
    # ModelConfiguration shape (reserved keys at top level, everything else
    # nested under user_option_values). Keeps runtime and code-generated
    # services consistent so scripts written for the chap-models ecosystem
    # behave the same regardless of which CLI spawned the service.
    runner: ShellModelRunner[BaseConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command=predict_command,
        config_format="chap_core",
    )

    info = ServiceInfo(
        id=slugify(mlproject.name),
        display_name=mlproject.name,
        description=f"Auto-discovered MLproject service for '{mlproject.name}'",
    )
    hierarchy = ArtifactHierarchy(
        name="mlproject",
        level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
    )

    app = MLServiceBuilder(
        info=info,
        config_schema=config_schema,
        hierarchy=hierarchy,
        runner=runner,
    ).build()

    run_app(app, host=host, port=port)
