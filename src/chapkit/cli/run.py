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
    typer.echo("WARNING: chapkit mlproject run does not auto-activate environments.", err=True)
    for field_name, value in mlproject.env_hints.items():
        typer.echo(f"  - {field_name}: {value}", err=True)
    typer.echo(
        "Activate the right runtime (R/renv, conda, Docker image, etc.) "
        "before launching chapkit mlproject run, or invoke it from inside the runtime.",
        err=True,
    )
    typer.echo("", err=True)


def _suggest_chapkit_image(project_dir: Path, mlproject: MLProject) -> str:
    """Pick the chapkit-images base image best suited to this MLproject.

    Light-touch variant of migrate's detect_base_image - just returns the
    `chapkit-py` / `chapkit-r` / `chapkit-r-inla` suffix so we can print a
    ready-made `docker run` one-liner. R + INLA detection mirrors migrate:
    `library(INLA)` / `library(fmesher)` in any root-level R script, or a
    `docker_r_inla` image in the MLproject's docker_env.
    """
    has_r = any(project_dir.glob("*.r")) or any(project_dir.glob("*.R"))
    has_py = any(project_dir.glob("*.py"))
    docker_env_image = mlproject.env_hints.get("docker_env", "")
    uses_inla = docker_env_image.startswith("ghcr.io/dhis2-chap/docker_r_inla")
    if not uses_inla and has_r:
        for script in project_dir.glob("*.[rR]"):
            try:
                text = script.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "library(INLA)" in text or "library(fmesher)" in text or 'library("INLA")' in text:
                uses_inla = True
                break
    if has_r and uses_inla:
        return "chapkit-r-inla"
    if has_r:
        return "chapkit-r"
    if has_py:
        return "chapkit-py"
    # Ambiguous (no .r/.R/.py at root); default to Python - works for MLprojects that
    # call into compiled binaries or do all work inside the entry-point commands.
    return "chapkit-py"


def _print_docker_hint(project_dir: Path, mlproject: MLProject, port: int) -> None:
    """Tell the user how to run the same MLproject via the prebuilt chapkit-images.

    Skipped when the host already looks like a chapkit container so we don't nest
    the hint inside itself.
    """
    if Path("/app/.venv/bin/chapkit").exists():
        return
    image = _suggest_chapkit_image(project_dir, mlproject)
    platform_flag = " --platform=linux/amd64" if image == "chapkit-r-inla" else ""
    typer.echo("")
    typer.echo("Tip: to run the same MLproject in Docker (no local R/Python env needed):")
    typer.echo(
        f"  docker run --rm -p {port}:8000{platform_flag} -v {project_dir}:/work ghcr.io/dhis2-chap/{image}:latest"
    )
    typer.echo("  # chapkit-images ship WORKDIR=/work + a preinstalled chapkit; model-specific R / Python")
    typer.echo("  # packages need to be installed separately (e.g. `chapkit mlproject migrate` + `docker build`).")
    typer.echo("")


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
    ] = 9090,
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

    _print_docker_hint(project_dir, mlproject, port)

    os.chdir(project_dir)

    # Lazy imports: avoid a circular import triggered by chapkit.__init__ loading the CLI.
    from chapkit import BaseConfig
    from chapkit.api import MLServiceBuilder, ServiceInfo, run_app
    from chapkit.artifact import ArtifactHierarchy
    from chapkit.ml import ShellModelRunner

    config_schema = build_config_schema(mlproject)
    # Match `chapkit mlproject migrate`'s default: emit config.yml in chap-core's
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
