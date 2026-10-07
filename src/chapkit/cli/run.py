"""Run an MLproject directory as a chapkit service."""

from __future__ import annotations

import errno
import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from chapkit.cli.migrate import _TIDYVERSE_HINTS, _any_r_script_uses
from chapkit.cli.mlproject import (
    MLProject,
    MLProjectError,
    build_config_schema,
    build_ml_service_info,
    find_mlproject,
    parse_mlproject,
    resolve_mlproject,
    translate_command,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from chapkit.api import MLServiceInfo


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
    `chapkit-py` / `chapkit-r` / `chapkit-r-tidyverse` / `chapkit-r-inla`
    suffix so we can print a ready-made `docker run` one-liner. R + INLA
    detection mirrors migrate: `library(INLA)` / `library(fmesher)` in any
    root-level R script, or a `docker_r_inla` image in the MLproject's
    docker_env. Tidyverse detection reuses migrate's _TIDYVERSE_HINTS list.
    """
    has_r = any(project_dir.glob("*.r")) or any(project_dir.glob("*.R"))
    has_py = any(project_dir.glob("*.py"))
    # Mixed R + Python at the project root: only chapkit-r-inla bundles both
    # runtimes. Mirrors migrate.detect_base_image's mixed-language branch so the
    # `docker run` hint matches what `chapkit mlproject migrate` would build.
    if has_r and has_py:
        return "chapkit-r-inla"
    docker_env_image = mlproject.env_hints.get("docker_env", "")
    # Match migrate.detect_base_image's substring check so the hint and the
    # actual migrated image agree on inputs like `docker_r_inla:master` or any
    # other registry path that still mentions docker_r_inla.
    uses_inla = "docker_r_inla" in docker_env_image
    if not uses_inla and has_r:
        uses_inla = _any_r_script_uses(project_dir, ("INLA", "fmesher", "inla"))
    if has_r and uses_inla:
        return "chapkit-r-inla"
    if has_r and _any_r_script_uses(project_dir, _TIDYVERSE_HINTS):
        return "chapkit-r-tidyverse"
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


def _check_port_available(host: str, port: int) -> str | None:
    """Return why host:port cannot be bound, or None when it is free."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror as error:
        return f"Cannot resolve host {host!r}: {error.strerror}"
    family, socktype, proto, _, address = infos[0]
    with socket.socket(family, socktype, proto) as probe:
        # Match uvicorn, which sets SO_REUSEADDR: a port in TIME_WAIT is free, one with a listener is not.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(address)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return f"Port {port} on {host} is already in use. Stop the other process or pick another with --port."
            return f"Cannot bind to {host}:{port}: {error.strerror}"
    return None


def _print_contract(info: MLServiceInfo) -> None:
    """Print the service contract chap-core will read from /api/v1/info."""
    metadata = info.model_metadata
    horizon = f"{info.min_prediction_periods}-{info.max_prediction_periods}"
    typer.echo(f"  period:  {info.period_type.value} (horizon {horizon} periods)")
    if info.required_covariates:
        typer.echo(f"  covariates: {', '.join(info.required_covariates)}")
    if metadata.author or metadata.organization:
        typer.echo(f"  author:  {', '.join(part for part in (metadata.author, metadata.organization) if part)}")
    if metadata.author_assessed_status is not None:
        typer.echo(f"  status:  {metadata.author_assessed_status.value}")


def _print_warnings(issues: list[str]) -> None:
    """Print MLproject values that were ignored or kept as written."""
    for issue in issues:
        typer.echo(f"WARNING: {issue}", err=True)


def build_mlproject_app(
    project_dir: Path,
    overrides: dict[str, str] | None = None,
    issues: list[str] | None = None,
    mlproject_file: Path | None = None,
) -> tuple[FastAPI, MLProject, MLServiceInfo]:
    """Build the chapkit service for an MLproject directory without starting a server.

    Must be called with the project directory as the working directory: ShellModelRunner
    copies the current directory into each train/predict workspace.
    """
    # Lazy imports: avoid a circular import triggered by chapkit.__init__ loading the CLI.
    from chapkit import BaseConfig
    from chapkit.api import MLServiceBuilder
    from chapkit.artifact import ArtifactHierarchy
    from chapkit.ml import ShellModelRunner

    mlproject = parse_mlproject(mlproject_file or find_mlproject(project_dir))
    if issues is not None:
        issues.extend(mlproject.parse_warnings)
    train_command = translate_command(mlproject.entry_points["train"].command, overrides)
    predict_command = translate_command(mlproject.entry_points["predict"].command, overrides)

    config_schema = build_config_schema(mlproject, issues)
    info = build_ml_service_info(mlproject, issues)
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
    return app, mlproject, info


def run_command(
    path: Annotated[
        Path,
        typer.Argument(
            help=(
                "MLproject file, or the directory containing it (default: current directory). "
                "Nothing is written to it: the service keeps its state in memory."
            ),
        ),
    ] = Path("."),
    host: Annotated[
        str,
        typer.Option(help="Host interface to bind to."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help="Port to listen on.", min=1, max=65535),
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
    """Run an MLproject as a chapkit service."""
    overrides = _parse_param_overrides(param)

    # Fail before any parsing output when the service could never start.
    port_problem = _check_port_available(host, port)
    if port_problem is not None:
        typer.echo(f"Error: {port_problem}", err=True)
        raise typer.Exit(code=1)

    try:
        mlproject_file = resolve_mlproject(path.resolve())
        project_dir = mlproject_file.parent
        mlproject = parse_mlproject(mlproject_file)
        train_command = translate_command(mlproject.entry_points["train"].command, overrides)
        predict_command = translate_command(mlproject.entry_points["predict"].command, overrides)
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    _warn_about_env(mlproject)

    os.chdir(project_dir)
    issues: list[str] = []
    try:
        app, mlproject, info = build_mlproject_app(project_dir, overrides, issues, mlproject_file)
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Starting chapkit service for MLproject '{mlproject.name}'")
    typer.echo(f"  source:  {mlproject_file}")
    typer.echo(f"  train:   {train_command}")
    typer.echo(f"  predict: {predict_command}")
    _print_contract(info)
    _print_warnings(issues)

    _print_docker_hint(project_dir, mlproject, port)

    from chapkit.api import run_app

    run_app(app, host=host, port=port)
