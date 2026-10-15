"""Run an MLproject directory as a chapkit service."""

from __future__ import annotations

import errno
import fnmatch
import os
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from chapkit.cli.mlproject import (
    MLProject,
    MLProjectError,
    _python_identifier,
    build_config_schema,
    build_ml_service_info,
    mlproject_config_format,
    parse_mlproject,
    resolve_mlproject,
    translate_command,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from chapkit.api import MLServiceInfo


# servicekit's registration settings; registration only happens when the orchestrator URL is set.
ORCHESTRATOR_URL_ENV = "SERVICEKIT_ORCHESTRATOR_URL"
ADVERTISED_PORT_ENV = "SERVICEKIT_PORT"
IN_MEMORY_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
MLPROJECT_LEVEL_LABELS = {0: "ml_training_workspace", 1: "ml_prediction"}


class ConfigFormat(StrEnum):
    """Layout of the config.yml written into each train/predict workspace."""

    chap_core = "chap_core"
    flat = "flat"


class LogLevel(StrEnum):
    """Log levels accepted by --log-level."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


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


def resolve_config_format(
    mlproject: MLProject, override: ConfigFormat | None = None, issues: list[str] | None = None
) -> ConfigFormat:
    """Return the config.yml layout: the --config-format override, else meta_data.config_format, else chap_core."""
    if override is not None:
        return override
    return ConfigFormat(mlproject_config_format(mlproject, issues) or ConfigFormat.chap_core)


def copied_into_workspaces(database_path: Path, project_dir: Path) -> bool:
    """Return whether a database file inside the project would be copied into every train/predict workspace."""
    from chapkit.ml.runner import WORKSPACE_EXCLUDE_PATTERNS

    try:
        relative = database_path.relative_to(project_dir)
    except ValueError:
        return False
    # The copy skips any file or directory whose name matches an exclude pattern, at every level.
    return not any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in WORKSPACE_EXCLUDE_PATTERNS)


def resolve_database_url(database: Path | None) -> str:
    """Return the SQLite URL for --database, creating its parent directory, or an in-memory URL without it.

    Relative paths resolve against the current directory, so call this before changing into the project.
    """
    if database is None:
        return IN_MEMORY_DATABASE_URL
    path = database.expanduser().resolve()
    if path.is_dir():
        raise MLProjectError(f"--database must be a file, not a directory: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MLProjectError(f"Cannot create the directory for --database {path}: {error.strerror}") from error
    return f"sqlite+aiosqlite:///{path}"


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


@dataclass(frozen=True, slots=True)
class Registration:
    """Where and how an mlproject service registers itself with an orchestrator such as chap-core."""

    orchestrator_url: str
    advertised_host: str | None
    advertised_port: int | None
    local_port: int | None
    keepalive_interval: float | None = None


def resolve_registration(
    port: int | None,
    register_url: str | None = None,
    advertise_host: str | None = None,
    advertise_port: int | None = None,
    keepalive_interval: float | None = None,
) -> Registration | None:
    """Resolve registration settings from CLI options with SERVICEKIT_* fallbacks, or None when not configured.

    Each option falls back to its environment variable (SERVICEKIT_ORCHESTRATOR_URL,
    SERVICEKIT_HOST, SERVICEKIT_PORT). Without an advertised port servicekit would
    advertise 8000, so the port the service listens on is advertised instead; that port
    is also probed for readiness before registering. Without a keepalive interval
    servicekit's default applies.
    """
    orchestrator_url = register_url or os.getenv(ORCHESTRATOR_URL_ENV)
    if not orchestrator_url:
        return None
    if advertise_port is None and not os.getenv(ADVERTISED_PORT_ENV):
        advertise_port = port
    # None host / port lets servicekit read SERVICEKIT_HOST / SERVICEKIT_PORT (or the hostname).
    return Registration(orchestrator_url, advertise_host, advertise_port, port, keepalive_interval)


def build_mlproject_app(
    mlproject: MLProject,
    overrides: dict[str, str] | None = None,
    issues: list[str] | None = None,
    port: int | None = None,
    registration: Registration | None = None,
    config_format: ConfigFormat | None = None,
    database_url: str = IN_MEMORY_DATABASE_URL,
) -> tuple[FastAPI, MLServiceInfo]:
    """Build the chapkit service for a parsed MLproject without starting a server.

    With `registration` the service registers itself with that orchestrator (chap-core)
    on startup; by default it is resolved from the SERVICEKIT_* environment variables.
    State lives in memory unless `database_url` points at a SQLite file. Without `config_format`
    the MLproject's meta_data.config_format applies, else chap_core.

    Must be called with the project directory as the working directory: ShellModelRunner
    copies the current directory into each train/predict workspace.
    """
    # Lazy imports: avoid a circular import triggered by chapkit.__init__ loading the CLI.
    from chapkit import BaseConfig
    from chapkit.api import MLServiceBuilder
    from chapkit.artifact import ArtifactHierarchy
    from chapkit.ml import ShellModelRunner

    if issues is not None:
        issues.extend(mlproject.parse_warnings)
    train_command = translate_command(mlproject.entry_points["train"].command, overrides)
    predict_command = translate_command(mlproject.entry_points["predict"].command, overrides)

    config_schema = build_config_schema(mlproject, issues)
    info = build_ml_service_info(mlproject, issues)
    config_format = resolve_config_format(mlproject, config_format, issues)
    # The chap_core default matches `chapkit mlproject migrate`: config.yml in chap-core's
    # ModelConfiguration shape (reserved keys at top level, everything else nested under
    # user_option_values), so scripts written for the chap-models ecosystem behave the same
    # regardless of which CLI spawned the service. flat suits scripts written for a main.py.
    runner: ShellModelRunner[BaseConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command=predict_command,
        config_format=config_format.value,
        # chap-core applies MLproject adapters itself only for models it runs directly,
        # not for chapkit services, so the runner adds the adapted columns.
        adapters=mlproject.adapters,
    )
    # Named after the MLproject, like the hierarchies `chapkit mlproject migrate` and `chapkit init` generate.
    hierarchy = ArtifactHierarchy(name=_python_identifier(mlproject.name), level_labels=MLPROJECT_LEVEL_LABELS)

    builder = MLServiceBuilder(
        info=info,
        config_schema=config_schema,
        hierarchy=hierarchy,
        runner=runner,
        database_url=database_url,
    )
    if registration is None:
        registration = resolve_registration(port)
    if registration is not None:
        # Leave keepalive_interval to servicekit's default unless one was given.
        keepalive: dict[str, Any] = (
            {} if registration.keepalive_interval is None else {"keepalive_interval": registration.keepalive_interval}
        )
        builder = builder.with_registration(
            orchestrator_url=registration.orchestrator_url,
            host=registration.advertised_host,
            port=registration.advertised_port,
            local_port=registration.local_port,
            **keepalive,
        )
    return builder.build(), info


def run_command(
    path: Annotated[
        Path,
        typer.Argument(
            help=(
                "MLproject file, or the directory containing it (default: current directory). "
                "Nothing is written to it: the service keeps its state in memory unless --database is given."
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
    register_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "Register with this orchestrator on startup, e.g. http://chap:8000/v2/services/$register. "
                "Defaults to SERVICEKIT_ORCHESTRATOR_URL; without either, nothing is registered. "
                "A shared secret is read from SERVICEKIT_REGISTRATION_KEY."
            ),
        ),
    ] = None,
    advertise_host: Annotated[
        str | None,
        typer.Option(help="Host the orchestrator should call. Defaults to SERVICEKIT_HOST, then the hostname."),
    ] = None,
    advertise_port: Annotated[
        int | None,
        typer.Option(
            help="Port the orchestrator should call. Defaults to SERVICEKIT_PORT, then --port.", min=1, max=65535
        ),
    ] = None,
    keepalive_interval: Annotated[
        float | None,
        typer.Option(
            help="Seconds between keepalive pings to the orchestrator once registered (default: 10).",
            min=1.0,
        ),
    ] = None,
    config_format: Annotated[
        ConfigFormat | None,
        typer.Option(
            help=(
                "Layout of the config.yml your scripts read: chap_core nests the user options under "
                "user_option_values, as chap-core does; flat puts every option at the top level. "
                "Defaults to the MLproject's meta_data.config_format, then chap_core."
            ),
            case_sensitive=False,
            show_default=False,
        ),
    ] = None,
    database: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Keep configs and trained models in this SQLite file so they survive a restart "
                "(job history stays in memory). "
                "Created if missing. Without it, everything is kept in memory."
            ),
            dir_okay=False,
        ),
    ] = None,
    log_level: Annotated[
        LogLevel | None,
        typer.Option(help="Log level. Defaults to LOG_LEVEL, then info.", case_sensitive=False),
    ] = None,
) -> None:
    """Run an MLproject as a chapkit service."""
    overrides = _parse_param_overrides(param)

    try:
        mlproject_file = resolve_mlproject(path.resolve())
        project_dir = mlproject_file.parent
        mlproject = parse_mlproject(mlproject_file)
        train_command = translate_command(mlproject.entry_points["train"].command, overrides)
        predict_command = translate_command(mlproject.entry_points["predict"].command, overrides)
        # Resolved before changing into the project, so a relative path is relative to where you ran the command.
        database_url = resolve_database_url(database)
        storage = str(database.expanduser().resolve()) if database is not None else "in memory"
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if log_level is not None:
        # servicekit's structured logging reads LOG_LEVEL; uvicorn gets the same level from run_app.
        os.environ["LOG_LEVEL"] = log_level.value.upper()
    from servicekit.logging import configure_logging

    # Configured now, not only when run_app starts, so the level also covers what is logged while building.
    configure_logging()

    # Fail before building the service when it could never start listening.
    port_problem = _check_port_available(host, port)
    if port_problem is not None:
        typer.echo(f"Error: {port_problem}", err=True)
        raise typer.Exit(code=1)

    _warn_about_env(mlproject)

    os.chdir(project_dir)
    issues: list[str] = []
    if database is not None and copied_into_workspaces(Path(storage), project_dir):
        issues.append(
            f"--database {storage} is inside the project and is copied into every train/predict workspace; "
            "give it a .db extension or keep it outside the project"
        )
    resolved_config_format = resolve_config_format(mlproject, config_format, issues)
    try:
        registration = resolve_registration(port, register_url, advertise_host, advertise_port, keepalive_interval)
        app, info = build_mlproject_app(
            mlproject,
            overrides,
            issues,
            port=port,
            registration=registration,
            config_format=resolved_config_format,
            database_url=database_url,
        )
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Starting chapkit service for MLproject '{mlproject.name}'")
    typer.echo(f"  source:  {mlproject_file}")
    typer.echo(f"  train:   {train_command}")
    typer.echo(f"  predict: {predict_command}")
    typer.echo(f"  config:  config.yml ({resolved_config_format.value} layout)")
    typer.echo(f"  storage: {storage}")
    _print_contract(info)
    if registration is not None:
        typer.echo(f"  registers with: {registration.orchestrator_url}")
    elif keepalive_interval is not None:
        issues.append("--keepalive-interval has no effect without --register-url or SERVICEKIT_ORCHESTRATOR_URL")
    _print_warnings(issues)

    from chapkit.api import run_app

    run_app(app, host=host, port=port, log_level=log_level.value if log_level is not None else None)
