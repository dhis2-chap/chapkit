"""Init subcommand for chapkit CLI."""

import re
import shutil
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Annotated, Optional

import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _get_chapkit_version() -> str:
    """Get chapkit version from package metadata."""
    try:
        return get_version("chapkit")
    except Exception:
        return "unknown"


def _slugify(text: str) -> str:
    """Convert text to a valid Python package name."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "_", text)
    return text


def _copy_static_file(template_dir: Path, target_dir: Path, filename: str) -> None:
    """Copy a static file from template directory to target directory."""
    src = template_dir / filename
    dst = target_dir / filename
    if src.exists():
        shutil.copy2(src, dst)
        typer.echo(f"  Created {filename}")


def _render_template(
    template_dir: Path, target_dir: Path, template_name: str, context: dict, output_name: str | None = None
) -> None:
    """Render a Jinja2 template and write to target directory."""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(template_name)
    content = template.render(**context)

    output_file = target_dir / (output_name or template_name.replace(".jinja2", ""))
    output_file.write_text(content)
    typer.echo(f"  Created {output_file.name}")


def init_command(
    project_name: Annotated[str, typer.Argument(help="Name of the project to create")],
    path: Annotated[
        Optional[Path],
        typer.Option(help="Target directory (default: current directory)"),
    ] = None,
    with_monitoring: Annotated[
        bool,
        typer.Option(help="Include Prometheus and Grafana monitoring stack"),
    ] = False,
    template: Annotated[
        str,
        typer.Option(help="Template type: 'ml', 'ml-shell', or 'task'"),
    ] = "ml",
) -> None:
    """Initialize a new chapkit ML service project."""
    target_dir = (path or Path.cwd()) / project_name

    if target_dir.exists():
        typer.echo(f"Error: Directory '{target_dir}' already exists", err=True)
        raise typer.Exit(code=1)

    if template not in ["ml", "ml-shell", "task"]:
        typer.echo(f"Error: Invalid template '{template}'. Must be 'ml', 'ml-shell', or 'task'", err=True)
        raise typer.Exit(code=1)

    project_slug = _slugify(project_name)

    typer.echo(f"Creating new chapkit project: {project_name}")
    typer.echo(f"Project directory: {target_dir}")
    typer.echo(f"Package name: {project_slug}")
    typer.echo(f"Template: {template}")
    if with_monitoring:
        typer.echo("Including monitoring stack (Prometheus + Grafana)")
    typer.echo()

    template_dir = Path(__file__).parent / "templates"

    if not template_dir.exists():
        typer.echo(f"Error: Template directory not found at {template_dir}", err=True)
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True)

    context = {
        "PROJECT_NAME": project_name,
        "PROJECT_SLUG": project_slug,
        "PROJECT_DESCRIPTION": f"ML service for {project_name}",
        "WITH_MONITORING": with_monitoring,
        "TEMPLATE": template,
        "CHAPKIT_VERSION": _get_chapkit_version(),
    }

    typer.echo("Generating project files...")

    # Render main.py based on template type
    if template == "ml-shell":
        _render_template(template_dir, target_dir, "main_shell.py.jinja2", context, "main.py")
    elif template == "task":
        _render_template(template_dir, target_dir, "main_task.py.jinja2", context, "main.py")
    else:
        _render_template(template_dir, target_dir, "main.py.jinja2", context, "main.py")

    _render_template(template_dir, target_dir, "pyproject.toml.jinja2", context, "pyproject.toml")
    _render_template(template_dir, target_dir, "Dockerfile.jinja2", context, "Dockerfile")
    _render_template(template_dir, target_dir, "README.md.jinja2", context, "README.md")

    # Render Postman collection based on template type
    if template == "ml-shell":
        _render_template(
            template_dir, target_dir, "postman_collection_ml_shell.json.jinja2", context, "postman_collection.json"
        )
    elif template == "task":
        _render_template(
            template_dir, target_dir, "postman_collection_task.json.jinja2", context, "postman_collection.json"
        )
    else:
        _render_template(
            template_dir, target_dir, "postman_collection_ml.json.jinja2", context, "postman_collection.json"
        )

    # For ml-shell template, create scripts directory
    if template == "ml-shell":
        scripts_dir = target_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        _render_template(
            template_dir / "scripts",
            scripts_dir,
            "train_model.py.jinja2",
            context,
            "train_model.py",
        )
        _render_template(
            template_dir / "scripts",
            scripts_dir,
            "predict_model.py.jinja2",
            context,
            "predict_model.py",
        )

    if with_monitoring:
        _render_template(template_dir, target_dir, "compose.monitoring.yml.jinja2", context, "compose.yml")

        monitoring_dir = target_dir / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)

        prom_dir = monitoring_dir / "prometheus"
        prom_dir.mkdir(parents=True, exist_ok=True)
        _render_template(
            template_dir / "monitoring" / "prometheus",
            prom_dir,
            "prometheus.yml.jinja2",
            context,
            "prometheus.yml",
        )

        grafana_dir = monitoring_dir / "grafana"
        (grafana_dir / "provisioning" / "datasources").mkdir(parents=True, exist_ok=True)
        (grafana_dir / "provisioning" / "dashboards").mkdir(parents=True, exist_ok=True)
        (grafana_dir / "dashboards").mkdir(parents=True, exist_ok=True)

        _copy_static_file(
            template_dir / "monitoring" / "grafana" / "provisioning" / "datasources",
            grafana_dir / "provisioning" / "datasources",
            "prometheus.yml",
        )
        _copy_static_file(
            template_dir / "monitoring" / "grafana" / "provisioning" / "dashboards",
            grafana_dir / "provisioning" / "dashboards",
            "dashboard.yml",
        )
        _copy_static_file(
            template_dir / "monitoring" / "grafana" / "dashboards",
            grafana_dir / "dashboards",
            "chapkit-service-metrics.json",
        )
    else:
        _render_template(template_dir, target_dir, "compose.yml.jinja2", context, "compose.yml")

    _copy_static_file(template_dir, target_dir, ".gitignore")

    typer.echo()
    typer.echo(f"Successfully created project '{project_name}' at {target_dir}")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo(f"  cd {project_name}")
    typer.echo("  uv sync")
    typer.echo("  uv run python main.py")
    typer.echo()
    if with_monitoring:
        typer.echo("To start with Docker (including monitoring):")
    else:
        typer.echo("To start with Docker:")
    typer.echo("  docker compose up --build")
    typer.echo()
