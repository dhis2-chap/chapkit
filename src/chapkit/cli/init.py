"""Init subcommand for chapkit CLI."""

import re
import shutil
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Annotated

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


def _to_service_id(text: str) -> str:
    """Convert text to a valid ServiceInfo id (lowercase letters, numbers, hyphens)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
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
        Path | None,
        typer.Option(help="Target directory (default: current directory)"),
    ] = None,
    with_validation: Annotated[
        bool,
        typer.Option(
            help=(
                "Include on_validate_train/on_validate_predict stubs so the $validate endpoint "
                "can emit domain-specific diagnostics (e.g. n_lags exceeds context). Off by default."
            ),
        ),
    ] = False,
    template: Annotated[
        str,
        typer.Option(
            help=(
                "Template type. "
                "fn-py: Python ML in main.py (chapkit-py). "
                "shell-py: Python train/predict scripts in scripts/ (chapkit-py). "
                "shell-r: R train/predict scripts in scripts/ on plain chapkit-r. "
                "shell-r-tidyverse: R scripts on chapkit-r-tidyverse (tidyverse + fable + forecasting/ML stack). "
                "shell-r-inla: R scripts on chapkit-r-inla (INLA + spatial stack, amd64-only)."
            ),
        ),
    ] = "fn-py",
) -> None:
    """Initialize a new chapkit ML service project."""
    target_dir = (path or Path.cwd()) / project_name

    if target_dir.exists():
        typer.echo(f"Error: Directory '{target_dir}' already exists", err=True)
        raise typer.Exit(code=1)

    valid_templates = ["fn-py", "shell-py", "shell-r", "shell-r-tidyverse", "shell-r-inla"]
    if template not in valid_templates:
        typer.echo(
            f"Error: Invalid template '{template}'. Must be one of: {', '.join(valid_templates)}",
            err=True,
        )
        raise typer.Exit(code=1)

    project_slug = _slugify(project_name)

    typer.echo(f"Creating new chapkit project: {project_name}")
    typer.echo(f"Project directory: {target_dir}")
    typer.echo(f"Package name: {project_slug}")
    typer.echo(f"Template: {template}")
    if with_validation:
        typer.echo("Including $validate hook stubs (on_validate_train / on_validate_predict)")
    typer.echo()

    template_dir = Path(__file__).parent / "templates"

    if not template_dir.exists():
        typer.echo(f"Error: Template directory not found at {template_dir}", err=True)
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True)

    service_id = _to_service_id(project_name)

    context = {
        "PROJECT_NAME": project_name,
        "PROJECT_SLUG": project_slug,
        "PROJECT_SERVICE_ID": service_id,
        "PROJECT_DESCRIPTION": f"ML service for {project_name}",
        "WITH_VALIDATION": with_validation,
        "TEMPLATE": template,
        "CHAPKIT_VERSION": _get_chapkit_version(),
    }

    typer.echo("Generating project files...")

    is_shell = template in ("shell-py", "shell-r", "shell-r-tidyverse", "shell-r-inla")

    # Render main.py based on template type
    if is_shell:
        _render_template(template_dir, target_dir, "main_shell.py.jinja2", context, "main.py")
    else:
        _render_template(template_dir, target_dir, "main.py.jinja2", context, "main.py")

    _render_template(template_dir, target_dir, "pyproject.toml.jinja2", context, "pyproject.toml")
    _render_template(template_dir, target_dir, "Dockerfile.jinja2", context, "Dockerfile")
    _render_template(template_dir, target_dir, "README.md.jinja2", context, "README.md")
    _render_template(template_dir, target_dir, "compose.yml.jinja2", context, "compose.yml")

    # For shell templates, create scripts directory with language-specific stubs
    if template == "shell-py":
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
    elif template in ("shell-r", "shell-r-tidyverse", "shell-r-inla"):
        scripts_dir = target_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        _render_template(
            template_dir / "scripts",
            scripts_dir,
            "train.R.jinja2",
            context,
            "train.R",
        )
        _render_template(
            template_dir / "scripts",
            scripts_dir,
            "predict.R.jinja2",
            context,
            "predict.R",
        )

    _copy_static_file(template_dir, target_dir, ".gitignore")
    _copy_static_file(template_dir, target_dir, ".python-version")

    typer.echo()
    typer.echo(f"Successfully created project '{project_name}' at {target_dir}")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo(f"  cd {project_name}")
    typer.echo("  uv sync")
    typer.echo("  uv run python main.py")
    typer.echo()
    typer.echo("To start with Docker:")
    typer.echo("  docker compose up --build")
    typer.echo()
