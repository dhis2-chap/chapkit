"""Migrate an MLproject directory into a chapkit service project."""

from __future__ import annotations

import fnmatch
import shutil
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape

from chapkit.cli.mlproject import (
    CANONICAL_FILENAMES,
    TYPE_MAP,
    MLProject,
    MLProjectError,
    _coerce_default,
    _python_identifier,
    find_mlproject,
    parse_mlproject,
    slugify,
    translate_command,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates" / "migrate"
_SHARED_TEMPLATE_DIR = Path(__file__).parent / "templates"

_VALID_BASE_IMAGES = ("chapkit-py", "chapkit-r", "chapkit-r-inla")

# Blocklist. Anything not explicitly chaff stays at root, because migrated
# scripts may read arbitrary data / config files at runtime (e.g. a YAML
# schema loaded by train.py, a JSON config source()d from an R helper). The
# only things we move are the user's chapkit-project-metadata (so our
# freshly-generated main.py / Dockerfile / README win) and obvious ad-hoc
# runners + stale data.
_CHAFF_FILENAMES = {
    # original MLproject definition
    "MLproject",
    "MLproject.yaml",
    "MLproject.yml",
    # user's existing project metadata - we regenerate these
    "README.md",
    "README.rst",
    "README",
    "README.txt",
    ".gitignore",
    ".dockerignore",
    ".python-version",
    "pyproject.toml",
    # chapkit scaffold artefacts from a prior greenfield init
    "Makefile",
    "Dockerfile",
    "compose.yml",
    "compose.yaml",
    "postman_collection.json",
    "CHAPKIT.md",
    # R profile
    ".Rprofile",
}
_CHAFF_DIRS = {"input", "output", "renv"}
_CHAFF_DIR_PATTERNS = ("example_data*", "examples")
_CHAFF_GLOBS_AT_ROOT = (
    "isolated_run.*",
    "evaluate_one_step.py",
    "slides.md",
    "*.pickle",
    "*.rds",
    "*.model",
    "predictions*.csv",
    "training_data*.csv",
    "future_data*.csv",
    "historic_data*.csv",
)

_IGNORE_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".DS_Store", ".ruff_cache"}
_GENERATED_FILENAMES: tuple[str, ...] = (
    "main.py",
    "pyproject.toml",
    "Dockerfile",
    "compose.yml",
    "Makefile",
    "CHAPKIT.md",
    "postman_collection.json",
    "README.md",
    ".gitignore",
    ".dockerignore",
)


class MigrateError(MLProjectError):
    """Raised when migrate cannot proceed."""


class Action(str, Enum):
    """What migrate plans to do with a single path."""

    KEEP = "keep"
    MOVE_TO_OLD = "chaff"
    IGNORE = "ignore"


@dataclass
class ClassifiedPath:
    """A top-level path with its classification decision."""

    path: Path
    action: Action
    reason: str


def classify(project_path: Path) -> list[ClassifiedPath]:
    """Classify every immediate child of the project path."""
    result: list[ClassifiedPath] = []
    for child in sorted(project_path.iterdir()):
        name = child.name
        if name in _IGNORE_NAMES:
            result.append(ClassifiedPath(child, Action.IGNORE, f"ignored ({name})"))
            continue
        if name == "_old":
            result.append(ClassifiedPath(child, Action.IGNORE, "existing _old/ is handled separately"))
            continue
        if child.is_file():
            result.append(_classify_file(child))
        else:
            result.append(_classify_dir(child))
    return result


def _classify_file(path: Path) -> ClassifiedPath:
    name = path.name
    if name in _CHAFF_FILENAMES:
        return ClassifiedPath(path, Action.MOVE_TO_OLD, f"chapkit-owned metadata ({name})")
    for pattern in _CHAFF_GLOBS_AT_ROOT:
        if fnmatch.fnmatch(name, pattern):
            return ClassifiedPath(path, Action.MOVE_TO_OLD, f"matches chaff pattern {pattern!r}")
    return ClassifiedPath(path, Action.KEEP, "kept at root")


def _classify_dir(path: Path) -> ClassifiedPath:
    name = path.name
    if name in _CHAFF_DIRS:
        return ClassifiedPath(path, Action.MOVE_TO_OLD, f"chaff dir {name}/")
    for pattern in _CHAFF_DIR_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return ClassifiedPath(path, Action.MOVE_TO_OLD, f"matches chaff dir pattern {pattern!r}")
    return ClassifiedPath(path, Action.KEEP, f"kept directory {name}/")


def _language_from_image(image: str) -> str:
    return {"chapkit-py": "python", "chapkit-r": "r", "chapkit-r-inla": "r"}[image]


def _extract_user_dependencies(project_path: Path) -> list[str]:
    """Read [project.dependencies] from the user's pyproject.toml.

    Returns the deps as raw requirement strings (e.g. "pandas>=2.0", "numpy").
    Entries that reference chapkit itself are dropped; the generated
    pyproject.toml adds chapkit with its own pinned version. Returns an empty
    list if the file is missing or unparseable.
    """
    pyproject = project_path / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
    except (tomllib.TOMLDecodeError, OSError):
        return []
    raw = data.get("project", {}).get("dependencies", [])
    if not isinstance(raw, list):
        return []
    kept: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        head = entry.split(";", 1)[0].strip()
        # dependency name is everything up to the first version/extra operator
        name = head.split("[", 1)[0]
        for op in ("==", ">=", "<=", "~=", "!=", ">", "<", " "):
            name = name.split(op, 1)[0]
        if name.strip().lower() == "chapkit":
            continue
        kept.append(entry)
    return kept


def _any_r_script_uses(project_path: Path, packages: tuple[str, ...]) -> bool:
    for script in project_path.glob("*.[rR]"):
        try:
            text = script.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pkg in packages:
            if f"library({pkg})" in text or f'library("{pkg}")' in text:
                return True
    return False


def detect_base_image(
    project_path: Path,
    mlproject: MLProject,
    override: str | None,
) -> tuple[str, str, str]:
    """Return (base_image, language, rationale)."""
    if override is not None:
        if override not in _VALID_BASE_IMAGES:
            raise MigrateError(f"--base-image must be one of {', '.join(_VALID_BASE_IMAGES)}; got {override!r}")
        return override, _language_from_image(override), f"explicit --base-image {override}"

    r_scripts = list(project_path.glob("*.r")) + list(project_path.glob("*.R"))
    py_scripts = list(project_path.glob("*.py"))
    docker_env = mlproject.env_hints.get("docker_env", "")
    env_suggests_inla = "docker_r_inla" in docker_env
    uses_inla = env_suggests_inla or _any_r_script_uses(project_path, ("INLA", "fmesher", "inla"))

    if r_scripts and py_scripts:
        rationale = "found both R and Python scripts; chapkit-r-inla covers both"
        return "chapkit-r-inla", "mixed", rationale
    if r_scripts:
        if uses_inla:
            hint = "MLproject declares docker_r_inla" if env_suggests_inla else "R script imports INLA"
            return "chapkit-r-inla", "r", f"R + INLA ({hint})"
        return "chapkit-r", "r", "R scripts, no INLA detected"
    if py_scripts:
        return "chapkit-py", "python", "Python scripts only"
    raise MigrateError("No Python (.py) or R (.r/.R) scripts at the project root; cannot pick a base image.")


_VALID_ASSESSED_STATUS = {"gray", "red", "orange", "yellow", "green"}
_VALID_PERIOD_TYPES = {"weekly", "monthly"}


def build_service_info_context(mlproject: MLProject) -> dict[str, Any]:
    """Build the ModelMetadata + capability-constraints block for MLServiceInfo.

    Handles the MLproject -> chapkit vocabulary mismatches:
    - MLproject `supported_period_type: any` becomes `monthly` (chapkit's
      PeriodType only has weekly/monthly); otherwise the declared value
      passes through if recognised, else `monthly`.
    - `author_assessed_status` is validated against chapkit's AssessedStatus
      enum; unknown values fall back to None.
    - URL-typed fields (organization_logo_url, repository_url,
      documentation_url) are only emitted if the value looks like a URL
      (starts with http:// or https://) - MLServiceInfo validates them as
      HttpUrl and we don't want to crash on a free-text slot.
    """
    meta = mlproject.meta_data

    def _str_or_none(key: str) -> str | None:
        value = meta.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _url_or_none(key: str) -> str | None:
        value = _str_or_none(key)
        if value and (value.startswith("http://") or value.startswith("https://")):
            return value
        return None

    assessed_raw = _str_or_none("author_assessed_status")
    assessed_status = assessed_raw if assessed_raw in _VALID_ASSESSED_STATUS else None

    period_raw = (mlproject.supported_period_type or "").strip().lower()
    if period_raw in _VALID_PERIOD_TYPES:
        period_type = period_raw
    else:
        period_type = "monthly"

    return {
        "META_AUTHOR": _str_or_none("author"),
        "META_AUTHOR_NOTE": _str_or_none("author_note") or _str_or_none("description"),
        "META_ASSESSED_STATUS": assessed_status,
        "META_CONTACT_EMAIL": _str_or_none("contact_email"),
        "META_ORGANIZATION": _str_or_none("organization"),
        "META_ORGANIZATION_LOGO_URL": _url_or_none("organization_logo_url"),
        "META_CITATION_INFO": _str_or_none("citation_info"),
        "META_REPOSITORY_URL": _url_or_none("repository_url") or _url_or_none("source_url"),
        "META_DOCUMENTATION_URL": _url_or_none("documentation_url"),
        "PERIOD_TYPE": period_type,
        "REQUIRED_COVARIATES": list(mlproject.required_covariates),
        "ALLOW_FREE_COVARIATES": mlproject.allow_free_additional_continuous_covariates,
        "REQUIRES_GEO": mlproject.requires_geo,
    }


def build_config_fields(mlproject: MLProject) -> list[tuple[str, str, str]]:
    """Turn MLproject user_options into (field_name, python_type, default_repr) tuples."""
    type_names = {int: "int", float: "float", str: "str", bool: "bool"}
    fields: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for opt_name, opt_body in mlproject.user_options.items():
        declared_type = str(opt_body.get("type", "string")).lower()
        py_type = TYPE_MAP.get(declared_type, str)
        type_name = type_names.get(py_type, "str")
        if "default" in opt_body:
            coerced = _coerce_default(opt_body["default"], py_type)
            default_repr = repr(coerced)
        else:
            default_repr = "..."
        fields.append((opt_name, type_name, default_repr))
        seen.add(opt_name)
    if "prediction_periods" not in seen:
        fields.insert(0, ("prediction_periods", "int", "3"))
    return fields


def _render(template_dir: Path, template_name: str, context: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader([template_dir, _SHARED_TEMPLATE_DIR]),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**context)


def _parse_param_overrides(raw: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not raw:
        return overrides
    for entry in raw:
        if "=" not in entry:
            raise typer.BadParameter(f"--param expects NAME=FILENAME, got: {entry}")
        name, _, value = entry.partition("=")
        name, value = name.strip(), value.strip()
        if not name or not value:
            raise typer.BadParameter(f"--param expects NAME=FILENAME, got: {entry}")
        overrides[name] = value
    return overrides


def _collect_unknown_params(command: str, known: dict[str, str]) -> list[str]:
    import string as _string

    missing: list[str] = []
    for _, field_name, _, _ in _string.Formatter().parse(command):
        if not field_name:
            continue
        base = field_name.split(".", 1)[0].split("[", 1)[0]
        if base not in known and base not in missing:
            missing.append(base)
    return missing


def _resolve_unknown_params(
    mlproject: MLProject,
    overrides: dict[str, str],
    assume_yes: bool,
) -> dict[str, str]:
    known = {**CANONICAL_FILENAMES, **overrides}
    for ep_name in ("train", "predict"):
        command = mlproject.entry_points[ep_name].command
        missing = _collect_unknown_params(command, known)
        for param in missing:
            if assume_yes:
                raise MigrateError(
                    f"Unknown MLproject parameter {{{param}}} in {ep_name!r}; "
                    "rerun without --yes to answer interactively, or pass --param."
                )
            filename = typer.prompt(
                f"MLproject parameter {{{param}}} has no canonical mapping. Filename to substitute (e.g. data.csv)"
            ).strip()
            if not filename:
                raise MigrateError(f"Empty filename for {{{param}}}; aborting.")
            overrides[param] = filename
            known[param] = filename
    return overrides


def _resolve_base_image(
    project_path: Path,
    mlproject: MLProject,
    override: str | None,
    assume_yes: bool,
) -> tuple[str, str]:
    base_image, language, rationale = detect_base_image(project_path, mlproject, override)
    typer.echo(f"  base image: ghcr.io/dhis2-chap/{base_image}  ({rationale})")
    if language == "mixed" and override is None and not assume_yes:
        if not typer.confirm("Using chapkit-r-inla to cover both languages. OK?", default=True):
            raise typer.Exit(code=1)
    return base_image, language


def _print_plan(
    project_path: Path,
    plan: list[ClassifiedPath],
    renders: list[str],
    base_image: str,
) -> None:
    typer.echo("")
    typer.echo(f"Plan for {project_path}:")
    typer.echo(f"  base image: ghcr.io/dhis2-chap/{base_image}:latest")
    typer.echo("")
    typer.echo("  Paths:")
    for item in plan:
        if item.action is Action.IGNORE:
            continue
        marker = "keep" if item.action is Action.KEEP else "-> _old/"
        typer.echo(f"    {marker:9s} {item.path.name}  ({item.reason})")
    typer.echo("")
    typer.echo("  Files to generate at root:")
    for filename in renders:
        typer.echo(f"    write  {filename}")
    typer.echo("")


def _execute_moves(plan: list[ClassifiedPath], old_dir: Path) -> int:
    moved = 0
    for item in plan:
        if item.action is not Action.MOVE_TO_OLD:
            continue
        dest = old_dir / item.path.name
        shutil.move(str(item.path), str(dest))
        moved += 1
    return moved


def _execute_writes(
    project_path: Path,
    renders: list[str],
    rendered: dict[str, str],
) -> int:
    for filename in renders:
        (project_path / filename).write_text(rendered[filename])
    return len(renders)


def migrate_command(
    path: Annotated[
        Path,
        typer.Argument(help="Directory containing the MLproject file (default: current directory)."),
    ] = Path("."),
    base_image: Annotated[
        str | None,
        typer.Option("--base-image", help="Override base image (chapkit-py, chapkit-r, chapkit-r-inla)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan without touching any files."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts (for scripting / CI)."),
    ] = False,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            help="Override the filename substituted for an MLproject parameter (repeatable).",
        ),
    ] = None,
) -> None:
    """Adopt an existing MLproject directory as a chapkit project."""
    project_path = path.resolve()
    try:
        _run(project_path, base_image, dry_run, yes, param)
    except MigrateError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _run(
    project_path: Path,
    base_image: str | None,
    dry_run: bool,
    assume_yes: bool,
    param: list[str] | None,
) -> None:
    old_dir = project_path / "_old"
    if old_dir.exists() and not dry_run:
        raise MigrateError(f"{old_dir} already exists. Rename or remove it before re-running migrate.")

    mlproject_file = find_mlproject(project_path)
    mlproject = parse_mlproject(mlproject_file)
    typer.echo(f"Found MLproject: {mlproject_file} (name={mlproject.name!r})")

    overrides = _parse_param_overrides(param)
    overrides = _resolve_unknown_params(mlproject, overrides, assume_yes)

    train_command = translate_command(mlproject.entry_points["train"].command, overrides)
    predict_command = translate_command(mlproject.entry_points["predict"].command, overrides)

    resolved_base_image, language = _resolve_base_image(project_path, mlproject, base_image, assume_yes)

    plan = classify(project_path)

    # Everything except source files / lockfiles / LICENSE / .github/ has been
    # bucketed as MOVE_TO_OLD, so by the time we generate files at root there
    # is nothing left to collide with. That includes any existing README.md,
    # pyproject.toml, .gitignore, .dockerignore, etc. - they land in _old/.
    renders = list(_GENERATED_FILENAMES)

    user_deps = _extract_user_dependencies(project_path)
    # Prefer a meta_data description from the MLproject for the service description
    # (e.g. ewars_template has a nice paragraph). Fall back to our generic template.
    meta_description = mlproject.meta_data.get("description")
    if isinstance(meta_description, str) and meta_description.strip():
        project_description = meta_description.strip()
    else:
        project_description = f"chapkit service migrated from the {mlproject.name} MLproject"
    meta_display_name = mlproject.meta_data.get("display_name")
    if isinstance(meta_display_name, str) and meta_display_name.strip():
        project_display_name = meta_display_name.strip()
    else:
        project_display_name = mlproject.name

    context: dict[str, Any] = {
        "PROJECT_NAME": mlproject.name,
        "PROJECT_DISPLAY_NAME": project_display_name,
        "PROJECT_SLUG": slugify(mlproject.name),
        "PROJECT_IDENTIFIER": _python_identifier(mlproject.name),
        "PROJECT_DESCRIPTION": project_description,
        "BASE_IMAGE": resolved_base_image,
        "LANGUAGE": language,
        "TRAIN_COMMAND": train_command,
        "PREDICT_COMMAND": predict_command,
        "CONFIG_FIELDS": build_config_fields(mlproject),
        "HAS_RENV": (project_path / "renv.lock").is_file(),
        "HAS_USER_PYPROJECT": (project_path / "pyproject.toml").is_file(),
        "USER_DEPENDENCIES": user_deps,
        "CHAPKIT_VERSION": _get_chapkit_version(),
        **build_service_info_context(mlproject),
    }

    rendered = _render_all(context)

    _print_plan(project_path, plan, renders, resolved_base_image)

    if dry_run:
        typer.echo("Dry run: no files changed.")
        return

    if not assume_yes:
        if not typer.confirm("Proceed?", default=True):
            typer.echo("Aborted.")
            raise typer.Exit(code=1)

    old_dir.mkdir()
    moved = _execute_moves(plan, old_dir)
    written = _execute_writes(project_path, renders, rendered)

    typer.echo("")
    typer.echo(f"Moved {moved} item(s) to _old/.")
    typer.echo(f"Generated {written} file(s) at project root.")
    if user_deps:
        typer.echo(f"Merged {len(user_deps)} dependency(ies) from the original pyproject.toml into the new one.")
    if (old_dir / "pyproject.toml").is_file():
        typer.echo("Your original pyproject.toml is preserved at _old/pyproject.toml for reference.")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  uv sync && uv run python main.py")
    typer.echo(
        f"  # or: docker build -t {context['PROJECT_SLUG']} . && docker run --rm -p 8000:8000 {context['PROJECT_SLUG']}"
    )
    typer.echo("See CHAPKIT.md for more.")


def _render_all(context: dict[str, Any]) -> dict[str, str]:
    t = _TEMPLATE_DIR
    return {
        "main.py": _render(t, "main_migrate.py.jinja2", context),
        "pyproject.toml": _render(t, "pyproject_migrate.toml.jinja2", context),
        "Dockerfile": _render(t, "Dockerfile_migrate.jinja2", context),
        "compose.yml": _render(_SHARED_TEMPLATE_DIR, "compose.yml.jinja2", context),
        "Makefile": _render(t, "Makefile_migrate.jinja2", context),
        "CHAPKIT.md": _render(t, "CHAPKIT_migrate.md.jinja2", context),
        "postman_collection.json": _render(_SHARED_TEMPLATE_DIR, "postman_collection_ml_shell.json.jinja2", context),
        "README.md": _render(t, "README_migrate.md.jinja2", context),
        ".gitignore": _render(t, "gitignore_migrate.jinja2", context),
        ".dockerignore": _render(t, "dockerignore_migrate.jinja2", context),
    }


#: Floor for the `chapkit>=...` dep that migrate emits into the generated
#: pyproject.toml. This is the minimum chapkit release that has `chapkit run`
#: (the command the generated main.py and Dockerfile CMD depend on). It's
#: explicitly NOT the running chapkit's own version, because that could be a
#: `.devN` not-yet-published release and would leave the migrated project
#: with an uninstallable dep.
_MIN_CHAPKIT_VERSION = "0.19.0"


def _get_chapkit_version() -> str:
    return _MIN_CHAPKIT_VERSION
