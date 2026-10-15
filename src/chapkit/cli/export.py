"""Generate an MLproject file from a chapkit service module such as main.py."""

from __future__ import annotations

import difflib
import os
import string
import sys
from collections.abc import Generator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Any, TypeGuard

import typer
import yaml
from pydantic_core import to_jsonable_python

from chapkit.cli.mlproject import (
    MLPROJECT_FILENAMES,
    RUNNER_PLACEHOLDERS,
    MLProjectError,
    build_config_schema,
    build_ml_service_info,
    parse_mlproject,
    slugify,
    translate_to_runner_template,
)

if TYPE_CHECKING:
    from chapkit.api import MLServiceInfo
    from chapkit.artifact import ArtifactHierarchy
    from chapkit.config import BaseConfig
    from chapkit.ml import ShellModelRunner

# First-line marker of a generated MLproject; anything else is treated as hand-written.
GENERATED_MARKER = "by `chapkit mlproject export`"

# Filled in by ServiceBuilder at build time, so they say nothing about the MLproject.
PROVENANCE_FIELDS: set[str] = {"git_revision", "chapkit_version", "servicekit_version"}

# chapkit's PeriodType values, spelled the way chap-core MLprojects write them.
MLPROJECT_PERIOD_TYPES = {"weekly": "week", "monthly": "month", "any": "any"}

# ShellModelRunner placeholder name -> MLproject parameter; literal paths (model, config.yml) stay as written.
MLPROJECT_PARAMETERS = {
    placeholder.strip("{}"): parameter
    for parameter, placeholder in RUNNER_PLACEHOLDERS.items()
    if placeholder.startswith("{")
}

# Schema keys that document an option without changing what it accepts.
ANNOTATION_KEYS = frozenset({"title", "description"})


@dataclass(frozen=True, slots=True)
class ServiceParts:
    """The objects in a service module that an MLproject describes."""

    info: MLServiceInfo
    config_schema: type[BaseConfig]
    runner: ShellModelRunner[Any]
    hierarchy: ArtifactHierarchy | None


@contextmanager
def _inside(directory: Path) -> Generator[None]:
    """Run with directory as the working directory and first on sys.path, restoring both afterwards."""
    previous_cwd = Path.cwd()
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if str(directory) in sys.path:
            sys.path.remove(str(directory))


def load_service_module(path: Path) -> ModuleType:
    """Import a service module from its file, from inside its directory as the service itself would run."""
    if not path.is_file():
        raise MLProjectError(f"Service module not found: {path}")
    # A private module name keeps a project's main.py from clashing with anything already imported.
    module = ModuleType(f"_chapkit_export_{path.stem}")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        # Compiled from source rather than imported: no stale or new __pycache__ in the project.
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        # Whatever the module prints while it builds its app (startup logs) goes to stderr, keeping stdout ours.
        with _inside(path.parent), redirect_stdout(sys.stderr):
            exec(code, module.__dict__)
    except Exception as error:
        raise MLProjectError(f"Importing {path} failed: {type(error).__name__}: {error}") from error
    finally:
        sys.modules.pop(module.__name__, None)
    return module


def _unique_instances[T](module: ModuleType, kind: type[T]) -> list[T]:
    """Return the distinct module-level objects of a type, in definition order."""
    seen: dict[int, T] = {}
    for value in vars(module).values():
        if isinstance(value, kind):
            seen.setdefault(id(value), value)
    return list(seen.values())


def _find_config_schema(module: ModuleType, config_name: str | None) -> type[BaseConfig]:
    """Return the module's config class: the one named, else the only BaseConfig subclass it defines."""
    from chapkit.config import BaseConfig

    def is_config(value: Any) -> TypeGuard[type[BaseConfig]]:
        """Return whether value is a config class other than BaseConfig itself."""
        return isinstance(value, type) and issubclass(value, BaseConfig) and value is not BaseConfig

    if config_name is not None:
        named = getattr(module, config_name, None)
        if not is_config(named):
            raise MLProjectError(f"--config {config_name!r} is not a BaseConfig subclass in the service module")
        return named

    candidates: list[type[BaseConfig]] = [value for value in vars(module).values() if is_config(value)]
    defined_here = [value for value in candidates if value.__module__ == module.__name__]
    configs = list(dict.fromkeys(defined_here or candidates))
    if not configs:
        raise MLProjectError("No BaseConfig subclass found in the service module")
    if len(configs) > 1:
        names = ", ".join(config.__name__ for config in configs)
        raise MLProjectError(f"Several config classes found ({names}); pick one with --config NAME")
    return configs[0]


def find_service_parts(module: ModuleType, config_name: str | None = None) -> ServiceParts:
    """Find the service info, config class, shell runner and artifact hierarchy defined in a service module."""
    from chapkit.api import MLServiceInfo
    from chapkit.artifact import ArtifactHierarchy
    from chapkit.ml import ShellModelRunner
    from chapkit.ml.runner import BaseModelRunner

    infos = _unique_instances(module, MLServiceInfo)
    if len(infos) != 1:
        raise MLProjectError(f"Expected one MLServiceInfo in the service module, found {len(infos)}")

    runners = _unique_instances(module, ShellModelRunner)
    if not runners:
        if any(isinstance(value, BaseModelRunner) for value in vars(module).values()):
            raise MLProjectError(
                "Only ShellModelRunner services can be exported: an MLproject runs train and predict as shell commands"
            )
        raise MLProjectError("No ShellModelRunner found in the service module")
    if len(runners) > 1:
        raise MLProjectError(f"Expected one ShellModelRunner in the service module, found {len(runners)}")

    hierarchies = _unique_instances(module, ArtifactHierarchy)
    return ServiceParts(
        info=infos[0],
        config_schema=_find_config_schema(module, config_name),
        runner=runners[0],
        hierarchy=hierarchies[0] if len(hierarchies) == 1 else None,
    )


def mlproject_name(parts: ServiceParts) -> str:
    """Return the MLproject name: the artifact hierarchy's when it slugifies to the service id, else the id."""
    if parts.hierarchy is not None and slugify(parts.hierarchy.name) == parts.info.id:
        return parts.hierarchy.name
    return parts.info.id.replace("-", "_")


def mlproject_command(runner_command: str) -> str:
    """Translate a ShellModelRunner command to MLproject placeholders, the inverse of translate_to_runner_template.

    An MLproject command goes through str.format twice before it runs (MLproject parameters,
    then the runner's own placeholders), so a literal brace is written four times: {{ in the
    runner command becomes {{{{.
    """
    parts: list[str] = []
    for literal, field_name, format_spec, conversion in string.Formatter().parse(runner_command):
        parts.append(literal.replace("{", "{{{{").replace("}", "}}}}"))
        if field_name is None:
            continue
        parameter = MLPROJECT_PARAMETERS.get(field_name)
        if parameter is None or format_spec or conversion:
            raise MLProjectError(f"Runner command placeholder {{{field_name}}} has no MLproject parameter")
        parts.append("{" + parameter + "}")
    return "".join(parts)


def _entry_point(runner_command: str) -> dict[str, Any]:
    """Build an MLproject entry point from a ShellModelRunner command."""
    command = mlproject_command(runner_command)
    parameters = {name: "path" for _, name, _, _ in string.Formatter().parse(command) if name}
    return {"parameters": parameters, "command": command} if parameters else {"command": command}


def _inline_refs(schema: Any, definitions: dict[str, Any]) -> Any:
    """Replace $ref pointers into $defs with the definitions they name, so each option stands alone."""
    if isinstance(schema, list):
        return [_inline_refs(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        target = dict(definitions[reference.removeprefix("#/$defs/")])
        # The definition's title is its Python class name, which means nothing in an MLproject.
        target.pop("title", None)
        merged = {**target, **{key: value for key, value in schema.items() if key != "$ref"}}
        return _inline_refs(merged, definitions)
    return {key: _inline_refs(value, definitions) for key, value in schema.items()}


def _validation_schema(schema: Any) -> Any:
    """Return a property schema without its default and documentation keys: what it accepts, nothing else."""
    if isinstance(schema, list):
        return [_validation_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {key: _validation_schema(value) for key, value in schema.items() if key not in ANNOTATION_KEYS}


def _field_validation_schemas(config_schema: type[BaseConfig]) -> dict[str, Any]:
    """Return what each config option accepts, keyed by option name, with $refs inlined and the default left out."""
    schema = config_schema.model_json_schema()
    definitions = schema.get("$defs", {})
    schemas: dict[str, Any] = {}
    for option_name, option in schema.get("properties", {}).items():
        accepted = _validation_schema(_inline_refs(option, definitions))
        accepted.pop("default", None)
        schemas[option_name] = accepted
    return schemas


def _generated_title(name: str) -> str:
    """Return the title Pydantic derives from a field name."""
    return name.replace("_", " ").title()


def mlproject_user_options(config_schema: type[BaseConfig]) -> dict[str, dict[str, Any]]:
    """Turn a config class into MLproject user_options, one JSON Schema property per field.

    Fields chapkit adds to every config (prediction_periods, additional_continuous_covariates)
    are left out unless the class changes their default or what they accept.
    """
    from chapkit.config import BaseConfig

    schema = config_schema.model_json_schema()
    definitions = schema.get("$defs", {})
    properties = schema.get("properties", {})
    reserved = BaseConfig.model_fields
    accepted = _field_validation_schemas(config_schema)
    accepted_by_base = _field_validation_schemas(BaseConfig)

    options: dict[str, dict[str, Any]] = {}
    for field_name, field in config_schema.model_fields.items():
        default = None if field.is_required() else field.get_default(call_default_factory=True)
        option_name = field.alias or field_name
        if (
            field_name in reserved
            and default == reserved[field_name].get_default(call_default_factory=True)
            and accepted[option_name] == accepted_by_base[field_name]
        ):
            continue
        option = _inline_refs(dict(properties[option_name]), definitions)
        if option.get("title") in {_generated_title(field_name), _generated_title(option_name)}:
            option.pop("title")
        description = option.pop("description", None)
        option.pop("default", None)
        if not field.is_required():
            # default_factory fields have no default in the JSON schema; tuples and enums become plain values.
            option["default"] = to_jsonable_python(default)
        # Title and type lead and the usually long description comes last, so each option reads top-down.
        if description is not None:
            option["description"] = description
        leading = [key for key in ("title", "type") if key in option]
        options[option_name] = {key: option[key] for key in [*leading, *(key for key in option if key not in leading)]}
    return options


def _url(value: Any) -> str | None:
    """Return a URL field as a string, or None."""
    return str(value) if value is not None else None


def build_mlproject(parts: ServiceParts) -> dict[str, Any]:
    """Build the MLproject document that `chapkit mlproject run` reads back as the same service."""
    info = parts.info
    metadata = info.model_metadata
    meta_data = {
        "display_name": info.display_name,
        "description": info.description,
        "author": metadata.author,
        "author_note": metadata.author_note,
        "author_assessed_status": metadata.author_assessed_status.value if metadata.author_assessed_status else None,
        "contact_email": metadata.contact_email,
        "organization": metadata.organization,
        "organization_logo_url": _url(metadata.organization_logo_url),
        "citation_info": metadata.citation_info,
        "repository_url": _url(metadata.repository_url),
        "documentation_url": _url(metadata.documentation_url),
    }
    document: dict[str, Any] = {
        "name": mlproject_name(parts),
        "version": info.version,
        "target": info.target,
        "supported_period_type": MLPROJECT_PERIOD_TYPES[info.period_type.value],
        "required_covariates": list(info.required_covariates),
        "allow_free_additional_continuous_covariates": info.allow_free_additional_continuous_covariates,
        "requires_geo": info.requires_geo,
        "min_prediction_periods": info.min_prediction_periods,
        "max_prediction_periods": info.max_prediction_periods,
    }
    if info.hpo_search_space is not None:
        document["hpo_search_space"] = info.hpo_search_space
    document["meta_data"] = {key: value for key, value in meta_data.items() if value is not None}
    user_options = mlproject_user_options(parts.config_schema)
    if user_options:
        document["user_options"] = user_options
    if parts.runner.adapters:
        document["adapters"] = dict(parts.runner.adapters)
    document["entry_points"] = {
        "train": _entry_point(parts.runner.train_command),
        "predict": _entry_point(parts.runner.predict_command),
    }
    return document


def render_mlproject(parts: ServiceParts, source: str) -> str:
    """Render the MLproject file, with a header saying where it comes from and how to regenerate it."""
    header = [
        f"# Generated from {source} {GENERATED_MARKER} -- do not edit by hand.",
        f"# Regenerate with `chapkit mlproject export {source}`; add --check to fail when it is stale.",
    ]
    if parts.runner.config_format == "flat":
        header.append("# The scripts read a flat config.yml: run with `chapkit mlproject run . --config-format flat`.")
    body = yaml.safe_dump(build_mlproject(parts), sort_keys=False, allow_unicode=True, width=100)
    return "\n".join(header) + "\n\n" + body


def _field_defaults(config_schema: type[BaseConfig]) -> dict[str, Any]:
    """Return each config option's default, keyed by option name; required options map to a marker."""
    return {
        field.alias or name: "<required>" if field.is_required() else field.get_default(call_default_factory=True)
        for name, field in config_schema.model_fields.items()
    }


def semantic_problems(mlproject_file: Path, parts: ServiceParts) -> list[str]:
    """Read an MLproject the way `chapkit mlproject run` does and list where it differs from the service."""
    problems: list[str] = []
    issues: list[str] = []
    mlproject = parse_mlproject(mlproject_file)
    problems += [f"chapkit warning: {warning}" for warning in mlproject.parse_warnings]

    served = build_ml_service_info(mlproject, issues).model_dump(mode="json", exclude=PROVENANCE_FIELDS)
    expected = parts.info.model_dump(mode="json", exclude=PROVENANCE_FIELDS)
    for key in sorted(set(served) | set(expected)):
        if served.get(key) != expected.get(key):
            problems.append(
                f"info.{key}: the MLproject gives {served.get(key)!r}, the service has {expected.get(key)!r}"
            )

    generated = _field_defaults(build_config_schema(mlproject, issues))
    defaults = _field_defaults(parts.config_schema)
    for key in sorted(set(generated) | set(defaults)):
        if key not in generated or key not in defaults:
            where = "the service" if key in defaults else "the MLproject"
            problems.append(f"config.{key}: only {where} has this option")
        elif generated[key] != defaults[key]:
            problems.append(
                f"config.{key}: the MLproject default is {generated[key]!r}, the service default is {defaults[key]!r}"
            )

    generated_schemas = _field_validation_schemas(build_config_schema(mlproject, issues))
    service_schemas = _field_validation_schemas(parts.config_schema)
    for key in sorted(set(generated_schemas) & set(service_schemas)):
        if generated_schemas[key] != service_schemas[key]:
            problems.append(
                f"config.{key}: the MLproject accepts {generated_schemas[key]!r}, "
                f"the service accepts {service_schemas[key]!r}"
            )

    for name in ("train", "predict"):
        command = translate_to_runner_template(mlproject.entry_points[name].command)
        runner_command = getattr(parts.runner, f"{name}_command")
        if command != runner_command:
            problems.append(f"{name} command: the MLproject runs {command!r}, the service runs {runner_command!r}")

    if mlproject.adapters != parts.runner.adapters:
        problems.append(
            f"adapters: the MLproject has {mlproject.adapters!r}, the service has {parts.runner.adapters!r}"
        )

    problems += [f"chapkit warning: {issue}" for issue in issues]
    return problems


def _default_output(module_path: Path) -> Path:
    """Return the existing MLproject next to the module, or where a new one goes."""
    for filename in MLPROJECT_FILENAMES:
        candidate = module_path.parent / filename
        if candidate.is_file():
            return candidate
    return module_path.parent / MLPROJECT_FILENAMES[0]


def export_command(
    path: Annotated[
        Path,
        typer.Argument(help="Service module to read (default: main.py in the current directory)."),
    ] = Path("main.py"),
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="MLproject file to write (default: the MLproject next to the module).",
            dir_okay=False,
        ),
    ] = None,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help=(
                "Write nothing; exit 1 when the MLproject is stale, or when chapkit mlproject run would "
                "read it as a different service."
            ),
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option(help="Name of the config class, when the module defines several."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an MLproject that was not generated by this command."),
    ] = False,
) -> None:
    """Generate an MLproject from a service module so chapkit mlproject run serves the same service."""
    module_path = path.expanduser().resolve()
    target = output.expanduser().resolve() if output is not None else _default_output(module_path)

    try:
        parts = find_service_parts(load_service_module(module_path), config)
    except MLProjectError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    source = module_path.name if target.parent == module_path.parent else str(module_path)
    wanted = render_mlproject(parts, source)
    current = target.read_text(encoding="utf-8") if target.is_file() else None

    if check:
        if current != wanted:
            sys.stdout.writelines(
                difflib.unified_diff(
                    (current or "").splitlines(True), wanted.splitlines(True), target.name, "generated"
                )
            )
            typer.echo(f"\n{target.name} is out of date with {module_path.name}; run `chapkit mlproject export`.")
            raise typer.Exit(code=1)
        problems = semantic_problems(target, parts)
        if problems:
            typer.echo(f"{target.name} does not match {module_path.name} as chapkit mlproject run reads it:")
            for problem in problems:
                typer.echo(f"  - {problem}")
            raise typer.Exit(code=1)
        typer.echo(f"{target.name} is up to date with {module_path.name}")
        return

    first_line = current.splitlines()[0] if current else ""
    if current is not None and GENERATED_MARKER not in first_line and not force:
        typer.echo(
            f"Error: {target} was not generated by chapkit mlproject export; pass --force to overwrite it.", err=True
        )
        raise typer.Exit(code=1)
    target.write_text(wanted, encoding="utf-8")
    typer.echo(f"Wrote {target}")
    for problem in semantic_problems(target, parts):
        typer.echo(f"WARNING: {problem}", err=True)
