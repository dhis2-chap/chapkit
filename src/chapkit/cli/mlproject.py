"""Parse MLproject files and adapt them to chapkit's ShellModelRunner conventions."""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, create_model

from chapkit.config.schemas import BaseConfig

CANONICAL_FILENAMES: dict[str, str] = {
    "train_data": "data.csv",
    "historic_data": "historic.csv",
    "future_data": "future.csv",
    "out_file": "predictions.csv",
    "model": "model",
    "model_config": "config.yml",
    "polygons": "geo.json",
}

# Maps canonical MLproject parameter names to ShellModelRunner's own template
# vocabulary (curly-braced placeholders) where applicable, or to literal paths
# that ShellModelRunner guarantees at train/predict time. Used by `chapkit
# migrate` to emit commands in runner-template form instead of fully-literal,
# so the generated main.py is insulated from future ShellModelRunner filename
# changes and reads more idiomatically.
RUNNER_PLACEHOLDERS: dict[str, str] = {
    "train_data": "{data_file}",
    "historic_data": "{historic_file}",
    "future_data": "{future_file}",
    "out_file": "{output_file}",
    "polygons": "{geo_file}",
    # Literals - ShellModelRunner has no placeholder for these; it just
    # writes `config.yml` to the workspace root and lets scripts save/load
    # `model` as they see fit.
    "model": "model",
    "model_config": "config.yml",
}

ENV_FIELDS: tuple[str, ...] = (
    "docker_env",
    "renv_env",
    "python_env",
    "uv_env",
    "conda_env",
)

MLPROJECT_FILENAMES: tuple[str, ...] = ("MLproject", "MLproject.yaml", "MLproject.yml")

TYPE_MAP: dict[str, type] = {
    "integer": int,
    "int": int,
    "number": float,
    "float": float,
    "string": str,
    "str": str,
    "boolean": bool,
    "bool": bool,
    "path": str,
}


class EntryPoint(BaseModel):
    """A single MLproject entry point (train or predict)."""

    command: str
    parameters: dict[str, str] = Field(default_factory=dict)


class MLProject(BaseModel):
    """Parsed MLproject definition."""

    name: str
    entry_points: dict[str, EntryPoint]
    user_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_hints: dict[str, str] = Field(default_factory=dict)
    meta_data: dict[str, Any] = Field(default_factory=dict)
    supported_period_type: str | None = None
    required_covariates: list[str] = Field(default_factory=list)
    additional_continuous_covariates: list[str] = Field(default_factory=list)
    allow_free_additional_continuous_covariates: bool = False
    requires_geo: bool = False
    target: str | None = None
    source_path: Path | None = None


class MLProjectError(ValueError):
    """Raised when an MLproject file is missing, malformed, or incompatible."""


def find_mlproject(path: Path) -> Path:
    """Find an MLproject file in the given directory."""
    if not path.exists():
        raise MLProjectError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise MLProjectError(f"Expected a directory, got: {path}")
    for name in MLPROJECT_FILENAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise MLProjectError(f"No MLproject file found in {path.resolve()} (looked for: {', '.join(MLPROJECT_FILENAMES)})")


def parse_mlproject(path: Path) -> MLProject:
    """Parse an MLproject file or directory into an MLProject model."""
    mlproject_file = path if path.is_file() else find_mlproject(path)
    with mlproject_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise MLProjectError(f"MLproject file must contain a YAML mapping: {mlproject_file}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MLProjectError(f"MLproject is missing a non-empty 'name' field: {mlproject_file}")

    raw_entry_points = raw.get("entry_points") or {}
    if not isinstance(raw_entry_points, dict):
        raise MLProjectError(f"MLproject 'entry_points' must be a mapping: {mlproject_file}")

    entry_points: dict[str, EntryPoint] = {}
    for ep_name, ep_body in raw_entry_points.items():
        if not isinstance(ep_body, dict):
            raise MLProjectError(f"Entry point '{ep_name}' must be a mapping in {mlproject_file}")
        command = ep_body.get("command")
        if not isinstance(command, str) or not command.strip():
            raise MLProjectError(f"Entry point '{ep_name}' is missing a 'command' string")
        raw_parameters = ep_body.get("parameters") or {}
        parameters: dict[str, str] = {}
        if isinstance(raw_parameters, dict):
            for param_name, param_type in raw_parameters.items():
                parameters[str(param_name)] = str(param_type)
        entry_points[str(ep_name)] = EntryPoint(command=command, parameters=parameters)

    for required in ("train", "predict"):
        if required not in entry_points:
            found = ", ".join(sorted(entry_points)) or "(none)"
            raise MLProjectError(f"MLproject entry_points must define '{required}'. Found: {found} ({mlproject_file})")

    raw_user_options = raw.get("user_options") or {}
    user_options: dict[str, dict[str, Any]] = {}
    if isinstance(raw_user_options, dict):
        for opt_name, opt_body in raw_user_options.items():
            if isinstance(opt_body, dict):
                user_options[str(opt_name)] = dict(opt_body)
            else:
                user_options[str(opt_name)] = {"default": opt_body}

    env_hints: dict[str, str] = {}
    for env_field in ENV_FIELDS:
        value = raw.get(env_field)
        if value is None:
            continue
        if env_field == "docker_env" and isinstance(value, dict):
            image = value.get("image")
            if image:
                env_hints[env_field] = str(image)
            else:
                env_hints[env_field] = yaml.safe_dump(value).strip()
        else:
            env_hints[env_field] = str(value)

    raw_meta = raw.get("meta_data") or {}
    meta_data = dict(raw_meta) if isinstance(raw_meta, dict) else {}

    required_covariates_raw = raw.get("required_covariates") or []
    required_covariates: list[str] = (
        [str(c) for c in required_covariates_raw] if isinstance(required_covariates_raw, list) else []
    )

    additional_continuous_covariates_raw = raw.get("additional_continuous_covariates") or []
    additional_continuous_covariates: list[str] = (
        [str(c) for c in additional_continuous_covariates_raw]
        if isinstance(additional_continuous_covariates_raw, list)
        else []
    )

    supported_period_type = raw.get("supported_period_type")
    if supported_period_type is not None:
        supported_period_type = str(supported_period_type)

    return MLProject(
        name=name.strip(),
        entry_points=entry_points,
        user_options=user_options,
        env_hints=env_hints,
        meta_data=meta_data,
        supported_period_type=supported_period_type,
        required_covariates=required_covariates,
        additional_continuous_covariates=additional_continuous_covariates,
        allow_free_additional_continuous_covariates=bool(raw.get("allow_free_additional_continuous_covariates", False)),
        requires_geo=bool(raw.get("requires_geo", False)),
        target=str(raw["target"]) if "target" in raw and raw["target"] is not None else None,
        source_path=mlproject_file,
    )


def translate_command(command: str, overrides: dict[str, str] | None = None) -> str:
    """Substitute MLproject {param} placeholders with chapkit workspace filenames."""
    return _apply_param_mapping(command, {**CANONICAL_FILENAMES, **(overrides or {})})


def translate_to_runner_template(command: str, overrides: dict[str, str] | None = None) -> str:
    """Substitute MLproject {param} placeholders with ShellModelRunner's template form.

    Unlike `translate_command`, which yields a fully-literal command, this
    returns a string that still contains curly-braced placeholders
    (`{data_file}`, `{historic_file}`, etc.) where ShellModelRunner does its
    own substitution at train/predict time. Literal paths (`model`,
    `config.yml`) are inlined because ShellModelRunner has no templating
    hook for them. This is what `chapkit migrate` embeds in main.py.
    """
    return _apply_param_mapping(command, {**RUNNER_PLACEHOLDERS, **(overrides or {})})


def _apply_param_mapping(command: str, mapping: dict[str, str]) -> str:
    formatter = string.Formatter()
    unknown: list[str] = []
    for _, field_name, _, _ in formatter.parse(command):
        if field_name is None or field_name == "":
            continue
        base = field_name.split(".", 1)[0].split("[", 1)[0]
        if base not in mapping:
            unknown.append(base)
    if unknown:
        known = ", ".join(sorted(mapping))
        missing = ", ".join(sorted(set(unknown)))
        raise MLProjectError(
            f"Unknown MLproject parameter(s): {missing}. Known: {known}. Use --param NAME=FILENAME to override."
        )
    # Escape any literal curly braces in the command that are NOT our placeholders,
    # then format(**mapping). Since `format` uses {key} syntax, all placeholders we
    # care about are already mapped; any stray `{{` / `}}` should survive.
    return command.format(**mapping)


def slugify(value: str) -> str:
    """Return a ServiceInfo-compatible slug: lowercase letters, digits, hyphens, starting with a letter."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug or not slug[0].isalpha():
        slug = f"mlproject-{slug}" if slug else "mlproject"
    return slug


def _python_identifier(value: str) -> str:
    """Return a valid Python identifier derived from value."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"ML_{cleaned}" if cleaned else "MLProject"
    return cleaned


def _python_class_name(value: str) -> str:
    """Return a PascalCase Python class name derived from value."""
    identifier = _python_identifier(value)
    parts = [part for part in identifier.split("_") if part]
    pascal = "".join(part[:1].upper() + part[1:] for part in parts)
    return pascal or "MLProject"


def build_config_schema(mlproject: MLProject) -> type[BaseConfig]:
    """Build a BaseConfig subclass from MLproject user_options, injecting prediction_periods."""
    fields: dict[str, Any] = {}
    for opt_name, opt_body in mlproject.user_options.items():
        declared_type = str(opt_body.get("type", "string")).lower()
        py_type = TYPE_MAP.get(declared_type, str)
        if "default" in opt_body:
            coerced_default = _coerce_default(opt_body["default"], py_type)
            fields[opt_name] = (py_type, coerced_default)
        else:
            fields[opt_name] = (py_type, ...)

    if "prediction_periods" not in fields:
        fields["prediction_periods"] = (int, 3)

    model_name = f"{_python_class_name(mlproject.name)}Config"
    return create_model(model_name, __base__=BaseConfig, **fields)


def _coerce_default(value: Any, target: type) -> Any:
    """Best-effort coercion of user_options defaults into the declared type."""
    if value is None or isinstance(value, target):
        return value
    if target is bool and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return value
    if target in (int, float, str):
        try:
            return target(value)
        except (TypeError, ValueError):
            return value
    return value
