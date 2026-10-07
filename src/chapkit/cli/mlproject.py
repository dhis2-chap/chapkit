"""Parse MLproject files and adapt them to chapkit's ShellModelRunner conventions."""

from __future__ import annotations

import keyword
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, cast

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo

from chapkit.config.schemas import BaseConfig

if TYPE_CHECKING:
    from chapkit.api.service_builder import MLServiceInfo, ModelMetadata

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

# Source spelling of each scalar type, used when describing a field to the user.
_TYPE_SOURCE: dict[type, str] = {int: "int", float: "float", str: "str", bool: "bool"}

# JSON Schema validation keywords mapped to the Pydantic Field kwarg they become, grouped by
# the kind of value they apply to. A keyword on the wrong kind of value is ignored, matching
# JSON Schema's own semantics (e.g. `minimum` has no effect on a string).
_NUMERIC_CONSTRAINTS: dict[str, str] = {
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
}
_STRING_CONSTRAINTS: dict[str, str] = {
    "minLength": "min_length",
    "maxLength": "max_length",
    "pattern": "pattern",
}
_ARRAY_CONSTRAINTS: dict[str, str] = {
    "minItems": "min_length",
    "maxItems": "max_length",
}

# Every spelling chap-core accepts for the forecast horizon bounds, canonical first.
_HORIZON_ALIASES: dict[str, tuple[str, ...]] = {
    bound: (
        f"{bound}_prediction_periods",
        f"{bound}PredictionPeriods",
        f"{bound}_prediction_length",
        f"{bound}PredictionLength",
    )
    for bound in ("min", "max")
}

# chap-core spells period types `week` / `month`; chapkit spells them `weekly` / `monthly`.
_PERIOD_TYPES: dict[str, str] = {
    "week": "weekly",
    "weekly": "weekly",
    "month": "monthly",
    "monthly": "monthly",
    "any": "any",
}
# A model that does not declare a period type accepts both, as in chap-core.
DEFAULT_PERIOD_TYPE = "any"

DEFAULT_PREDICTION_PERIODS = 3


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
    allow_free_additional_continuous_covariates: bool = False
    requires_geo: bool = False
    target: str | None = None
    version: str | None = None
    source_url: str | None = None
    min_prediction_periods: int | None = None
    max_prediction_periods: int | None = None
    source_path: Path | None = None
    parse_warnings: list[str] = Field(default_factory=list)


class MLProjectError(ValueError):
    """Raised when an MLproject file is missing, malformed, or incompatible."""


def resolve_mlproject(path: Path) -> Path:
    """Return the MLproject file at path, which is either the file itself or the directory holding it."""
    if path.is_file():
        return path
    return find_mlproject(path)


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

    supported_period_type = raw.get("supported_period_type")
    if supported_period_type is not None:
        supported_period_type = str(supported_period_type)

    # Contract fields chapkit did not read before are parsed leniently: a bad value is
    # reported and ignored so MLprojects that ran before keep running.
    parse_warnings: list[str] = []
    min_prediction_periods = _parse_horizon_bound(raw, "min", parse_warnings)
    max_prediction_periods = _parse_horizon_bound(raw, "max", parse_warnings)
    if (
        min_prediction_periods is not None
        and max_prediction_periods is not None
        and min_prediction_periods > max_prediction_periods
    ):
        parse_warnings.append(
            f"min_prediction_periods ({min_prediction_periods}) is greater than "
            f"max_prediction_periods ({max_prediction_periods}); ignoring both"
        )
        min_prediction_periods = max_prediction_periods = None

    raw_version = raw.get("version")

    return MLProject(
        name=name.strip(),
        entry_points=entry_points,
        user_options=user_options,
        env_hints=env_hints,
        meta_data=meta_data,
        supported_period_type=supported_period_type,
        required_covariates=required_covariates,
        allow_free_additional_continuous_covariates=bool(raw.get("allow_free_additional_continuous_covariates", False)),
        requires_geo=bool(raw.get("requires_geo", False)),
        target=str(raw["target"]) if "target" in raw and raw["target"] is not None else None,
        version=_clean_text(raw_version),
        source_url=_clean_text(raw.get("source_url")),
        min_prediction_periods=min_prediction_periods,
        max_prediction_periods=max_prediction_periods,
        source_path=mlproject_file,
        parse_warnings=parse_warnings,
    )


def _parse_horizon_bound(raw: dict[str, Any], bound: str, parse_warnings: list[str]) -> int | None:
    """Read a forecast horizon bound under any of the spellings chap-core accepts, ignoring invalid values."""
    for key in _HORIZON_ALIASES[bound]:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            parsed = value
        elif isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
        else:
            parsed = -1
        if parsed < 0:
            parse_warnings.append(f"{key} must be a non-negative integer, got {value!r}; ignoring it")
            return None
        return parsed
    return None


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
    hook for them. This is what `chapkit mlproject migrate` embeds in main.py.
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


def python_field_name(raw: str) -> str:
    """Normalize an MLproject user_option name to a valid Python / Pydantic field name.

    Replaces non-identifier characters (hyphens, dots, spaces, ...) with underscores
    and suffixes Python keywords with a trailing underscore. Names starting with a
    digit are rejected outright - Pydantic forbids leading underscores on field
    names, so there's no safe prefix that wouldn't collide with real user names.

    Raises MLProjectError if the name cannot be sanitized.
    """
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not normalized:
        raise MLProjectError(f"user_option name {raw!r} cannot be sanitized to a valid Python identifier")
    if normalized[0].isdigit():
        raise MLProjectError(
            f"user_option name {raw!r} starts with a digit; rename it to start with a letter "
            "(Pydantic field names cannot have a leading underscore, so there is no safe mapping)"
        )
    if keyword.iskeyword(normalized):
        normalized = normalized + "_"
    if not normalized.isidentifier():
        raise MLProjectError(f"user_option name {raw!r} cannot be sanitized to a valid Python identifier")
    return normalized


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Typed description of one MLproject user_option, ready to become a Pydantic field."""

    name: str
    field_name: str
    annotation: Any
    type_source: str
    required: bool
    default: Any = None
    description: str | None = None
    title: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

    @property
    def alias(self) -> str | None:
        """Return the MLproject option name when it differs from the Python field name."""
        return self.name if self.name != self.field_name else None

    def field_info(self) -> FieldInfo:
        """Return the Pydantic field definition for this option."""
        kwargs: dict[str, Any] = dict(self.constraints)
        if self.description:
            kwargs["description"] = self.description
        if self.title:
            kwargs["title"] = self.title
        if self.alias:
            kwargs["alias"] = self.alias
        default = ... if self.required else self.default
        field_info: FieldInfo = Field(default, **kwargs)
        return field_info


def _clean_text(value: Any) -> str | None:
    """Return a stripped string, or None for missing or blank values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_scalar(declared: Any) -> tuple[Any, str, str]:
    """Map a JSON Schema scalar type name to (python type, source spelling, value kind)."""
    scalar = TYPE_MAP.get(str(declared).lower(), str)
    kind = "number" if scalar in (int, float) else "boolean" if scalar is bool else "string"
    return scalar, _TYPE_SOURCE[scalar], kind


def _enum_values(body: dict[str, Any]) -> list[Any] | None:
    """Return the enum values of a schema when they can form a Literal, otherwise None."""
    values = body.get("enum")
    if isinstance(values, list) and values and all(isinstance(v, (str, int, float, bool)) for v in values):
        return values
    return None


def _resolve_items(items: Any) -> tuple[Any, str]:
    """Map a JSON Schema `items` declaration to (python type, source spelling)."""
    if items is None:
        return Any, "Any"
    if not isinstance(items, dict):
        items = {"type": items}
    values = _enum_values(items)
    if values is not None:
        return cast(Any, Literal)[tuple(values)], f"Literal[{', '.join(repr(v) for v in values)}]"
    if str(items.get("type", "")).lower() == "object":
        return dict[str, Any], "dict[str, Any]"
    scalar, source, _ = _resolve_scalar(items.get("type", "string"))
    return scalar, source


def _resolve_declared(declared: str, body: dict[str, Any]) -> tuple[Any, str, str]:
    """Map one JSON Schema type name to (annotation, source spelling, value kind)."""
    if declared == "array":
        item_annotation, item_source = _resolve_items(body.get("items"))
        return list[item_annotation], f"list[{item_source}]", "array"  # type: ignore[valid-type]
    if declared == "object":
        return dict[str, Any], "dict[str, Any]", "object"
    return _resolve_scalar(declared)


def _union_of(variants: list[tuple[Any, str, str, dict[str, Any]]]) -> tuple[Any, str, str]:
    """Combine (annotation, source, kind, schema) variants into one union, each keeping its own constraints."""
    members: list[Any] = []
    for annotation, _, kind, schema in variants:
        constraints = _collect_constraints(schema, kind)
        members.append(Annotated[annotation, Field(**constraints)] if constraints else annotation)
    if len(members) == 1:
        return members[0], variants[0][1], "union"
    return cast(Any, Union)[tuple(members)], " | ".join(variant[1] for variant in variants), "union"


def _resolve_type(body: dict[str, Any]) -> tuple[Any, str, str]:
    """Map a user_option body to (annotation, source spelling, value kind).

    A list `type` (e.g. `[array, integer]`) or an `anyOf` / `oneOf` list becomes a union of
    its members, each carrying the validation keywords that apply to it; the returned kind
    is then "union" so no keyword is applied to the union as a whole.
    """
    annotation: Any
    subschemas = body.get("anyOf") or body.get("oneOf")
    if isinstance(subschemas, list) and subschemas and all(isinstance(sub, dict) for sub in subschemas):
        nullable = any(str(sub.get("type", "")).lower() == "null" for sub in subschemas)
        non_null = [sub for sub in subschemas if str(sub.get("type", "")).lower() != "null"]
        if non_null:
            annotation, source, kind = _union_of([(*_resolve_type(sub), sub) for sub in non_null])
        else:
            annotation, source, kind = str, "str", "string"
    else:
        declared: Any = body.get("type", "string")
        names = [str(name).lower() for name in declared] if isinstance(declared, list) else [str(declared).lower()]
        nullable = "null" in names
        non_null_names = list(dict.fromkeys(name for name in names if name != "null")) or ["string"]

        values = _enum_values(body)
        if values is not None:
            annotation = cast(Any, Literal)[tuple(values)]
            source = f"Literal[{', '.join(repr(v) for v in values)}]"
            kind = "enum"
        elif len(non_null_names) == 1:
            annotation, source, kind = _resolve_declared(non_null_names[0], body)
        else:
            annotation, source, kind = _union_of([(*_resolve_declared(name, body), body) for name in non_null_names])

    # A null default declares the option optional, as a nullable type does.
    if nullable or ("default" in body and body["default"] is None):
        annotation = annotation | None
        source = f"{source} | None"
    return annotation, source, kind


def _collect_constraints(body: dict[str, Any], kind: str) -> dict[str, Any]:
    """Translate JSON Schema validation keywords into Pydantic Field kwargs for the given value kind."""
    constraints: dict[str, Any] = {}
    if kind == "number":
        for keyword_name, kwarg in _NUMERIC_CONSTRAINTS.items():
            value = body.get(keyword_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                constraints[kwarg] = value
        # Draft 4 spells exclusive bounds as booleans that modify minimum / maximum.
        if body.get("exclusiveMinimum") is True and "ge" in constraints:
            constraints["gt"] = constraints.pop("ge")
        if body.get("exclusiveMaximum") is True and "le" in constraints:
            constraints["lt"] = constraints.pop("le")
    elif kind == "string":
        for keyword_name, kwarg in _STRING_CONSTRAINTS.items():
            value = body.get(keyword_name)
            if kwarg == "pattern" and isinstance(value, str):
                constraints[kwarg] = value
            elif kwarg != "pattern" and isinstance(value, int) and not isinstance(value, bool):
                constraints[kwarg] = value
    elif kind == "array":
        for keyword_name, kwarg in _ARRAY_CONSTRAINTS.items():
            value = body.get(keyword_name)
            if isinstance(value, int) and not isinstance(value, bool):
                constraints[kwarg] = value
    return constraints


def parse_option(name: str, body: dict[str, Any], issues: list[str] | None = None) -> OptionSpec:
    """Translate one user_option, a JSON Schema property as chap-core reads it, into an OptionSpec.

    Supports `type` (integer, number, string, boolean, array with `items`, object, or a
    list including "null"), `enum`, `minimum` / `maximum` / `exclusiveMinimum` /
    `exclusiveMaximum`, `minLength` / `maxLength` / `pattern`, `minItems` / `maxItems`,
    `description`, `title` and `default`. The default is checked against the declared
    type and constraints; a default that does not fit is reported through `issues` and
    kept as written, so MLprojects that ran before keep running.
    """
    field_name = python_field_name(name)
    annotation, type_source, kind = _resolve_type(body)
    constraints = _collect_constraints(body, kind)

    required = "default" not in body
    default: Any = None
    if not required and body["default"] is not None:
        default = _validate_default(name, body["default"], annotation, type_source, constraints, issues)

    return OptionSpec(
        name=name,
        field_name=field_name,
        annotation=annotation,
        type_source=type_source,
        required=required,
        default=default,
        description=_clean_text(body.get("description")),
        title=_clean_text(body.get("title")),
        constraints=constraints,
    )


def _validate_default(
    name: str,
    raw_default: Any,
    annotation: Any,
    type_source: str,
    constraints: dict[str, Any],
    issues: list[str] | None,
) -> Any:
    """Return the default validated against its option's type, or the best-effort coerced value with an issue."""
    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[annotation, Field(**constraints)])
    try:
        return adapter.validate_python(raw_default)
    except ValidationError as error:
        reason = error.errors()[0].get("msg", str(error))
    if annotation in _TYPE_SOURCE:
        # Same coercion chapkit applied before typed options existed (e.g. 5 -> "5" for a string).
        coerced = _coerce_default(raw_default, annotation)
        try:
            return adapter.validate_python(coerced)
        except ValidationError:
            pass
    if issues is not None:
        issues.append(
            f"user_option {name!r} default {raw_default!r} does not match its declared type {type_source} "
            f"({reason}); keeping it as written"
        )
    return raw_default


def parse_options(mlproject: MLProject, issues: list[str] | None = None) -> list[OptionSpec]:
    """Translate every user_option of an MLproject, rejecting names that collide once normalized."""
    specs: list[OptionSpec] = []
    seen: set[str] = set()
    for opt_name, opt_body in mlproject.user_options.items():
        spec = parse_option(opt_name, opt_body, issues)
        if spec.field_name in seen:
            raise MLProjectError(
                f"user_option {opt_name!r} normalizes to {spec.field_name!r}, which collides with an earlier option"
            )
        seen.add(spec.field_name)
        specs.append(spec)
    return specs


def default_prediction_periods(mlproject: MLProject) -> int:
    """Return the default forecast horizon, clamped into the bounds the MLproject declares."""
    periods = DEFAULT_PREDICTION_PERIODS
    if mlproject.min_prediction_periods is not None:
        periods = max(periods, mlproject.min_prediction_periods)
    if mlproject.max_prediction_periods is not None:
        periods = min(periods, mlproject.max_prediction_periods)
    return periods


def build_config_schema(mlproject: MLProject, issues: list[str] | None = None) -> type[BaseConfig]:
    """Build a typed BaseConfig subclass from MLproject user_options, injecting prediction_periods.

    Non-identifier option names (hyphens, keywords) are normalized to valid Python
    identifiers. The original name is preserved via Pydantic `Field(alias=...)` so POSTed
    configs and emitted config.yml keep the wire contract the MLproject declared.
    """
    fields: dict[str, Any] = {
        spec.field_name: (spec.annotation, spec.field_info()) for spec in parse_options(mlproject, issues)
    }

    if "prediction_periods" not in fields:
        fields["prediction_periods"] = (
            int,
            Field(
                default=default_prediction_periods(mlproject),
                description="Number of periods to predict into the future.",
            ),
        )

    model_name = f"{_python_class_name(mlproject.name)}Config"
    # The docstring becomes the schema description, as it would for a hand-written config class.
    return create_model(model_name, __base__=BaseConfig, __doc__=f"Configuration for {mlproject.name}.", **fields)


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


def normalize_period_type(raw: str | None, issues: list[str] | None = None) -> str:
    """Map an MLproject supported_period_type (chap-core or chapkit spelling) to a chapkit PeriodType value."""
    if raw is None or not raw.strip():
        return DEFAULT_PERIOD_TYPE
    normalized = _PERIOD_TYPES.get(raw.strip().lower())
    if normalized is None:
        if issues is not None:
            issues.append(
                f"supported_period_type {raw!r} is not one of week, month or any; using {DEFAULT_PERIOD_TYPE!r}"
            )
        return DEFAULT_PERIOD_TYPE
    return normalized


def _validated_text(
    value: str | None,
    adapter: TypeAdapter[Any],
    key: str,
    expected: str,
    issues: list[str] | None,
) -> str | None:
    """Return value when the adapter accepts it, otherwise record why it was dropped and return None."""
    if value is None:
        return None
    try:
        adapter.validate_python(value)
    except ValidationError:
        if issues is not None:
            issues.append(f"{key} {value!r} is not a valid {expected}; ignoring it")
        return None
    return value


def build_model_metadata(mlproject: MLProject, issues: list[str] | None = None) -> ModelMetadata:
    """Build the ModelMetadata block (authors, organization, citation, ...) from an MLproject's meta_data.

    Values that would fail MLServiceInfo validation (a non-URL logo, a malformed email, an
    unknown assessed status) are dropped and reported through `issues` instead of stopping
    the service from starting.
    """
    from pydantic import EmailStr, HttpUrl

    from chapkit.api.service_builder import AssessedStatus, ModelMetadata

    meta = mlproject.meta_data
    url_adapter: TypeAdapter[Any] = TypeAdapter(HttpUrl)
    email_adapter: TypeAdapter[Any] = TypeAdapter(EmailStr)

    def text(key: str) -> str | None:
        """Return a meta_data value as text."""
        return _clean_text(meta.get(key))

    def url(key: str) -> str | None:
        """Return a meta_data value when it is an http(s) URL."""
        return _validated_text(text(key), url_adapter, f"meta_data.{key}", "http(s) URL", issues)

    assessed_status: AssessedStatus | None = None
    raw_status = text("author_assessed_status")
    if raw_status is not None:
        try:
            assessed_status = AssessedStatus(raw_status.lower())
        except ValueError:
            if issues is not None:
                allowed = ", ".join(status.value for status in AssessedStatus)
                issues.append(f"meta_data.author_assessed_status {raw_status!r} is not one of {allowed}; ignoring it")

    repository_url = url("repository_url") or url("source_url")
    if repository_url is None and mlproject.source_url is not None:
        repository_url = _validated_text(mlproject.source_url, url_adapter, "source_url", "http(s) URL", issues)

    return ModelMetadata.model_validate(
        {
            "author": text("author"),
            "author_note": text("author_note") or text("description"),
            "author_assessed_status": assessed_status,
            "contact_email": _validated_text(
                text("contact_email"), email_adapter, "meta_data.contact_email", "email address", issues
            ),
            "organization": text("organization"),
            "organization_logo_url": url("organization_logo_url"),
            "citation_info": text("citation_info"),
            "repository_url": repository_url,
            "documentation_url": url("documentation_url"),
        }
    )


def build_ml_service_info(mlproject: MLProject, issues: list[str] | None = None) -> MLServiceInfo:
    """Build the full MLServiceInfo contract (identity, metadata, capabilities) served on /api/v1/info."""
    from chapkit.api.service_builder import MLServiceInfo

    info: dict[str, Any] = {
        "id": slugify(mlproject.name),
        "display_name": _clean_text(mlproject.meta_data.get("display_name")) or mlproject.name,
        "description": _clean_text(mlproject.meta_data.get("description"))
        or f"Chapkit service for the {mlproject.name} MLproject",
        "model_metadata": build_model_metadata(mlproject, issues),
        "period_type": normalize_period_type(mlproject.supported_period_type, issues),
        "required_covariates": list(mlproject.required_covariates),
        "allow_free_additional_continuous_covariates": mlproject.allow_free_additional_continuous_covariates,
        "requires_geo": mlproject.requires_geo,
    }
    if mlproject.version is not None:
        info["version"] = mlproject.version
    if mlproject.min_prediction_periods is not None:
        info["min_prediction_periods"] = mlproject.min_prediction_periods
    if mlproject.max_prediction_periods is not None:
        info["max_prediction_periods"] = mlproject.max_prediction_periods
    return MLServiceInfo.model_validate(info)
