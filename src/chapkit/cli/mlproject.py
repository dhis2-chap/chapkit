"""Parse MLproject files and adapt them to chapkit's ShellModelRunner conventions."""

from __future__ import annotations

import keyword
import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, cast

import yaml
from pydantic import AfterValidator, BaseModel, Field, TypeAdapter, ValidationError, WithJsonSchema, create_model
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
# JSON Schema's own semantics (e.g. `minimum` has no effect on a string). Array lengths
# (`minItems` / `maxItems`) are mapped in `_member_for`.
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
    adapters: dict[str, str] = Field(default_factory=dict)
    hpo_search_space: dict[str, Any] | None = None
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

    raw_adapters = raw.get("adapters") or {}
    adapters: dict[str, str] = {}
    if isinstance(raw_adapters, dict):
        adapters = {str(to_name): str(from_name) for to_name, from_name in raw_adapters.items()}
    else:
        parse_warnings.append(f"adapters must be a mapping of column names, got {raw_adapters!r}; ignoring it")

    raw_search_space = raw.get("hpo_search_space")
    hpo_search_space: dict[str, Any] | None = None
    if isinstance(raw_search_space, dict):
        hpo_search_space = dict(raw_search_space)
    elif raw_search_space is not None:
        parse_warnings.append(f"hpo_search_space must be a mapping, got {raw_search_space!r}; ignoring it")

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
        adapters=adapters,
        hpo_search_space=hpo_search_space,
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

    @property
    def alias(self) -> str | None:
        """Return the MLproject option name when it differs from the Python field name."""
        return self.name if self.name != self.field_name else None

    def field_info(self) -> FieldInfo:
        """Return the Pydantic field definition for this option."""
        kwargs: dict[str, Any] = {}
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


# MLproject type names (JSON Schema plus the aliases chapkit always accepted) to canonical JSON Schema names.
# Unknown names fall back to "string", as they did before typed options existed.
_CANONICAL_TYPES: dict[str, str] = {
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "str": "string",
    "path": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "object": "object",
    "null": "null",
}
_LOWER_BOUNDS: tuple[str, ...] = ("minimum", "exclusiveMinimum", "minLength", "minItems")
_UPPER_BOUNDS: tuple[str, ...] = ("maximum", "exclusiveMaximum", "maxLength", "maxItems")
_SCALAR_TYPES: dict[str, type] = {"integer": int, "number": float, "string": str, "boolean": bool}


@dataclass(slots=True)
class _Schema:
    """The validation keywords that apply at one point of a user_option schema, already intersected."""

    types: list[str] | None = None
    bounds: dict[str, int | float] = field(default_factory=dict)
    patterns: list[str] = field(default_factory=list)
    enum: list[Any] | None = None
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class _Member:
    """One non-null alternative of an option's type, with the Field kwargs that constrain it."""

    annotation: Any
    source: str
    constraints: dict[str, Any] = field(default_factory=dict)
    extra_patterns: list[str] = field(default_factory=list)
    scalar: type | None = None

    def constrained(self) -> Any:
        """Return the annotation with its constraints attached, for use inside a union."""
        metadata: list[Any] = []
        if self.constraints:
            metadata.append(Field(**self.constraints))
        if self.extra_patterns:
            metadata.append(AfterValidator(_pattern_checker(self.extra_patterns)))
        return Annotated[self.annotation, *metadata] if metadata else self.annotation


def _pattern_checker(patterns: list[str]) -> Callable[[str], str]:
    """Return a validator requiring every pattern to match, for strings under more than one `pattern`."""
    compiled = [re.compile(pattern) for pattern in patterns]

    def check(value: str) -> str:
        """Reject a string that does not match all patterns."""
        for pattern in compiled:
            if not pattern.search(value):
                raise ValueError(f"String should match pattern '{pattern.pattern}'")
        return value

    return check


def _type_names(body: dict[str, Any]) -> list[str] | None:
    """Return the canonical type names a schema declares, or None when it declares no type."""
    declared = body.get("type")
    if declared is None:
        return None
    names = declared if isinstance(declared, list) else [declared]
    return list(dict.fromkeys(_CANONICAL_TYPES.get(str(name).lower(), "string") for name in names))


def _normalize(body: dict[str, Any]) -> _Schema:
    """Read the validation keywords of one schema object."""
    bounds: dict[str, int | float] = {}
    for key in _LOWER_BOUNDS + _UPPER_BOUNDS:
        value = body.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            bounds[key] = value
    # Draft 4 spells exclusive bounds as booleans that modify minimum / maximum.
    if body.get("exclusiveMinimum") is True and "minimum" in bounds:
        bounds["exclusiveMinimum"] = bounds.pop("minimum")
    if body.get("exclusiveMaximum") is True and "maximum" in bounds:
        bounds["exclusiveMaximum"] = bounds.pop("maximum")
    pattern = body.get("pattern")
    enum = body.get("enum")
    items = body.get("items")
    return _Schema(
        types=_type_names(body),
        bounds=bounds,
        patterns=[pattern] if isinstance(pattern, str) else [],
        enum=list(enum) if isinstance(enum, list) else None,
        items=[items if isinstance(items, dict) else {"type": items}] if items is not None else [],
    )


def _same_value(left: Any, right: Any) -> bool:
    """Compare JSON values the way JSON Schema does, so True is not 1."""
    return type(left) is type(right) and left == right


def _intersect_types(left: list[str] | None, right: list[str] | None) -> list[str] | None:
    """Return the types allowed by both declarations (None means any type)."""
    if left is None or right is None:
        return right if left is None else left
    allowed: list[str] = []
    for name in left:
        for other in right:
            if name == other:
                allowed.append(name)
            elif {name, other} == {"integer", "number"}:
                allowed.append("integer")
    return list(dict.fromkeys(allowed))


def _merge(parent: _Schema, child: _Schema) -> _Schema:
    """Intersect two schemas: a value must satisfy both, so every bound takes its stricter side."""
    bounds = dict(parent.bounds)
    for key, value in child.bounds.items():
        if key not in bounds:
            bounds[key] = value
        elif key in _LOWER_BOUNDS:
            bounds[key] = max(bounds[key], value)
        else:
            bounds[key] = min(bounds[key], value)
    if parent.enum is None or child.enum is None:
        enum = child.enum if parent.enum is None else parent.enum
    else:
        enum = [value for value in parent.enum if any(_same_value(value, other) for other in child.enum)]
    return _Schema(
        types=_intersect_types(parent.types, child.types),
        bounds=bounds,
        patterns=parent.patterns + [pattern for pattern in child.patterns if pattern not in parent.patterns],
        enum=enum,
        items=parent.items + child.items,
    )


def _fits_type(value: Any, name: str) -> bool:
    """Return True when a JSON value is an instance of the named JSON Schema type."""
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    return name == "object" and isinstance(value, dict)


def _fits_schema(value: Any, schema: _Schema) -> bool:
    """Return True when an enum value satisfies the types and bounds that apply beside the enum."""
    if schema.types is not None and not any(_fits_type(value, name) for name in schema.types):
        return False
    bounds = schema.bounds
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in bounds and value < bounds["minimum"]:
            return False
        if "maximum" in bounds and value > bounds["maximum"]:
            return False
        if "exclusiveMinimum" in bounds and value <= bounds["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in bounds and value >= bounds["exclusiveMaximum"]:
            return False
    if isinstance(value, str):
        if "minLength" in bounds and len(value) < bounds["minLength"]:
            return False
        if "maxLength" in bounds and len(value) > bounds["maxLength"]:
            return False
        if not all(re.search(pattern, value) for pattern in schema.patterns):
            return False
    return True


def _member_for(name: str, schema: _Schema) -> _Member:
    """Build the alternative for one non-null type name, with the bounds that apply to it."""
    bounds = schema.bounds
    constraints: dict[str, Any] = {}
    if name == "array":
        item_annotation, item_source = _resolve_items(schema.items)
        for key, kwarg in (("minItems", "min_length"), ("maxItems", "max_length")):
            if key in bounds:
                constraints[kwarg] = bounds[key]
        return _Member(list[item_annotation], f"list[{item_source}]", constraints)  # type: ignore[valid-type]
    if name == "object":
        return _Member(dict[str, Any], "dict[str, Any]")
    scalar = _SCALAR_TYPES[name]
    extra_patterns: list[str] = []
    if name in ("integer", "number"):
        constraints = {kwarg: bounds[key] for key, kwarg in _NUMERIC_CONSTRAINTS.items() if key in bounds}
    elif name == "string":
        constraints = {kwarg: bounds[key] for key, kwarg in _STRING_CONSTRAINTS.items() if key in bounds}
        if schema.patterns:
            constraints["pattern"] = schema.patterns[0]
            extra_patterns = schema.patterns[1:]
    return _Member(scalar, _TYPE_SOURCE[scalar], constraints, extra_patterns, scalar)


def _resolve_items(item_schemas: list[dict[str, Any]]) -> tuple[Any, str]:
    """Map the `items` declarations that apply to an array (all must hold) to (python type, source).

    Intersecting several item schemas is only done when none of them has branches; otherwise
    the item type stays `Any` (never stricter than the schema) and the option's JSON Schema
    validator enforces the items exactly.
    """
    if not item_schemas:
        return Any, "Any"
    if len(item_schemas) > 1 and any(_has_combinator(item_schema) for item_schema in item_schemas):
        return Any, "Any"
    inherited: _Schema | None = None
    for item_schema in item_schemas[:-1]:
        normalized = _normalize(item_schema)
        inherited = normalized if inherited is None else _merge(inherited, normalized)
    members, nullable = _resolve(item_schemas[-1], inherited)
    annotation, source = _combine(members, nullable)
    return annotation, source


def _resolve(body: dict[str, Any], inherited: _Schema | None = None) -> tuple[list[_Member], bool]:
    """Resolve a schema into its non-null alternatives and whether null is allowed.

    Keywords beside `anyOf` / `oneOf` are intersected into every branch, so a branch
    inherits the parent's type, bounds, items and enum and can only tighten them. A
    branch whose type cannot meet the parent's (e.g. `null` under `type: integer`) is
    dropped. An enum keeps only the values that fit the types and bounds beside it.
    """
    own = _normalize(body)
    schema = own if inherited is None else _merge(inherited, own)

    subschemas = body.get("anyOf") or body.get("oneOf")
    if isinstance(subschemas, list) and subschemas and all(isinstance(sub, dict) for sub in subschemas):
        members: list[_Member] = []
        nullable = False
        for subschema in subschemas:
            branch_members, branch_nullable = _resolve(subschema, schema)
            members.extend(branch_members)
            nullable = nullable or branch_nullable
        return members, nullable

    if schema.enum is not None and all(
        value is None or isinstance(value, (str, int, float, bool)) for value in schema.enum
    ):
        allowed = [value for value in schema.enum if _fits_schema(value, schema)]
        values = [value for value in allowed if value is not None]
        literal_members = (
            [_Member(cast(Any, Literal)[tuple(values)], f"Literal[{', '.join(repr(v) for v in values)}]")]
            if values
            else []
        )
        return literal_members, any(value is None for value in allowed)

    # No type anywhere means a string, as for options declared before typed options existed.
    types = schema.types if schema.types is not None else ["string"]
    return [_member_for(name, schema) for name in types if name != "null"], "null" in types


def _combine(members: list[_Member], nullable: bool) -> tuple[Any, str]:
    """Join resolved alternatives into one annotation and its source spelling."""
    if not members:
        annotation: Any = type(None) if nullable else str
        return annotation, "None" if nullable else "str"
    if len(members) == 1:
        annotation = members[0].constrained()
    else:
        annotation = cast(Any, Union)[tuple(member.constrained() for member in members)]
    source = " | ".join(member.source for member in members)
    if nullable:
        annotation = annotation | None
        source = f"{source} | None"
    return annotation, source


# Keywords the Pydantic translation reproduces exactly, in validation and in the published schema.
_TRANSLATED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "enum",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "items",
        "anyOf",
        "title",
        "description",
        "default",
        "examples",
        "$comment",
    }
)
_COMBINATORS: tuple[str, ...] = ("anyOf", "oneOf", "allOf", "not")


def _subschemas(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the schema objects nested directly in a schema (branches and items)."""
    nested: list[Any] = []
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(body.get(key), list):
            nested.extend(body[key])
    for key in ("not", "items"):
        if key in body:
            nested.append(body[key])
    return [sub for sub in nested if isinstance(sub, dict)]


def _has_combinator(body: Any) -> bool:
    """Return True when a schema or any schema nested in it has anyOf / oneOf / allOf / not."""
    if not isinstance(body, dict):
        return False
    return any(key in body for key in _COMBINATORS) or any(_has_combinator(sub) for sub in _subschemas(body))


def _pattern_count(body: Any) -> int:
    """Count the `pattern` keywords in a schema and the schemas nested in it."""
    if not isinstance(body, dict):
        return 0
    return int(isinstance(body.get("pattern"), str)) + sum(_pattern_count(sub) for sub in _subschemas(body))


def _translated_exactly(body: Any) -> bool:
    """Return True when the Pydantic type publishes the same JSON Schema the MLproject declares."""
    if not isinstance(body, dict):
        return True
    if set(body) - _TRANSLATED_KEYWORDS:
        return False
    branches = [branch for branch in body.get("anyOf", []) if isinstance(branch, dict)]
    item_schemas = [body["items"]] if "items" in body else []
    item_schemas += [branch["items"] for branch in branches if "items" in branch]
    if len(item_schemas) > 1 and any(_has_combinator(item_schema) for item_schema in item_schemas):
        return False
    return all(_translated_exactly(sub) for sub in _subschemas(body))


def _canonical_schema(body: Any) -> Any:
    """Return a copy of a schema in standard JSON Schema, read the way the translation reads it.

    chapkit's legacy type aliases (`int`, `str`, `path`, ...) and unknown type names become
    their JSON Schema types, and draft 4 boolean `exclusiveMinimum` / `exclusiveMaximum`
    become the numeric form, so MLprojects that ran before stay valid schemas.
    """
    if not isinstance(body, dict):
        return body
    schema = dict(body)
    declared = schema.get("type")
    if isinstance(declared, list):
        schema["type"] = list(dict.fromkeys(_CANONICAL_TYPES.get(str(name).lower(), "string") for name in declared))
    elif declared is not None:
        schema["type"] = _CANONICAL_TYPES.get(str(declared).lower(), "string")
    for exclusive, inclusive in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
        if isinstance(schema.get(exclusive), bool):
            if schema.pop(exclusive) and inclusive in schema:
                schema[exclusive] = schema.pop(inclusive)
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            schema[key] = [_canonical_schema(sub) for sub in schema[key]]
    for key in ("not", "items"):
        if key in schema:
            schema[key] = _canonical_schema(schema[key] if isinstance(schema[key], dict) else {"type": schema[key]})
    return schema


def _json_schema_check(name: str, body: dict[str, Any], issues: list[str] | None) -> Callable[[Any], Any] | None:
    """Return a validator enforcing the option's own JSON Schema, as chap-core validates user options.

    Returns None, with an issue, when the option is not a valid JSON Schema (chap-core could
    not validate it either); the translated Pydantic type then validates on its own. A null
    default keeps the option optional, so None is accepted for it as before typed options.
    """
    from jsonschema.exceptions import SchemaError, best_match
    from jsonschema.validators import validator_for

    body = _canonical_schema(body)
    validator_class = validator_for(body)
    try:
        validator_class.check_schema(body)
    except SchemaError as error:
        if issues is not None:
            issues.append(
                f"user_option {name!r} is not a valid JSON Schema ({error.message}); "
                "it is validated by its translated type only"
            )
        return None
    validator = validator_class(body)
    none_by_default = "default" in body and body["default"] is None

    def check(value: Any) -> Any:
        """Raise when the value does not satisfy the option's JSON Schema."""
        if value is None and none_by_default:
            return value
        error = best_match(validator.iter_errors(value))
        if error is not None:
            raise ValueError(error.message)
        return value

    return check


def parse_option(name: str, body: dict[str, Any], issues: list[str] | None = None) -> OptionSpec:
    """Translate one user_option, a JSON Schema property as chap-core reads it, into an OptionSpec.

    Supports `type` (integer, number, string, boolean, array with `items`, object, or a
    list including "null"), `anyOf` / `oneOf`, `enum`, `minimum` / `maximum` /
    `exclusiveMinimum` / `exclusiveMaximum`, `minLength` / `maxLength` / `pattern`,
    `minItems` / `maxItems`, `description`, `title` and `default` into a typed annotation
    (see `_resolve`). Values are also validated against the option's own JSON Schema with
    the `jsonschema` validator chap-core uses, so every keyword (including `oneOf`,
    `allOf`, `not`, `const`, ...) is enforced exactly as chap-core enforces it. The default
    is checked the same way; a default that does not fit is reported through `issues` and
    kept as written, so MLprojects that ran before keep running.
    """
    field_name = python_field_name(name)
    members, nullable = _resolve(body)
    # A null default declares the option optional, as a nullable type does.
    nullable = nullable or ("default" in body and body["default"] is None)

    scalar = members[0].scalar if len(members) == 1 and not nullable else None
    annotation, type_source = _combine(members, nullable)

    # The Pydantic type is never stricter than the schema; the option's own JSON Schema,
    # checked with the validator chap-core uses, makes acceptance exact. Keywords the
    # type cannot express (oneOf, allOf, not, const, ...) are published as declared.
    check = _json_schema_check(name, body, issues)
    metadata: list[Any] = []
    if check is not None:
        metadata.append(AfterValidator(check))
        if not _translated_exactly(body):
            published = {
                key: value
                for key, value in _canonical_schema(body).items()
                if key not in ("title", "description", "default")
            }
            metadata.append(WithJsonSchema(published))
    if metadata:
        annotation = Annotated[annotation, *metadata]

    required = "default" not in body
    default: Any = None
    if not required and body["default"] is not None:
        default = _validate_default(name, body["default"], annotation, type_source, {}, scalar, issues)

    return OptionSpec(
        name=name,
        field_name=field_name,
        annotation=annotation,
        type_source=type_source,
        required=required,
        default=default,
        description=_clean_text(body.get("description")),
        title=_clean_text(body.get("title")),
    )


def _validate_default(
    name: str,
    raw_default: Any,
    annotation: Any,
    type_source: str,
    constraints: dict[str, Any],
    scalar: type | None,
    issues: list[str] | None,
) -> Any:
    """Return the default validated against its option's type, or the best-effort coerced value with an issue."""
    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[annotation, Field(**constraints)])
    try:
        return adapter.validate_python(raw_default)
    except ValidationError as error:
        reason = error.errors()[0].get("msg", str(error))
    if scalar is not None:
        # Same coercion chapkit applied before typed options existed (e.g. 5 -> "5" for a string).
        coerced = _coerce_default(raw_default, scalar)
        try:
            return adapter.validate_python(coerced)
        except ValidationError:
            pass
    if issues is not None:
        issues.append(
            f"user_option {name!r} default {raw_default!r} does not match its declared type {type_source} "
            f"({reason}); keeping it as written, so configs must set {name!r} explicitly"
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
        "target": mlproject.target or "disease_cases",
        "hpo_search_space": mlproject.hpo_search_space,
    }
    if mlproject.version is not None:
        info["version"] = mlproject.version
    if mlproject.min_prediction_periods is not None:
        info["min_prediction_periods"] = mlproject.min_prediction_periods
    if mlproject.max_prediction_periods is not None:
        info["max_prediction_periods"] = mlproject.max_prediction_periods
    return MLServiceInfo.model_validate(info)
