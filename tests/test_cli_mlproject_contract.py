"""Tests for the model contract and typed user_options that chapkit mlproject run reads from an MLproject."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import Field, TypeAdapter, ValidationError
from typer.testing import CliRunner

from chapkit.cli.cli import app
from chapkit.cli.mlproject import (
    build_config_schema,
    build_ml_service_info,
    normalize_period_type,
    parse_mlproject,
    parse_option,
    resolve_mlproject,
)
from chapkit.cli.run import _check_port_available, build_mlproject_app

ENTRY_POINTS = """
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""

CONTRACT_MLPROJECT = (
    """
name: contract_model
version: 2.1.0
supported_period_type: week
required_covariates: [population]
allow_free_additional_continuous_covariates: true
requires_geo: true
min_prediction_length: 2
maxPredictionPeriods: 8
source_url: https://github.com/chap-models/contract_model
meta_data:
  display_name: Contract Model
  description: Model with a full contract.
  author: CHAP team
  author_note: Use with care.
  author_assessed_status: orange
  contact_email: chap@example.org
  organization: HISP Centre, University of Oslo
  organization_logo_url: https://example.org/logo.png
  citation_info: Cite me.
  documentation_url: https://example.org/docs
"""
    + ENTRY_POINTS
)

USER_OPTIONS = """
user_options:
  n_lags:
    type: [array, integer]
    items:
      type: integer
    default: [3]
    description: Lags per covariate, or one value for all.
  cell:
    title: Recurrent cell type
    type: string
    enum: [gru, simple]
    default: gru
  learning_rate:
    type: number
    minimum: 0
    exclusiveMaximum: 1
    default: 0.01
  candidate_lags:
    type: array
    items:
      type: integer
      minimum: 1
    minItems: 1
    default: [7, 10, 12]
  max_epochs:
    anyOf:
    - type: integer
      minimum: 1
    - type: 'null'
    default: null
  label:
    type: string
    pattern: "^[a-z]+$"
    default: abc
  settings:
    type: object
    default: {}
"""

TYPED_MLPROJECT = "name: typed_model\n" + USER_OPTIONS + ENTRY_POINTS


def _write_mlproject(tmp_path: Path, contents: str) -> Path:
    target = tmp_path / "MLproject"
    target.write_text(contents)
    return target


def test_parse_reads_horizon_bounds_under_chap_core_aliases(tmp_path: Path) -> None:
    mlproject = parse_mlproject(_write_mlproject(tmp_path, CONTRACT_MLPROJECT))
    assert mlproject.min_prediction_periods == 2
    assert mlproject.max_prediction_periods == 8
    assert mlproject.version == "2.1.0"
    assert mlproject.parse_warnings == []


@pytest.mark.parametrize(
    ("bounds", "expected_warning"),
    [
        ("min_prediction_periods: -1", "non-negative integer"),
        ("max_prediction_periods: soon", "non-negative integer"),
        ("min_prediction_periods: 5\nmax_prediction_periods: 2", "greater than"),
    ],
)
def test_parse_ignores_invalid_horizon_bounds_with_a_warning(
    tmp_path: Path, bounds: str, expected_warning: str
) -> None:
    mlproject = parse_mlproject(_write_mlproject(tmp_path, f"name: bad_bounds\n{bounds}\n{ENTRY_POINTS}"))
    assert mlproject.min_prediction_periods is None
    assert mlproject.max_prediction_periods is None
    assert any(expected_warning in warning for warning in mlproject.parse_warnings)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("week", "weekly"),
        ("weekly", "weekly"),
        ("month", "monthly"),
        ("Monthly", "monthly"),
        ("any", "any"),
        (None, "any"),
        ("", "any"),
    ],
)
def test_normalize_period_type(raw: str | None, expected: str) -> None:
    issues: list[str] = []
    assert normalize_period_type(raw, issues) == expected
    assert issues == []


def test_normalize_period_type_unknown_value_falls_back_to_any_with_a_warning() -> None:
    issues: list[str] = []
    assert normalize_period_type("daily", issues) == "any"
    assert "daily" in issues[0]


def test_build_ml_service_info_maps_full_contract(tmp_path: Path) -> None:
    issues: list[str] = []
    info = build_ml_service_info(parse_mlproject(_write_mlproject(tmp_path, CONTRACT_MLPROJECT)), issues)

    assert issues == []
    assert info.id == "contract-model"
    assert info.display_name == "Contract Model"
    assert info.description == "Model with a full contract."
    assert info.version == "2.1.0"
    assert info.period_type.value == "weekly"
    assert info.min_prediction_periods == 2
    assert info.max_prediction_periods == 8
    assert info.required_covariates == ["population"]
    assert info.allow_free_additional_continuous_covariates is True
    assert info.requires_geo is True

    metadata = info.model_metadata
    assert metadata.author == "CHAP team"
    assert metadata.author_note == "Use with care."
    assert metadata.author_assessed_status is not None
    assert metadata.author_assessed_status.value == "orange"
    assert metadata.contact_email == "chap@example.org"
    assert metadata.organization == "HISP Centre, University of Oslo"
    assert str(metadata.organization_logo_url) == "https://example.org/logo.png"
    assert metadata.citation_info == "Cite me."
    assert str(metadata.repository_url) == "https://github.com/chap-models/contract_model"
    assert str(metadata.documentation_url) == "https://example.org/docs"


def test_build_ml_service_info_minimal_mlproject_uses_defaults(tmp_path: Path) -> None:
    info = build_ml_service_info(parse_mlproject(_write_mlproject(tmp_path, f"name: bare\n{ENTRY_POINTS}")))
    assert info.display_name == "bare"
    assert info.description == "Chapkit service for the bare MLproject"
    assert info.period_type.value == "any"
    assert info.min_prediction_periods == 0
    assert info.max_prediction_periods == 100
    assert info.model_metadata.author is None
    assert info.model_metadata.author_assessed_status is None


def test_build_ml_service_info_drops_invalid_metadata_with_warnings(tmp_path: Path) -> None:
    contents = f"""
name: sloppy_meta
meta_data:
  description: Falls back into author_note.
  author_assessed_status: magenta
  contact_email: not-an-email
  organization_logo_url: /local/logo.png
{ENTRY_POINTS}
"""
    issues: list[str] = []
    info = build_ml_service_info(parse_mlproject(_write_mlproject(tmp_path, contents)), issues)

    metadata = info.model_metadata
    assert metadata.author_assessed_status is None
    assert metadata.contact_email is None
    assert metadata.organization_logo_url is None
    assert metadata.author_note == "Falls back into author_note."
    assert len(issues) == 3
    assert any("author_assessed_status" in issue for issue in issues)
    assert any("contact_email" in issue for issue in issues)
    assert any("organization_logo_url" in issue for issue in issues)


def test_typed_options_json_schema_matches_the_mlproject(tmp_path: Path) -> None:
    schema: Any = build_config_schema(parse_mlproject(_write_mlproject(tmp_path, TYPED_MLPROJECT)))
    properties = schema.model_json_schema()["properties"]

    assert schema.model_json_schema()["description"] == "Configuration for typed_model."
    assert properties["n_lags"]["anyOf"] == [{"items": {"type": "integer"}, "type": "array"}, {"type": "integer"}]
    assert properties["n_lags"]["description"] == "Lags per covariate, or one value for all."
    assert properties["cell"]["enum"] == ["gru", "simple"]
    assert properties["cell"]["title"] == "Recurrent cell type"
    assert properties["learning_rate"]["minimum"] == 0
    assert properties["learning_rate"]["exclusiveMaximum"] == 1
    assert properties["candidate_lags"]["minItems"] == 1
    assert properties["candidate_lags"]["items"] == {"minimum": 1, "type": "integer"}
    assert properties["max_epochs"]["anyOf"] == [{"minimum": 1, "type": "integer"}, {"type": "null"}]
    assert properties["label"]["pattern"] == "^[a-z]+$"
    assert properties["settings"]["type"] == "object"


def test_typed_options_validate_values(tmp_path: Path) -> None:
    schema: Any = build_config_schema(parse_mlproject(_write_mlproject(tmp_path, TYPED_MLPROJECT)))

    defaults = schema()
    assert defaults.n_lags == [3]
    assert defaults.candidate_lags == [7, 10, 12]
    assert defaults.max_epochs is None

    accepted = schema(n_lags=2, cell="simple", learning_rate=0.5, max_epochs=10, settings={"a": 1})
    assert accepted.n_lags == 2
    assert accepted.max_epochs == 10

    bad_values: list[dict[str, Any]] = [
        {"cell": "lstm"},
        {"learning_rate": 1},
        {"learning_rate": -0.1},
        {"candidate_lags": []},
        {"candidate_lags": [0]},
        {"max_epochs": 0},
        {"label": "ABC"},
        {"n_lags": "three"},
    ]
    for bad in bad_values:
        with pytest.raises(ValidationError):
            schema(**bad)


def test_typed_options_accept_chap_core_user_option_values(tmp_path: Path) -> None:
    """chap-core creates configs with user options nested under user_option_values."""
    schema: Any = build_config_schema(parse_mlproject(_write_mlproject(tmp_path, TYPED_MLPROJECT)))
    instance = schema.model_validate({"user_option_values": {"cell": "simple", "n_lags": [1, 2]}})
    assert instance.cell == "simple"
    assert instance.n_lags == [1, 2]


def test_option_default_that_does_not_fit_is_kept_with_a_warning() -> None:
    issues: list[str] = []
    spec = parse_option("cell", {"type": "string", "enum": ["gru", "simple"], "default": "lstm"}, issues)
    assert spec.default == "lstm"
    assert "cell" in issues[0]


def test_option_default_is_coerced_as_before_typed_options() -> None:
    issues: list[str] = []
    assert parse_option("count", {"type": "integer", "default": "10"}, issues).default == 10
    assert parse_option("label", {"type": "mystery", "default": 5}, issues).default == "5"
    assert parse_option("flag", {"type": "boolean", "default": "yes"}, issues).default is True
    assert issues == []


def test_prediction_periods_default_is_clamped_into_declared_bounds(tmp_path: Path) -> None:
    contents = f"name: short_horizon\nmax_prediction_periods: 1\n{ENTRY_POINTS}"
    schema: Any = build_config_schema(parse_mlproject(_write_mlproject(tmp_path, contents)))
    assert schema().prediction_periods == 1


def test_resolve_mlproject_accepts_file_or_directory(tmp_path: Path) -> None:
    mlproject_file = _write_mlproject(tmp_path, CONTRACT_MLPROJECT)
    assert resolve_mlproject(mlproject_file) == mlproject_file
    assert resolve_mlproject(tmp_path) == mlproject_file


def test_build_mlproject_app_serves_contract_and_typed_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_mlproject(tmp_path, CONTRACT_MLPROJECT + USER_OPTIONS)
    monkeypatch.chdir(tmp_path)
    service, _ = build_mlproject_app(parse_mlproject(tmp_path / "MLproject"))

    with TestClient(service) as client:
        info = client.get("/api/v1/info").json()
        assert info["period_type"] == "weekly"
        assert info["min_prediction_periods"] == 2
        assert info["max_prediction_periods"] == 8
        assert info["model_metadata"]["author"] == "CHAP team"
        assert info["model_metadata"]["contact_email"] == "chap@example.org"

        config_schema = client.get("/api/v1/configs/$schema").json()
        assert config_schema["properties"]["cell"]["enum"] == ["gru", "simple"]

        rejected = client.post("/api/v1/configs", json={"name": "bad", "data": {"cell": "lstm"}})
        assert rejected.status_code == 422
        created = client.post("/api/v1/configs", json={"name": "ok", "data": {"user_option_values": {"n_lags": 4}}})
        assert created.status_code == 201
        assert created.json()["data"]["n_lags"] == 4

    # Nothing is written into the MLproject directory.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["MLproject"]


def test_check_port_available_reports_port_in_use() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        problem = _check_port_available("127.0.0.1", port)
    assert problem is not None
    assert "already in use" in problem
    assert "--port" in problem


def test_check_port_available_accepts_free_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert _check_port_available("127.0.0.1", port) is None


def test_run_command_exits_when_port_is_in_use(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, CONTRACT_MLPROJECT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = CliRunner().invoke(app, ["mlproject", "run", str(tmp_path / "MLproject"), "--port", str(port)])
    assert result.exit_code == 1
    assert "already in use" in result.output


DOCUMENTED_MLPROJECT = (
    """
name: ewars_template
user_options:
  n_lags:
    type: [array, integer]
    items:
      type: integer
    default: [3]
    description: Number of lags per covariate, or one value for all.
  cell:
    title: Recurrent cell type
    type: string
    enum: [gru, simple]
    default: gru
  precision:
    type: number
    minimum: 0
    default: 0.01
  max_epochs:
    anyOf:
    - type: integer
    - type: 'null'
    default: null
"""
    + ENTRY_POINTS
)


def test_generated_schema_matches_hand_written_config_class(tmp_path: Path) -> None:
    """The runtime config class has the same JSON Schema as the equivalent hand-written main.py class."""
    from typing import Literal

    from chapkit import BaseConfig

    class EwarsTemplateConfig(BaseConfig):
        """Configuration for ewars_template."""

        n_lags: list[int] | int = Field([3], description="Number of lags per covariate, or one value for all.")
        cell: Literal["gru", "simple"] = Field("gru", title="Recurrent cell type")
        precision: float = Field(0.01, ge=0)
        max_epochs: int | None = None
        prediction_periods: int = Field(3, description="Number of periods to predict into the future.")

    generated: Any = build_config_schema(parse_mlproject(_write_mlproject(tmp_path, DOCUMENTED_MLPROJECT)))
    assert generated.model_json_schema() == EwarsTemplateConfig.model_json_schema()


def test_nullable_enum_keeps_its_non_null_members() -> None:
    spec = parse_option("mode", {"type": ["string", "null"], "enum": ["a", None], "default": "a"})
    adapter = TypeAdapter(spec.annotation)
    assert adapter.validate_python("a") == "a"
    assert adapter.validate_python(None) is None
    with pytest.raises(ValidationError):
        adapter.validate_python("b")
    assert spec.type_source == "Literal['a'] | None"


def test_constraints_beside_any_of_apply_to_each_branch() -> None:
    spec = parse_option("max_epochs", {"anyOf": [{"type": "integer"}, {"type": "null"}], "minimum": 1, "default": None})
    adapter = TypeAdapter(spec.annotation)
    assert adapter.validate_python(1) == 1
    assert adapter.validate_python(None) is None
    with pytest.raises(ValidationError):
        adapter.validate_python(0)
    assert adapter.json_schema() == {"anyOf": [{"minimum": 1, "type": "integer"}, {"type": "null"}]}


def test_branch_bounds_combine_with_bounds_beside_any_of() -> None:
    spec = parse_option(
        "size",
        {"anyOf": [{"type": "integer", "minimum": 5}, {"type": "string", "maxLength": 2}], "minimum": 1, "default": 5},
    )
    adapter = TypeAdapter(spec.annotation)
    assert adapter.validate_python(5) == 5
    assert adapter.validate_python("ab") == "ab"
    for bad in (3, "abc"):
        with pytest.raises(ValidationError):
            adapter.validate_python(bad)


def _accepts(body: dict[str, Any], value: Any) -> bool:
    spec = parse_option("option", body)
    adapter: TypeAdapter[Any] = TypeAdapter(spec.annotation)
    try:
        adapter.validate_python(value)
    except ValidationError:
        return False
    return True


def test_parent_and_branch_bounds_must_both_hold() -> None:
    body = {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "minimum": 5, "default": None}
    assert not _accepts(body, 3)
    assert _accepts(body, 5)
    assert _accepts(body, None)


def test_parent_items_apply_inside_any_of_branches() -> None:
    body = {"anyOf": [{"type": "array"}, {"type": "integer"}], "items": {"type": "integer", "minimum": 1}, "default": 1}
    assert not _accepts(body, ["bad"])
    assert not _accepts(body, [0])
    assert _accepts(body, [1, 2])
    assert _accepts(body, 3)


def test_untyped_branch_inherits_parent_type() -> None:
    body = {"type": ["integer", "null"], "anyOf": [{"minimum": 1}, {"type": "null"}], "default": None}
    assert _accepts(body, 1)
    assert _accepts(body, None)
    assert not _accepts(body, 0)
    assert not _accepts(body, "a")


def test_branch_that_cannot_meet_parent_type_is_dropped() -> None:
    body = {"type": "integer", "anyOf": [{"minimum": 1}, {"type": "null"}], "default": 1}
    assert parse_option("option", body).type_source == "int"
    assert not _accepts(body, None)


def test_enum_is_filtered_by_type_and_bounds() -> None:
    assert not _accepts({"type": ["string", "null"], "enum": ["a"], "default": "a"}, None)
    integer_enum = {"type": "integer", "enum": [0, 1, 2], "minimum": 1, "default": 1}
    assert not _accepts(integer_enum, 0)
    assert _accepts(integer_enum, 2)
    assert parse_option("option", integer_enum).type_source == "Literal[1, 2]"


def test_enum_distinguishes_booleans_from_numbers() -> None:
    assert parse_option("option", {"type": "integer", "enum": [True, 1], "default": 1}).type_source == "Literal[1]"


def test_every_pattern_that_applies_must_match() -> None:
    body = {"type": "string", "pattern": "^a", "anyOf": [{"pattern": "z$"}], "default": "az"}
    assert _accepts(body, "az")
    assert not _accepts(body, "a")
    assert not _accepts(body, "z")


def test_one_of_requires_exactly_one_matching_branch() -> None:
    body = {"oneOf": [{"type": "integer"}, {"type": "number"}], "default": 1.5}
    assert not _accepts(body, 1)
    assert _accepts(body, 1.5)
    published = TypeAdapter(parse_option("option", body).annotation).json_schema()
    assert published == {"oneOf": [{"type": "integer"}, {"type": "number"}]}


def test_keywords_without_a_pydantic_form_are_enforced_and_published() -> None:
    body = {"type": "integer", "multipleOf": 5, "default": 10}
    assert _accepts(body, 10)
    assert not _accepts(body, 7)
    assert TypeAdapter(parse_option("option", body).annotation).json_schema() == {"type": "integer", "multipleOf": 5}
    assert not _accepts({"type": "array", "uniqueItems": True, "default": []}, [1, 1])
    assert not _accepts({"not": {"type": "string"}, "default": 1}, "a")


def test_null_default_still_makes_a_typed_option_optional() -> None:
    assert _accepts({"type": "integer", "minimum": 1, "default": None}, None)
    assert not _accepts({"type": "integer", "minimum": 1, "default": None}, 0)


def test_legacy_type_aliases_and_draft4_bounds_are_valid_schemas() -> None:
    issues: list[str] = []
    parse_option("count", {"type": "int", "default": 1}, issues)
    parse_option("label", {"type": "path", "default": "x"}, issues)
    rate = parse_option("rate", {"type": "number", "minimum": 0, "exclusiveMinimum": True, "default": 0.5}, issues)
    assert issues == []
    assert TypeAdapter(rate.annotation).validate_python(0.5) == 0.5
    with pytest.raises(ValidationError):
        TypeAdapter(rate.annotation).validate_python(0)


def test_invalid_json_schema_falls_back_to_the_translated_type_with_a_warning() -> None:
    issues: list[str] = []
    spec = parse_option("count", {"type": "integer", "minimum": "five", "default": 1}, issues)
    assert "not a valid JSON Schema" in issues[0]
    assert TypeAdapter(spec.annotation).validate_python(3) == 3


ITEMS_WITH_BRANCHES_MLPROJECT = (
    """
name: items_with_branches
user_options:
  tags:
    type: array
    items:
      anyOf:
      - type: integer
        minimum: 5
      - type: string
        pattern: "^a"
    anyOf:
    - items:
        maxLength: 3
    default: [5]
"""
    + ENTRY_POINTS
)


def test_items_with_branches_validate_exactly_through_the_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_mlproject(tmp_path, ITEMS_WITH_BRANCHES_MLPROJECT)
    monkeypatch.chdir(tmp_path)
    service, _ = build_mlproject_app(parse_mlproject(tmp_path / "MLproject"))

    def create(tags: list[Any]) -> int:
        payload = {"name": "c", "data": {"user_option_values": {"tags": tags}}}
        return client.post("/api/v1/configs", json=payload).status_code

    with TestClient(service) as client:
        assert create(["bad"]) == 422
        assert create([4]) == 422
        assert create(["abcd"]) == 422
        assert create([5]) == 201
        assert create(["abc", 9]) == 201
        tags_schema = client.get("/api/v1/configs/$schema").json()["properties"]["tags"]
        assert tags_schema["items"]["anyOf"][1] == {"type": "string", "pattern": "^a"}
