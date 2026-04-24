"""Tests for MLproject parsing, command translation, and dynamic config schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from chapkit.cli.cli import app
from chapkit.cli.mlproject import (
    MLProjectError,
    _python_identifier,
    build_config_schema,
    find_mlproject,
    parse_mlproject,
    slugify,
    translate_command,
)

MINIMALIST_MLPROJECT = """
name: minimalist_r

renv_env: renv.lock


entry_points:
  train:
    parameters:
      train_data: path
      model: str
    command: "Rscript train.r {train_data} {model}"
  predict:
    parameters:
      historic_data: path
      future_data: path
      model: str
      out_file: path
    command: "Rscript predict.r {model} {historic_data} {future_data} {out_file}"
"""

EWARS_MLPROJECT = """
name: ewars_template

target: disease_cases
required_covariates:
  - population
supported_period_type: any
allow_free_additional_continuous_covariates: true
user_options:
  n_lags:
    type: integer
    default: 3
    description: Number of lags to include in the model.
  precision:
    type: number
    default: 0.01
    description: Prior on the precision of fixed effects. Works as regularization

meta_data:
  display_name: CHAP-EWARS Model
  description: Modified version of the WHO EWARS model.
  author: CHAP team
  author_assessed_status: orange

adapters:
  Cases: disease_cases
  E: population

docker_env:
  image: ghcr.io/dhis2-chap/docker_r_inla:master

entry_points:
  train:
    parameters:
      train_data: path
      model: str
      model_config: path
    command: "Rscript train.R {train_data} {model} {model_config}"
  predict:
    parameters:
      historic_data: path
      future_data: path
      model: str
      out_file: path
      model_config: path
    command: "Rscript predict.R {model} {historic_data} {future_data} {out_file} {model_config}"
"""


def _write_mlproject(tmp_path: Path, contents: str, filename: str = "MLproject") -> Path:
    target = tmp_path / filename
    target.write_text(contents)
    return target


def test_find_mlproject_locates_standard_name(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, MINIMALIST_MLPROJECT)
    assert find_mlproject(tmp_path).name == "MLproject"


def test_find_mlproject_accepts_yaml_extension(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, MINIMALIST_MLPROJECT, filename="MLproject.yaml")
    assert find_mlproject(tmp_path).name == "MLproject.yaml"


def test_find_mlproject_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MLProjectError) as excinfo:
        find_mlproject(tmp_path)
    assert str(tmp_path.resolve()) in str(excinfo.value)


def test_parse_minimalist(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, MINIMALIST_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    assert mlproject.name == "minimalist_r"
    assert set(mlproject.entry_points) == {"train", "predict"}
    assert mlproject.entry_points["train"].parameters == {"train_data": "path", "model": "str"}
    assert mlproject.user_options == {}
    assert mlproject.env_hints == {"renv_env": "renv.lock"}


def test_parse_ewars(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, EWARS_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    assert mlproject.name == "ewars_template"
    assert "n_lags" in mlproject.user_options
    assert mlproject.user_options["n_lags"]["type"] == "integer"
    assert mlproject.user_options["precision"]["default"] == 0.01
    assert mlproject.env_hints == {"docker_env": "ghcr.io/dhis2-chap/docker_r_inla:master"}


def test_parse_rejects_missing_train_entry_point(tmp_path: Path) -> None:
    contents = """
name: broken
entry_points:
  predict:
    command: "echo predict"
"""
    _write_mlproject(tmp_path, contents)
    with pytest.raises(MLProjectError) as excinfo:
        parse_mlproject(tmp_path)
    assert "train" in str(excinfo.value)
    assert "predict" in str(excinfo.value)


def test_parse_rejects_missing_name(tmp_path: Path) -> None:
    contents = """
entry_points:
  train:
    command: "echo train"
  predict:
    command: "echo predict"
"""
    _write_mlproject(tmp_path, contents)
    with pytest.raises(MLProjectError):
        parse_mlproject(tmp_path)


def test_translate_minimalist_commands() -> None:
    train = translate_command("Rscript train.r {train_data} {model}")
    assert train == "Rscript train.r data.csv model"

    predict = translate_command("Rscript predict.r {model} {historic_data} {future_data} {out_file}")
    assert predict == "Rscript predict.r model historic.csv future.csv predictions.csv"


def test_translate_ewars_commands() -> None:
    train = translate_command("Rscript train.R {train_data} {model} {model_config}")
    assert train == "Rscript train.R data.csv model config.yml"

    predict = translate_command("Rscript predict.R {model} {historic_data} {future_data} {out_file} {model_config}")
    assert predict == "Rscript predict.R model historic.csv future.csv predictions.csv config.yml"


def test_translate_unknown_param_raises_with_known_names() -> None:
    with pytest.raises(MLProjectError) as excinfo:
        translate_command("Rscript x.r {dataset}")
    message = str(excinfo.value)
    assert "dataset" in message
    assert "--param" in message
    assert "train_data" in message


def test_translate_override_wins_over_default() -> None:
    result = translate_command("python train.py {train_data}", overrides={"train_data": "custom.csv"})
    assert result == "python train.py custom.csv"


def test_translate_override_enables_unknown_name() -> None:
    result = translate_command("python train.py {dataset}", overrides={"dataset": "input.csv"})
    assert result == "python train.py input.csv"


def test_translate_ignores_empty_braces() -> None:
    assert translate_command("echo hello") == "echo hello"


def test_build_config_schema_ewars(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, EWARS_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    schema: Any = build_config_schema(mlproject)

    instance = schema()
    assert instance.prediction_periods == 3
    assert instance.n_lags == 3
    assert instance.precision == pytest.approx(0.01)

    overridden = schema(n_lags=7, precision=0.5, prediction_periods=12)
    assert overridden.n_lags == 7
    assert overridden.precision == pytest.approx(0.5)
    assert overridden.prediction_periods == 12


def test_build_config_schema_minimalist_has_only_prediction_periods(tmp_path: Path) -> None:
    _write_mlproject(tmp_path, MINIMALIST_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    schema: Any = build_config_schema(mlproject)
    instance = schema()
    assert instance.prediction_periods == 3
    assert set(schema.model_fields) >= {"prediction_periods"}


def test_build_config_schema_aliases_non_identifier_option_names(tmp_path: Path) -> None:
    """Runtime Config schema (used by chapkit run) normalizes hyphenated / keyword names.

    Python attribute gets the sanitized form; wire-format POST keys + config.yml keys
    keep the MLproject original via Pydantic Field(alias=...).
    """
    contents = """
name: hyphens_live_here
user_options:
  n-lags:
    type: integer
    default: 3
  class:
    type: string
    default: forecast
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    _write_mlproject(tmp_path, contents)
    mlproject = parse_mlproject(tmp_path)
    schema: Any = build_config_schema(mlproject)

    # Instantiate via wire-format kwargs (what a POST body would carry).
    instance = schema.model_validate({"n-lags": 7, "class": "ensemble"})
    # Python attribute uses the sanitized name.
    assert instance.n_lags == 7
    assert instance.class_ == "ensemble"

    # model_dump(by_alias=True) round-trips the wire names (what ShellModelRunner's
    # dump_config_yaml uses when writing config.yml for scripts to read).
    dumped = instance.model_dump(by_alias=True)
    assert dumped["n-lags"] == 7
    assert dumped["class"] == "ensemble"


def test_build_config_schema_rejects_leading_digit_option_names(tmp_path: Path) -> None:
    """Leading-digit option names cannot be mapped to a valid Pydantic field - reject with a clear error."""
    contents = """
name: leading_digit_model
user_options:
  1st_period:
    type: integer
    default: 1
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    _write_mlproject(tmp_path, contents)
    mlproject = parse_mlproject(tmp_path)
    with pytest.raises(MLProjectError, match="starts with a digit"):
        build_config_schema(mlproject)


def test_build_config_schema_required_option_without_default(tmp_path: Path) -> None:
    contents = """
name: needs_input
user_options:
  required_knob:
    type: integer
    description: must be provided
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    _write_mlproject(tmp_path, contents)
    mlproject = parse_mlproject(tmp_path)
    schema: Any = build_config_schema(mlproject)
    with pytest.raises(Exception):  # pydantic ValidationError for missing required field
        schema()
    instance = schema(required_knob=42)
    assert instance.required_knob == 42
    assert instance.prediction_periods == 3


def test_build_config_schema_string_default_is_coerced(tmp_path: Path) -> None:
    contents = """
name: coerce_me
user_options:
  some_option:
    type: integer
    default: '10'
    description: default given as string
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    _write_mlproject(tmp_path, contents)
    mlproject = parse_mlproject(tmp_path)
    schema: Any = build_config_schema(mlproject)
    instance = schema()
    assert instance.some_option == 10


def test_slugify_produces_valid_service_id() -> None:
    assert slugify("ewars_template") == "ewars-template"
    assert slugify("CHAP EWARS Model") == "chap-ewars-model"
    assert slugify("123bad") == "mlproject-123bad"
    assert slugify("") == "mlproject"


def test_python_identifier_is_valid() -> None:
    assert _python_identifier("ewars_template") == "ewars_template"
    assert _python_identifier("123bad") == "ML_123bad"
    assert _python_identifier("") == "MLProject"


def test_run_command_errors_cleanly_when_no_mlproject(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
    assert "No MLproject file found" in result.stdout or "No MLproject file found" in result.output


def test_run_command_errors_on_unknown_param(tmp_path: Path) -> None:
    contents = """
name: broken_param
entry_points:
  train:
    command: "Rscript train.r {dataset}"
  predict:
    command: "Rscript predict.r {historic_data} {future_data} {out_file}"
"""
    _write_mlproject(tmp_path, contents)
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
    combined = result.output + result.stdout
    assert "dataset" in combined
    assert "--param" in combined
