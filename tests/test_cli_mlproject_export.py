"""Tests for chapkit mlproject export, which generates an MLproject from a service module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from chapkit.cli.cli import app
from chapkit.cli.export import (
    PROVENANCE_FIELDS,
    find_service_parts,
    load_service_module,
    mlproject_command,
    semantic_problems,
)
from chapkit.cli.mlproject import (
    MLProjectError,
    build_ml_service_info,
    parse_mlproject,
    translate_command,
    translate_to_runner_template,
)
from chapkit.cli.run import build_mlproject_app

SERVICE_MODULE = '''
"""Service used by the export tests."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner


class Family(StrEnum):
    """Likelihood families."""

    poisson = "poisson"
    nbinomial = "nbinomial"


class ExportConfig(BaseConfig):
    """Config for the export test model."""

    n_samples: int = Field(default=100, ge=1, description="Posterior draws per row.")
    family: Family = Family.nbinomial
    cell: Literal["gru", "simple"] = Field("gru", title="Recurrent cell type")
    lags: list[int] = Field(default_factory=lambda: [1, 2])
    max_epochs: int | None = None
    additional_continuous_covariates: list[str] = Field(default_factory=lambda: ["rainfall"])


runner = ShellModelRunner(
    train_command="Rscript train.R --data {data_file} --geo {geo_file}",
    predict_command="Rscript predict.R {historic_file} {future_file} {output_file} config.yml",
    adapters={"Cases": "disease_cases"},
)

info = MLServiceInfo(
    id="export-model",
    display_name="Export Model",
    version="1.2.3",
    description="A model used to test chapkit mlproject export.",
    model_metadata=ModelMetadata(
        author="CHAP team",
        author_note="Test only.",
        author_assessed_status=AssessedStatus.orange,
        contact_email="chap@example.org",
        repository_url="https://example.org/export-model",
    ),
    period_type=PeriodType.monthly,
    required_covariates=["population"],
    allow_free_additional_continuous_covariates=True,
    requires_geo=True,
    min_prediction_periods=1,
    max_prediction_periods=12,
)

HIERARCHY = ArtifactHierarchy(name="export_model", level_labels={0: "ml_training_workspace", 1: "ml_prediction"})

print("building the service")
app = MLServiceBuilder(info=info, config_schema=ExportConfig, hierarchy=HIERARCHY, runner=runner).build()
'''


def _write_service(tmp_path: Path, source: str = SERVICE_MODULE) -> Path:
    module = tmp_path / "main.py"
    module.write_text(source)
    return module


def test_export_writes_an_mlproject_that_runs_as_the_same_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _write_service(tmp_path)
    result = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output

    mlproject_file = tmp_path / "MLproject"
    text = mlproject_file.read_text()
    assert text.startswith("# Generated from main.py by `chapkit mlproject export`")

    document = yaml.safe_load(text)
    assert document["name"] == "export_model"
    assert document["supported_period_type"] == "month"
    assert document["adapters"] == {"Cases": "disease_cases"}
    # The runner writes a flat config.yml; chap-core ignores meta_data keys it does not know.
    assert document["meta_data"]["config_format"] == "flat"
    assert document["entry_points"]["train"] == {
        "parameters": {"train_data": "path", "polygons": "path"},
        "command": "Rscript train.R --data {train_data} --geo {polygons}",
    }
    assert document["entry_points"]["predict"]["command"] == (
        "Rscript predict.R {historic_data} {future_data} {out_file} config.yml"
    )

    options = document["user_options"]
    assert list(options) == ["additional_continuous_covariates", "n_samples", "family", "cell", "lags", "max_epochs"]
    assert options["n_samples"] == {
        "type": "integer",
        "minimum": 1,
        "default": 100,
        "description": "Posterior draws per row.",
    }
    # Enum classes are inlined: their docstring stays as the description, their class name is dropped.
    assert options["family"] == {
        "type": "string",
        "enum": ["poisson", "nbinomial"],
        "default": "nbinomial",
        "description": "Likelihood families.",
    }
    assert options["cell"]["title"] == "Recurrent cell type"
    assert options["lags"]["default"] == [1, 2]
    assert options["max_epochs"]["default"] is None
    # prediction_periods keeps BaseConfig's default, so chapkit adds it itself.
    assert "prediction_periods" not in options

    # chapkit mlproject run reads the file back as the same service.
    parts = find_service_parts(load_service_module(module))
    served = build_ml_service_info(parse_mlproject(mlproject_file))
    exclude = set(PROVENANCE_FIELDS)
    assert served.model_dump(mode="json", exclude=exclude) == parts.info.model_dump(mode="json", exclude=exclude)

    monkeypatch.chdir(tmp_path)
    service, _ = build_mlproject_app(parse_mlproject(mlproject_file))
    with TestClient(service) as client:
        created = client.post("/api/v1/configs", json={"name": "defaults", "data": {}})
        assert created.status_code == 201, created.text
        expected = parts.config_schema().model_dump(mode="json")
        assert {key: created.json()["data"][key] for key in expected} == expected


def test_check_passes_for_a_fresh_export_and_fails_when_stale(tmp_path: Path) -> None:
    module = _write_service(tmp_path)
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0

    fresh = CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"])
    assert fresh.exit_code == 0, fresh.output
    assert "MLproject is up to date with main.py" in fresh.stdout
    # Output the module prints while importing stays out of stdout.
    assert "building the service" not in fresh.stdout

    _write_service(tmp_path, SERVICE_MODULE.replace("default=100", "default=200"))
    stale = CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"])
    assert stale.exit_code == 1
    assert "-    default: 100" in stale.stdout
    assert "+    default: 200" in stale.stdout
    assert "out of date" in stale.stdout


def test_check_reports_semantic_drift_that_the_text_does_not_show(tmp_path: Path) -> None:
    # chapkit mlproject run strips the surrounding blanks the service keeps.
    module = _write_service(tmp_path, SERVICE_MODULE.replace('author_note="Test only."', 'author_note="Test only.  "'))
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0

    result = CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"])
    assert result.exit_code == 1
    assert "info.model_metadata" in result.stdout


def test_check_accepts_a_service_without_author_note(tmp_path: Path) -> None:
    # chapkit mlproject run serves the description in its place, as chap-core does.
    module = _write_service(tmp_path, SERVICE_MODULE.replace('        author_note="Test only.",\n', ""))
    exported = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert exported.exit_code == 0
    assert "author_note" not in exported.output

    result = CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"])
    assert result.exit_code == 0, result.output


def test_check_fails_without_an_mlproject(tmp_path: Path) -> None:
    module = _write_service(tmp_path)
    result = CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"])
    assert result.exit_code == 1
    assert "out of date" in result.stdout


def test_export_refuses_to_overwrite_a_hand_written_mlproject(tmp_path: Path) -> None:
    module = _write_service(tmp_path)
    (tmp_path / "MLproject").write_text("name: hand_written\n")

    refused = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert refused.exit_code == 1
    assert "--force" in refused.output
    assert (tmp_path / "MLproject").read_text() == "name: hand_written\n"

    forced = CliRunner().invoke(app, ["mlproject", "export", str(module), "--force"])
    assert forced.exit_code == 0, forced.output
    assert "chapkit mlproject export" in (tmp_path / "MLproject").read_text()


def test_export_writes_to_output_and_updates_an_existing_mlproject_yaml(tmp_path: Path) -> None:
    module = _write_service(tmp_path)
    other = tmp_path / "out" / "MLproject"
    other.parent.mkdir()
    assert CliRunner().invoke(app, ["mlproject", "export", str(module), "-o", str(other)]).exit_code == 0
    assert other.is_file()
    assert not (tmp_path / "MLproject").exists()

    assert (
        CliRunner().invoke(app, ["mlproject", "export", str(module), "-o", str(tmp_path / "MLproject.yaml")]).exit_code
        == 0
    )
    assert CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"]).exit_code == 0
    assert not (tmp_path / "MLproject").exists()


def test_export_names_the_mlproject_after_the_service_id_when_the_hierarchy_does_not_match(tmp_path: Path) -> None:
    module = _write_service(tmp_path, SERVICE_MODULE.replace('name="export_model"', 'name="ml"'))
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    assert yaml.safe_load((tmp_path / "MLproject").read_text())["name"] == "export_model"


def test_export_keeps_a_changed_prediction_periods_default(tmp_path: Path) -> None:
    source = SERVICE_MODULE.replace(
        "    max_epochs: int | None = None\n", "    max_epochs: int | None = None\n    prediction_periods: int = 6\n"
    )
    module = _write_service(tmp_path, source)
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    options = yaml.safe_load((tmp_path / "MLproject").read_text())["user_options"]
    assert options["prediction_periods"]["default"] == 6
    assert CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"]).exit_code == 0


def test_export_needs_a_config_name_when_the_module_defines_several(tmp_path: Path) -> None:
    source = SERVICE_MODULE.replace(
        "runner = ShellModelRunner(",
        'class OtherConfig(BaseConfig):\n    """Another config."""\n\n\nrunner = ShellModelRunner(',
    )
    module = _write_service(tmp_path, source)

    ambiguous = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert ambiguous.exit_code == 1
    assert "ExportConfig, OtherConfig" in ambiguous.output
    assert "--config" in ambiguous.output

    chosen = CliRunner().invoke(app, ["mlproject", "export", str(module), "--config", "ExportConfig"])
    assert chosen.exit_code == 0, chosen.output
    assert "n_samples" in (tmp_path / "MLproject").read_text()


def test_export_rejects_a_functional_runner(tmp_path: Path) -> None:
    source = SERVICE_MODULE.replace(
        "from chapkit.ml import ShellModelRunner", "from chapkit.ml import FunctionalModelRunner"
    )
    head, rest = source.split("runner = ShellModelRunner(")
    tail = rest.split('    adapters={"Cases": "disease_cases"},\n)\n', 1)[1]
    functional_runner = (
        "async def on_train(*args, **kwargs):\n"
        '    """Train."""\n\n\n'
        "async def on_predict(*args, **kwargs):\n"
        '    """Predict."""\n\n\n'
        "runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)\n"
    )
    source = head + functional_runner + tail
    module = _write_service(tmp_path, source)
    with pytest.raises(MLProjectError, match="Only ShellModelRunner services"):
        find_service_parts(load_service_module(module))


def test_export_reports_a_module_that_fails_to_import(tmp_path: Path) -> None:
    module = _write_service(tmp_path, "raise RuntimeError('boom')\n")
    result = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert result.exit_code == 1
    assert "Importing" in result.output
    assert "boom" in result.output


def test_export_reports_a_missing_module(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["mlproject", "export", str(tmp_path / "main.py")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_load_service_module_leaves_no_trace(tmp_path: Path) -> None:
    import os
    import sys

    module = _write_service(tmp_path)
    before = (os.getcwd(), list(sys.path))
    load_service_module(module)
    assert (os.getcwd(), list(sys.path)) == before
    assert not (tmp_path / "__pycache__").exists()


def test_export_keeps_constraints_on_a_reserved_field_with_its_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SERVICE_MODULE.replace(
        "    max_epochs: int | None = None\n",
        "    max_epochs: int | None = None\n    prediction_periods: int = Field(default=3, ge=2, le=6)\n",
    )
    module = _write_service(tmp_path, source)
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    option = yaml.safe_load((tmp_path / "MLproject").read_text())["user_options"]["prediction_periods"]
    assert option == {"type": "integer", "maximum": 6, "minimum": 2, "default": 3}

    monkeypatch.chdir(tmp_path)
    service, _ = build_mlproject_app(parse_mlproject(tmp_path / "MLproject"))
    with TestClient(service) as client:
        rejected = client.post("/api/v1/configs", json={"name": "c", "data": {"prediction_periods": 1000}})
        assert rejected.status_code == 422


def test_export_leaves_out_a_reserved_field_that_only_gains_a_description(tmp_path: Path) -> None:
    source = SERVICE_MODULE.replace(
        "    max_epochs: int | None = None\n",
        '    max_epochs: int | None = None\n    prediction_periods: int = Field(default=3, description="Horizon.")\n',
    )
    module = _write_service(tmp_path, source)
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    assert "prediction_periods" not in yaml.safe_load((tmp_path / "MLproject").read_text())["user_options"]


def test_check_reports_an_option_that_accepts_different_values(tmp_path: Path) -> None:
    module = _write_service(tmp_path)
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    mlproject_file = tmp_path / "MLproject"
    mlproject_file.write_text(mlproject_file.read_text().replace("    minimum: 1\n", ""))

    problems = semantic_problems(mlproject_file, find_service_parts(load_service_module(module)))
    assert any(problem.startswith("config.n_samples: the MLproject accepts") for problem in problems)


@pytest.mark.parametrize(
    "runner_command",
    [
        'python -c "print({{}})" {data_file}',
        "awk '{{print $1}}' {historic_file} > {output_file}",
        "Rscript predict.R {historic_file} {future_file} {output_file}",
    ],
)
def test_exported_commands_run_as_the_runner_would(runner_command: str) -> None:
    command = mlproject_command(runner_command)
    assert translate_to_runner_template(command) == runner_command
    # mlproject run formats twice: MLproject parameters, then the runner's placeholders.
    runner_values = {"data_file": "data.csv", "historic_file": "historic.csv", "output_file": "predictions.csv"}
    runner_values |= {"future_file": "future.csv", "geo_file": ""}
    assert translate_command(command).format(**runner_values) == runner_command.format(**runner_values)


def test_export_rejects_a_runner_placeholder_without_an_mlproject_parameter() -> None:
    with pytest.raises(MLProjectError, match=r"\{dataset\} has no MLproject parameter"):
        mlproject_command("python train.py {dataset}")


def test_export_writes_no_config_format_for_a_chap_core_runner(tmp_path: Path) -> None:
    module = _write_service(
        tmp_path, SERVICE_MODULE.replace("    adapters=", '    config_format="chap_core",\n    adapters=')
    )
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    assert "config_format" not in yaml.safe_load((tmp_path / "MLproject").read_text())["meta_data"]


def test_check_reports_a_config_layout_the_service_does_not_write(tmp_path: Path) -> None:
    module = _write_service(tmp_path)
    assert CliRunner().invoke(app, ["mlproject", "export", str(module)]).exit_code == 0
    mlproject_file = tmp_path / "MLproject"
    mlproject_file.write_text(mlproject_file.read_text().replace("  config_format: flat\n", ""))

    problems = semantic_problems(mlproject_file, find_service_parts(load_service_module(module)))
    assert "config.yml layout: the MLproject gives 'chap_core', the service writes 'flat'" in problems


def test_export_warns_about_settings_the_mlproject_cannot_carry(tmp_path: Path) -> None:
    source = SERVICE_MODULE.replace(
        "runner = ShellModelRunner(\n",
        "async def reject_train(*args, **kwargs):\n"
        '    """Validate training data."""\n\n\n'
        "runner = ShellModelRunner(\n    on_validate_train=reject_train,\n",
    )
    source = source.replace(
        'level_labels={0: "ml_training_workspace", 1: "ml_prediction"}', 'level_labels={0: "trained", 1: "predicted"}'
    )
    source = source.replace(
        "runner=runner).build()",
        'runner=runner).with_auth(api_keys=["secret"]).build()\n\n\n@app.get("/custom")\ndef custom() -> dict:\n'
        '    """Serve an extra route."""\n    return {}\n',
    )
    module = _write_service(tmp_path, source)

    result = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert result.exit_code == 0, result.output
    assert "on_validate_train hook" in result.stderr
    assert "on_validate_predict" not in result.stderr
    assert "artifact level labels" in result.stderr
    assert "API key authentication" in result.stderr
    assert "routes mlproject run does not serve: /custom" in result.stderr

    # --check still passes: the MLproject is what export writes; the warnings repeat on stderr.
    checked = CliRunner().invoke(app, ["mlproject", "export", str(module), "--check"])
    assert checked.exit_code == 0, checked.output
    assert "on_validate_train hook" in checked.stderr
    assert "WARNING" not in checked.stdout


def test_export_warns_about_a_custom_keepalive_interval(tmp_path: Path) -> None:
    module = _write_service(
        tmp_path,
        SERVICE_MODULE.replace(
            "runner=runner).build()", "runner=runner).with_registration(keepalive_interval=15).build()"
        ),
    )
    result = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert result.exit_code == 0, result.output
    assert "registration keepalive interval 15s" in result.stderr
    assert "--keepalive-interval 15" in result.stderr


def test_export_does_not_warn_about_the_default_keepalive_interval(tmp_path: Path) -> None:
    module = _write_service(
        tmp_path, SERVICE_MODULE.replace("runner=runner).build()", "runner=runner).with_registration().build()")
    )
    result = CliRunner().invoke(app, ["mlproject", "export", str(module)])
    assert result.exit_code == 0, result.output
    assert "keepalive" not in result.stderr
