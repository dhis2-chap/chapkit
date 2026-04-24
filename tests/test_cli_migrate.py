"""Tests for `chapkit migrate`: classification, detection, schema codegen, and end-to-end."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chapkit.cli.cli import app
from chapkit.cli.migrate import (
    Action,
    MigrateError,
    build_config_fields,
    build_service_info_context,
    classify,
    detect_base_image,
)
from chapkit.cli.mlproject import MLProject, _python_class_name, parse_mlproject


def test_python_class_name_pascal_cases_snake() -> None:
    assert _python_class_name("ewars_template") == "EwarsTemplate"
    assert _python_class_name("minimalist_r") == "MinimalistR"
    assert _python_class_name("simple_multistep_example") == "SimpleMultistepExample"
    # Edge cases: leading digit gets ML_ prefix via _python_identifier, then PascalCase.
    # "123bad" -> "ML_123bad" -> ["ML","123bad"] -> "ML" + "123bad" (digit cannot upper) -> "ML123bad"
    assert _python_class_name("123bad") == "ML123bad"


def test_python_class_name_preserves_internal_caps() -> None:
    # We uppercase the first letter of each segment and leave the rest alone,
    # so "foo_BAR" -> "FooBAR" (not "FooBar") - useful for acronyms.
    assert _python_class_name("foo_BAR") == "FooBAR"


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
meta_data:
  display_name: CHAP-EWARS Model
  description: Modified version of the WHO EWARS model.
  author: CHAP team
  author_assessed_status: orange
  organization: HISP Centre, University of Oslo
  organization_logo_url: https://example.org/logo.png
  contact_email: knut.rand@dhis2.org
  citation_info: Cite me.
docker_env:
  image: ghcr.io/dhis2-chap/docker_r_inla:master
user_options:
  n_lags:
    type: integer
    default: 3
  precision:
    type: number
    default: 0.01
entry_points:
  train:
    command: "Rscript train.R {train_data} {model} {model_config}"
  predict:
    command: "Rscript predict.R {model} {historic_data} {future_data} {out_file} {model_config}"
"""

PYTHON_MLPROJECT = """
name: simple_multistep_example
uv_env: pyproject.toml
entry_points:
  train:
    command: "python train.py {train_data} {model}"
  predict:
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
"""

UNKNOWN_PARAM_MLPROJECT = """
name: custom_param_model
entry_points:
  train:
    command: "python train.py {dataset}"
  predict:
    command: "python predict.py {historic_data} {future_data} {out_file}"
"""


def _seed_project(root: Path, mlproject_yaml: str, files: dict[str, str]) -> MLProject:
    root.mkdir(parents=True, exist_ok=True)
    (root / "MLproject").write_text(mlproject_yaml)
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return parse_mlproject(root)


def test_classify_chaff_metadata_moves_to_old_user_data_stays(tmp_path: Path) -> None:
    # Source / helpers / lockfiles / LICENSE / arbitrary user data files stay.
    (tmp_path / "train.r").write_text("# train\n")
    (tmp_path / "predict.r").write_text("# predict\n")
    (tmp_path / "lib.R").write_text("# helpers\n")
    (tmp_path / "renv.lock").write_text("{}\n")
    (tmp_path / "LICENSE").write_text("MIT\n")
    (tmp_path / "schema.yaml").write_text("kind: config\n")  # arbitrary config script may read
    (tmp_path / "my_package").mkdir()
    (tmp_path / "my_package" / "__init__.py").write_text("\n")
    (tmp_path / ".github").mkdir()

    # Project metadata, MLproject, ad-hoc runners, stale data -> _old/.
    (tmp_path / "MLproject").write_text("name: x\nentry_points: {}\n")
    (tmp_path / "README.md").write_text("# doc\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    (tmp_path / ".dockerignore").write_text(".venv\n")
    (tmp_path / ".python-version").write_text("3.13\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\nversion="0"\n')
    (tmp_path / ".Rprofile").write_text("# profile\n")
    (tmp_path / "Makefile").write_text("all:\n\ttrue\n")
    (tmp_path / "isolated_run.r").write_text("# dev runner\n")
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "example_data_monthly").mkdir()

    # Ignored.
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()

    result = {c.path.name: c.action for c in classify(tmp_path)}

    assert result["train.r"] is Action.KEEP
    assert result["predict.r"] is Action.KEEP
    assert result["lib.R"] is Action.KEEP
    assert result["renv.lock"] is Action.KEEP
    assert result["LICENSE"] is Action.KEEP
    assert result["schema.yaml"] is Action.KEEP  # user data file, not chapkit metadata
    assert result["my_package"] is Action.KEEP
    assert result[".github"] is Action.KEEP

    assert result["MLproject"] is Action.MOVE_TO_OLD
    assert result["README.md"] is Action.MOVE_TO_OLD
    assert result[".gitignore"] is Action.MOVE_TO_OLD
    assert result[".dockerignore"] is Action.MOVE_TO_OLD
    assert result[".python-version"] is Action.MOVE_TO_OLD
    assert result["pyproject.toml"] is Action.MOVE_TO_OLD
    assert result[".Rprofile"] is Action.MOVE_TO_OLD
    assert result["Makefile"] is Action.MOVE_TO_OLD
    assert result["isolated_run.r"] is Action.MOVE_TO_OLD
    assert result["input"] is Action.MOVE_TO_OLD
    assert result["output"] is Action.MOVE_TO_OLD
    assert result["example_data_monthly"] is Action.MOVE_TO_OLD

    assert result[".git"] is Action.IGNORE
    assert result["__pycache__"] is Action.IGNORE


def test_detect_base_image_r_no_inla(tmp_path: Path) -> None:
    mlproject = _seed_project(
        tmp_path,
        MINIMALIST_MLPROJECT,
        {"train.r": "df <- read.csv('x')\n", "predict.r": "df <- read.csv('x')\n"},
    )
    image, language, _ = detect_base_image(tmp_path, mlproject, override=None)
    assert image == "chapkit-r"
    assert language == "r"


def test_detect_base_image_r_with_inla(tmp_path: Path) -> None:
    mlproject = _seed_project(
        tmp_path,
        EWARS_MLPROJECT,
        {"train.R": "library(INLA)\n", "predict.R": "library(INLA)\n"},
    )
    image, language, rationale = detect_base_image(tmp_path, mlproject, override=None)
    assert image == "chapkit-r-inla"
    assert language == "r"
    assert "docker_r_inla" in rationale or "INLA" in rationale


def test_detect_base_image_python(tmp_path: Path) -> None:
    mlproject = _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {"train.py": "import pandas\n", "predict.py": "import pandas\n"},
    )
    image, language, _ = detect_base_image(tmp_path, mlproject, override=None)
    assert image == "chapkit-py"
    assert language == "python"


def test_detect_base_image_mixed_picks_fat(tmp_path: Path) -> None:
    mlproject = _seed_project(
        tmp_path,
        MINIMALIST_MLPROJECT,
        {"train.r": "...", "predict.r": "...", "helper.py": "..."},
    )
    image, language, _ = detect_base_image(tmp_path, mlproject, override=None)
    assert image == "chapkit-r-inla"
    assert language == "mixed"


def test_detect_base_image_override_respected(tmp_path: Path) -> None:
    mlproject = _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {"train.py": "...", "predict.py": "..."},
    )
    image, language, _ = detect_base_image(tmp_path, mlproject, override="chapkit-r")
    assert image == "chapkit-r"
    assert language == "r"


def test_detect_base_image_invalid_override(tmp_path: Path) -> None:
    mlproject = _seed_project(tmp_path, PYTHON_MLPROJECT, {"train.py": "...", "predict.py": "..."})
    with pytest.raises(MigrateError):
        detect_base_image(tmp_path, mlproject, override="something-else")


def test_detect_base_image_requires_scripts(tmp_path: Path) -> None:
    mlproject = _seed_project(tmp_path, PYTHON_MLPROJECT, {})
    with pytest.raises(MigrateError):
        detect_base_image(tmp_path, mlproject, override=None)


def test_build_service_info_context_ewars(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(EWARS_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    ctx = build_service_info_context(mlproject)

    assert ctx["META_AUTHOR"] == "CHAP team"
    assert ctx["META_ASSESSED_STATUS"] == "orange"
    assert ctx["META_CONTACT_EMAIL"] == "knut.rand@dhis2.org"
    assert ctx["META_ORGANIZATION"] == "HISP Centre, University of Oslo"
    assert ctx["META_ORGANIZATION_LOGO_URL"] == "https://example.org/logo.png"
    assert ctx["META_CITATION_INFO"] == "Cite me."
    # supported_period_type: any -> monthly (chapkit's PeriodType only has weekly/monthly).
    assert ctx["PERIOD_TYPE"] == "monthly"
    assert ctx["REQUIRED_COVARIATES"] == ["population"]
    assert ctx["ALLOW_FREE_COVARIATES"] is True
    assert ctx["REQUIRES_GEO"] is False


def test_build_service_info_context_minimalist_defaults(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(MINIMALIST_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    ctx = build_service_info_context(mlproject)

    assert ctx["META_AUTHOR"] is None
    assert ctx["META_ASSESSED_STATUS"] is None
    assert ctx["PERIOD_TYPE"] == "monthly"  # default when supported_period_type absent
    assert ctx["REQUIRED_COVARIATES"] == []
    assert ctx["ALLOW_FREE_COVARIATES"] is False


def test_build_service_info_context_invalid_assessed_status(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(
        """
name: magenta_model
meta_data:
  author_assessed_status: magenta
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    )
    mlproject = parse_mlproject(tmp_path)
    ctx = build_service_info_context(mlproject)
    # Unknown status falls back to None so the template emits the TODO default.
    assert ctx["META_ASSESSED_STATUS"] is None


def test_build_service_info_context_weekly_period(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(
        """
name: weekly_model
supported_period_type: weekly
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    )
    mlproject = parse_mlproject(tmp_path)
    assert build_service_info_context(mlproject)["PERIOD_TYPE"] == "weekly"


def test_build_service_info_context_drops_non_url_logo(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(
        """
name: bad_url_model
meta_data:
  organization_logo_url: /local/path/logo.png
  repository_url: not-a-url
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    )
    mlproject = parse_mlproject(tmp_path)
    ctx = build_service_info_context(mlproject)
    # Non-http(s) values get dropped so MLServiceInfo's HttpUrl validation doesn't blow up.
    assert ctx["META_ORGANIZATION_LOGO_URL"] is None
    assert ctx["META_REPOSITORY_URL"] is None


def test_build_config_fields_ewars_user_options(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(EWARS_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    fields = build_config_fields(mlproject)
    by_name = {name: (ty, default, desc) for name, ty, default, desc in fields}
    # prediction_periods is injected automatically; its description is set by migrate.
    assert by_name["prediction_periods"][:2] == ("int", "3")
    assert by_name["prediction_periods"][2] is not None
    # user_options without descriptions carry None forward (ewars_template's n_lags/precision
    # in the test fixture declare no description).
    assert by_name["n_lags"] == ("int", "3", None)
    assert by_name["precision"] == ("float", "0.01", None)


def test_build_config_fields_description_pulled_from_user_option(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(
        """
name: with_descriptions
user_options:
  n_lags:
    type: integer
    default: 3
    description: Number of temporal lags included in the INLA basis.
entry_points:
  train:
    command: "echo {train_data}"
  predict:
    command: "echo {historic_data} {future_data} {out_file}"
"""
    )
    mlproject = parse_mlproject(tmp_path)
    fields = {name: desc for name, _ty, _default, desc in build_config_fields(mlproject)}
    assert fields["n_lags"] == "Number of temporal lags included in the INLA basis."


def test_build_config_fields_minimalist_only_prediction_periods(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(MINIMALIST_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    fields = build_config_fields(mlproject)
    # Just the auto-injected prediction_periods, with its canned description.
    assert len(fields) == 1
    assert fields[0][:3] == ("prediction_periods", "int", "3")
    assert fields[0][3] is not None


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        MINIMALIST_MLPROJECT,
        {"train.r": "...", "predict.r": "...", "input/data.csv": "a,b\n1,2\n"},
    )
    snapshot_before = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}

    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run: no files changed." in result.output

    snapshot_after = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")}
    assert snapshot_before == snapshot_after


def test_full_run_moves_chaff_and_writes_outputs(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        EWARS_MLPROJECT,
        {
            "train.R": "library(INLA)\n",
            "predict.R": "library(INLA)\n",
            "lib.R": "# helpers\n",
            "input/data.csv": "a,b\n1,2\n",
            "isolated_run.R": "# ad-hoc\n",
        },
    )

    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "_old" / "MLproject").is_file()
    assert (tmp_path / "_old" / "input" / "data.csv").is_file()
    assert (tmp_path / "_old" / "isolated_run.R").is_file()

    assert (tmp_path / "train.R").is_file()
    assert (tmp_path / "predict.R").is_file()
    assert (tmp_path / "lib.R").is_file()

    main_py = tmp_path / "main.py"
    assert main_py.is_file()
    source = main_py.read_text()
    assert "class EwarsTemplateConfig(BaseConfig)" in source
    # Config fields without user-option descriptions stay as plain `name: type = default`;
    # description-carrying ones get wrapped in Field(...).
    assert "n_lags: int = 3" in source
    assert "precision: float = 0.01" in source
    # Commands use ShellModelRunner's template vocabulary, not literal filenames.
    assert "Rscript train.R {data_file} model config.yml" in source
    assert "Rscript predict.R model {historic_file} {future_file} {output_file} config.yml" in source
    # Migrated projects opt into chap-core's nested config.yml shape so ported
    # train/predict scripts can read config["user_option_values"][...] unchanged.
    assert 'config_format="chap_core"' in source
    # Persistent SQLite block is always emitted.
    assert 'os.getenv("DATABASE_URL"' in source
    assert "db_path.parent.mkdir" in source
    # chap-core ecosystem default for additional_continuous_covariates is pinned
    # to rainfall + mean_temperature (with a comment telling users to delete the
    # block if the model doesn't use climate covariates).
    assert "additional_continuous_covariates" in source
    assert 'default=["rainfall", "mean_temperature"]' in source
    assert "DELETE the whole" in source  # the "remove if not intended" note
    ast.parse(source)

    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "FROM ghcr.io/dhis2-chap/chapkit-r-inla:latest" in dockerfile

    assert (tmp_path / "CHAPKIT.md").is_file()
    assert (tmp_path / "pyproject.toml").is_file()
    assert (tmp_path / "compose.yml").is_file()


def test_old_dir_already_exists_errors(tmp_path: Path) -> None:
    _seed_project(tmp_path, PYTHON_MLPROJECT, {"train.py": "...", "predict.py": "..."})
    (tmp_path / "_old").mkdir()

    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "_old" in result.output


def test_existing_readme_moves_to_old(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {"train.py": "...", "predict.py": "...", "README.md": "# user readme\n"},
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "_old" / "README.md").read_text() == "# user readme\n"
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "README.md").read_text() != "# user readme\n"
    assert (tmp_path / "CHAPKIT.md").is_file()


def test_existing_user_pyproject_moves_to_old_and_deps_merge(tmp_path: Path) -> None:
    user_pyproject = """\
[project]
name = "user-model"
version = "0.0.1"
description = "user's model"
dependencies = [
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "numpy",
    "chapkit>=0.1",
]
"""
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {
            "train.py": "...",
            "predict.py": "...",
            "pyproject.toml": user_pyproject,
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output

    # Original user pyproject.toml is preserved under _old/ for reference.
    assert (tmp_path / "_old" / "pyproject.toml").is_file()
    assert "user-model" in (tmp_path / "_old" / "pyproject.toml").read_text()

    # Generated root pyproject.toml is chapkit-shaped AND includes the user's deps,
    # except for any chapkit entry (we add chapkit ourselves with a pinned version).
    generated = (tmp_path / "pyproject.toml").read_text()
    assert "user-model" not in generated
    assert "chapkit>=" in generated
    assert "pandas>=2.0" in generated
    assert "scikit-learn>=1.3" in generated
    assert '"numpy"' in generated
    # Must NOT contain the user's own chapkit entry (our version wins).
    assert '"chapkit>=0.1"' not in generated

    assert "Merged 3 dependency" in result.output
    assert "_old/pyproject.toml" in result.output


def test_pyenv_yaml_fallback_when_mlproject_uses_docker_env_only(tmp_path: Path) -> None:
    """pyenv.yaml at project root is picked up even when MLproject declares only docker_env.

    Repos like chap-models/rwanda_sarimax ship pyenv.yaml alongside a docker_env-only
    MLproject; without this fallback, the image would lack pandas/statsmodels/joblib
    and train.py would crash on import. Falls back only when the MLproject is silent
    about python_env / conda_env so explicit declarations stay authoritative.
    """
    docker_env_only_mlproject = """
name: docker_env_only_model
docker_env:
  image: ghcr.io/dhis2-chap/python_base_image:master
entry_points:
  train:
    command: "python train.py {train_data} {model}"
  predict:
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
"""
    pyenv_yaml = """
python: "3.11.3"
dependencies:
  - pandas
  - statsmodels
  - joblib
"""
    _seed_project(
        tmp_path,
        docker_env_only_mlproject,
        {
            "train.py": "...",
            "predict.py": "...",
            "pyenv.yaml": pyenv_yaml,
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output

    generated = (tmp_path / "pyproject.toml").read_text()
    assert '"pandas"' in generated
    assert '"statsmodels"' in generated
    assert '"joblib"' in generated


def test_pyproject_without_deps_still_works(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {
            "train.py": "...",
            "predict.py": "...",
            "pyproject.toml": '[project]\nname = "nodeps"\nversion = "0.0.1"\n',
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    generated = (tmp_path / "pyproject.toml").read_text()
    assert "chapkit>=" in generated
    # No "Merged N dependencies" line when there were none.
    assert "Merged" not in result.output


def test_invalid_pyproject_does_not_crash(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {
            "train.py": "...",
            "predict.py": "...",
            "pyproject.toml": "this is: not @@@ valid TOML",
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    generated = (tmp_path / "pyproject.toml").read_text()
    assert "chapkit>=" in generated
    # Broken toml was still moved aside, and no deps were merged.
    assert (tmp_path / "_old" / "pyproject.toml").is_file()
    assert "Merged" not in result.output


def test_unknown_param_hard_errors_with_yes(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        UNKNOWN_PARAM_MLPROJECT,
        {"train.py": "...", "predict.py": "..."},
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "dataset" in result.output


def test_unknown_param_override_via_flag(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        UNKNOWN_PARAM_MLPROJECT,
        {"train.py": "...", "predict.py": "..."},
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes", "--param", "dataset=data.csv"])
    assert result.exit_code == 0, result.output
    source = (tmp_path / "main.py").read_text()
    assert "python train.py data.csv" in source


def test_unknown_param_interactive_prompt(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        UNKNOWN_PARAM_MLPROJECT,
        {"train.py": "...", "predict.py": "..."},
    )
    runner = CliRunner()
    # Answer the prompt with "data.csv\n" then confirm the plan with "y\n".
    result = runner.invoke(app, ["migrate", str(tmp_path)], input="data.csv\ny\n")
    assert result.exit_code == 0, result.output
    source = (tmp_path / "main.py").read_text()
    assert "python train.py data.csv" in source
