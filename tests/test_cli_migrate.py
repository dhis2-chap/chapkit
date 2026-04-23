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
    classify,
    detect_base_image,
)
from chapkit.cli.mlproject import MLProject, parse_mlproject

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


def test_classify_separates_chaff_from_keep_from_ignore(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text("name: x\nentry_points: {}\n")
    (tmp_path / "train.r").write_text("# train\n")
    (tmp_path / "predict.r").write_text("# predict\n")
    (tmp_path / "lib.R").write_text("# helpers\n")
    (tmp_path / "renv.lock").write_text("{}\n")
    (tmp_path / ".Rprofile").write_text("# .Rprofile\n")
    (tmp_path / "isolated_run.r").write_text("# dev runner\n")
    (tmp_path / "Makefile").write_text("all:\n\ttrue\n")
    (tmp_path / "README.md").write_text("# doc\n")
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "example_data_monthly").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()

    result = {c.path.name: c.action for c in classify(tmp_path)}

    assert result["MLproject"] is Action.MOVE_TO_OLD
    assert result["Makefile"] is Action.MOVE_TO_OLD
    assert result[".Rprofile"] is Action.MOVE_TO_OLD
    assert result["isolated_run.r"] is Action.MOVE_TO_OLD
    assert result["input"] is Action.MOVE_TO_OLD
    assert result["output"] is Action.MOVE_TO_OLD
    assert result["example_data_monthly"] is Action.MOVE_TO_OLD

    assert result["train.r"] is Action.KEEP
    assert result["predict.r"] is Action.KEEP
    assert result["lib.R"] is Action.KEEP
    assert result["renv.lock"] is Action.KEEP
    assert result["README.md"] is Action.KEEP

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


def test_build_config_fields_ewars_user_options(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(EWARS_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    fields = build_config_fields(mlproject)
    by_name = {name: (ty, default) for name, ty, default in fields}
    assert by_name["prediction_periods"] == ("int", "3")
    assert by_name["n_lags"] == ("int", "3")
    assert by_name["precision"] == ("float", "0.01")


def test_build_config_fields_minimalist_only_prediction_periods(tmp_path: Path) -> None:
    (tmp_path / "MLproject").write_text(MINIMALIST_MLPROJECT)
    mlproject = parse_mlproject(tmp_path)
    fields = build_config_fields(mlproject)
    assert fields == [("prediction_periods", "int", "3")]


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
    assert "class ewars_templateConfig(BaseConfig)" in source
    assert "n_lags: int = 3" in source
    assert "precision: float = 0.01" in source
    assert "Rscript train.R data.csv model config.yml" in source
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


def test_existing_readme_preserved(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {"train.py": "...", "predict.py": "...", "README.md": "# user readme\n"},
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "README.md").read_text() == "# user readme\n"
    assert (tmp_path / "CHAPKIT.md").is_file()


def test_generated_file_collision_errors(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {"train.py": "...", "predict.py": "...", "main.py": "# existing\n"},
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "main.py" in result.output


def test_existing_user_pyproject_renamed_to_original(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {
            "train.py": "...",
            "predict.py": "...",
            "pyproject.toml": '[project]\nname = "user-model"\nversion = "0.0.1"\n',
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "pyproject.original.toml").is_file()
    assert "user-model" in (tmp_path / "pyproject.original.toml").read_text()

    assert (tmp_path / "pyproject.toml").is_file()
    generated = (tmp_path / "pyproject.toml").read_text()
    assert "chapkit" in generated
    assert "user-model" not in generated


def test_pyproject_original_collision_errors(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        PYTHON_MLPROJECT,
        {
            "train.py": "...",
            "predict.py": "...",
            "pyproject.toml": "[project]\nname='x'\nversion='0'\n",
            "pyproject.original.toml": "# blocker\n",
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["migrate", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "pyproject.original.toml" in result.output


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
