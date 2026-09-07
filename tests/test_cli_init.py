"""Fast tests for `chapkit init` template rendering (no uv sync, no Docker)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chapkit.cli.cli import app


@pytest.mark.parametrize(
    ("template", "expected_from", "expects_amd64"),
    [
        ("fn-py", "FROM ghcr.io/dhis2-chap/chapkit-py:latest", False),
        ("shell-py", "FROM ghcr.io/dhis2-chap/chapkit-py:latest", False),
        ("shell-r", "FROM ghcr.io/dhis2-chap/chapkit-r:latest", False),
        ("shell-r-tidyverse", "FROM ghcr.io/dhis2-chap/chapkit-r-tidyverse:latest", False),
        (
            "shell-r-inla",
            "FROM --platform=${BASE_PLATFORM} ghcr.io/dhis2-chap/chapkit-r-inla:latest",
            True,
        ),
    ],
)
def test_init_dockerfile_and_compose_per_template(
    tmp_path: Path,
    template: str,
    expected_from: str,
    expects_amd64: bool,
) -> None:
    """Each template scaffolds a Dockerfile + compose.yml that targets the right image."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", f"demo-{template}", "--template", template, "--path", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / f"demo-{template}"
    dockerfile = (project_dir / "Dockerfile").read_text()
    compose_yml = (project_dir / "compose.yml").read_text()

    assert expected_from in dockerfile

    # GIT_REVISION is baked into every image and forwarded by compose as a build arg.
    assert 'ARG GIT_REVISION=""' in dockerfile
    assert "ENV GIT_REVISION=${GIT_REVISION}" in dockerfile
    assert "GIT_REVISION: ${GIT_REVISION:-}" in compose_yml

    workflow = (project_dir / ".github" / "workflows" / "publish-docker.yml").read_text()
    assert "GIT_REVISION=${{ github.sha }}" in workflow
    assert "type=sha,format=short" in workflow
    assert ("platforms: linux/amd64" in workflow) == expects_amd64

    dockerignore = (project_dir / ".dockerignore").read_text()
    assert ".git/" in dockerignore
    assert ".venv/" in dockerignore

    # Only chapkit-r-inla pins amd64 - the other R images and chapkit-py are multi-arch.
    if expects_amd64:
        assert "ARG BASE_PLATFORM=linux/amd64" in dockerfile
        # compose.yml emits the active `platform: linux/amd64` only for shell-r-inla;
        # all templates carry a commented-out hint, so match the active line specifically.
        active_platform_lines = [
            line
            for line in compose_yml.splitlines()
            if "platform: linux/amd64" in line and not line.strip().startswith("#")
        ]
        assert active_platform_lines, "shell-r-inla should emit an active platform: linux/amd64 line"
    else:
        assert "ARG BASE_PLATFORM" not in dockerfile
        active_platform_lines = [
            line
            for line in compose_yml.splitlines()
            if "platform: linux/amd64" in line and not line.strip().startswith("#")
        ]
        assert not active_platform_lines, f"{template} should not emit an active platform pin"


@pytest.mark.parametrize("template", ["shell-r", "shell-r-tidyverse", "shell-r-inla"])
def test_init_r_template_emits_train_and_predict_r(tmp_path: Path, template: str) -> None:
    """All three R templates scaffold the shared train.R + predict.R script stubs."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", f"demo-{template}", "--template", template, "--path", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / f"demo-{template}"
    assert (project_dir / "scripts" / "train.R").is_file()
    assert (project_dir / "scripts" / "predict.R").is_file()


def test_init_rejects_unknown_template(tmp_path: Path) -> None:
    """Unknown template names are rejected with a helpful error."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", "demo-bogus", "--template", "shell-haskell", "--path", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "shell-r-tidyverse" in result.output
    assert "shell-r-inla" in result.output


def test_init_pyproject_pins_chapkit_with_floor_and_ceiling(tmp_path: Path) -> None:
    """`chapkit init` emits a bounded chapkit requirement without a dev suffix."""
    runner = CliRunner()
    result = runner.invoke(app, ["init", "demo-pin", "--template", "fn-py", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output

    pyproject = (tmp_path / "demo-pin" / "pyproject.toml").read_text()
    match = re.search(r'"chapkit>=(\d+)\.(\d+)\.(\d+),<(\d+)"', pyproject)
    assert match is not None, pyproject
    assert int(match.group(4)) == int(match.group(1)) + 1
    assert "dev" not in match.group(0)
    assert "rc" not in match.group(0)


@pytest.mark.parametrize("template", ["fn-py", "shell-py", "shell-r"])
def test_init_readme_curl_examples_match_request_schemas(tmp_path: Path, template: str) -> None:
    """The scaffolded README documents $train/$predict payloads that the ML schemas accept."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", f"demo-{template}", "--template", template, "--path", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    readme = (tmp_path / f"demo-{template}" / "README.md").read_text()

    # PredictRequest takes artifact_id, historic and future - not the old model_id shape.
    assert "model_id" not in readme
    assert '"artifact_id": "YOUR_TRAIN_ARTIFACT_ID"' in readme
    assert '"historic": {' in readme
    assert '"future": {' in readme

    # DataFrame payloads are {columns: [...], data: [[...]]}, not column-keyed objects.
    assert '"columns": ["time_period", "location", "rainfall", "mean_temperature", "disease_cases"]' in readme
    assert '"config_id": "YOUR_CONFIG_ID"' in readme

    # Both operations are asynchronous, so the README has to show polling the job.
    assert "/api/v1/jobs/YOUR_TRAIN_JOB_ID" in readme
    assert "/api/v1/jobs/YOUR_PREDICT_JOB_ID" in readme
    assert "/api/v1/artifacts/YOUR_PREDICT_ARTIFACT_ID" in readme


def test_init_r_readme_documents_local_yaml_package(tmp_path: Path) -> None:
    """The R scaffold tells the user to install the R yaml package for local runs."""
    runner = CliRunner()
    result = runner.invoke(app, ["init", "demo-r", "--template", "shell-r", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output

    readme = (tmp_path / "demo-r" / "README.md").read_text()
    assert "Rscript -e 'install.packages(\"yaml\")'" in readme


def test_init_dockerfile_documents_where_uvicorn_comes_from(tmp_path: Path) -> None:
    """The Dockerfile comment credits fastapi[standard] rather than the base image for uvicorn."""
    runner = CliRunner()
    result = runner.invoke(app, ["init", "demo-py", "--template", "fn-py", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output

    dockerfile = (tmp_path / "demo-py" / "Dockerfile").read_text()
    assert "fastapi[standard]" in dockerfile
    assert "uvicorn etc. ship with" not in dockerfile
