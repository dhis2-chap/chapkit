"""Fast tests for `chapkit init` template rendering (no uv sync, no Docker)."""

from __future__ import annotations

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
