"""Tests for MLproject adapters applied by ShellModelRunner, matching chap-core's column adapters."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from chapkit import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml import ShellModelRunner
from chapkit.ml.runner import apply_adapters

EWARS_ADAPTERS = {"Cases": "disease_cases", "E": "population", "ID_year": "year", "ID_spat": "location"}


def _column(frame: DataFrame, name: str) -> list[object]:
    return frame.get_column(name)


def test_apply_adapters_copies_columns_and_derives_monthly_period_parts() -> None:
    frame = DataFrame(
        columns=["time_period", "location", "disease_cases", "population"],
        data=[["2020-04", "a", 5, 100], ["2021-12", "b", 7, 200]],
    )
    adapted = apply_adapters(frame, {**EWARS_ADAPTERS, "month": "month", "week": "week"})

    assert _column(adapted, "Cases") == [5, 7]
    assert _column(adapted, "E") == [100, 200]
    assert _column(adapted, "ID_spat") == ["a", "b"]
    assert _column(adapted, "ID_year") == [2020, 2021]
    assert _column(adapted, "month") == [4, 12]
    # chap-core only derives week numbers for weekly data.
    assert "week" not in adapted.columns
    # Source columns are kept.
    assert _column(adapted, "disease_cases") == [5, 7]


@pytest.mark.parametrize(
    ("period", "week", "year"),
    [
        # Values pandas gives chap-core for Period(..., freq="W"): ISO week of the start, year of the end.
        ("2003-12-29/2004-01-04", 1, 2004),
        ("2020-12-28/2021-01-03", 53, 2021),
        ("2019-12-30/2020-01-05", 1, 2020),
        ("2021-01-04/2021-01-10", 1, 2021),
        ("2020-W04", 4, 2020),
        ("2020-S07", 7, 2020),
        ("2020W09", 9, 2020),
        ("2020SunW11", 11, 2020),
    ],
)
def test_apply_adapters_derives_weekly_period_parts(period: str, week: int, year: int) -> None:
    frame = DataFrame(columns=["time_period", "location"], data=[[period, "a"]])
    adapted = apply_adapters(frame, {"week": "week", "ID_year": "year", "month": "month"})

    assert _column(adapted, "week") == [week]
    assert _column(adapted, "ID_year") == [year]
    assert "month" not in adapted.columns


def test_apply_adapters_skips_missing_target_in_future_frame() -> None:
    future = DataFrame(columns=["time_period", "location", "population"], data=[["2020-05", "a", 100]])
    adapted = apply_adapters(future, EWARS_ADAPTERS)

    assert "Cases" not in adapted.columns
    assert _column(adapted, "E") == [100]


def test_apply_adapters_identity_and_overwrite() -> None:
    frame = DataFrame(columns=["time_period", "rainfall", "rainsum"], data=[["2020-01", 1.5, 0.0]])
    adapted = apply_adapters(frame, {"rainfall": "rainfall", "rainsum": "rainfall"})

    assert adapted.columns == ["time_period", "rainfall", "rainsum"]
    assert _column(adapted, "rainsum") == [1.5]


def test_apply_adapters_without_mapping_returns_frame_unchanged() -> None:
    frame = DataFrame(columns=["time_period"], data=[["2020-01"]])
    assert apply_adapters(frame, {}) is frame


async def test_shell_runner_writes_adapted_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    runner: ShellModelRunner[BaseConfig] = ShellModelRunner(
        train_command="cp {data_file} model",
        predict_command="cp future.csv {output_file}",
        adapters=EWARS_ADAPTERS,
    )
    data = DataFrame(
        columns=["time_period", "location", "disease_cases", "population"],
        data=[["2020-01", "a", 3, 50], ["2020-02", "a", 4, 50]],
    )

    result = await runner.on_train(BaseConfig(), data)
    workspace = Path(result["workspace_dir"])
    try:
        assert result["exit_code"] == 0, result["stderr"]
        with (workspace / "data.csv").open(newline="") as handle:
            header = next(csv.reader(handle))
        assert header == ["time_period", "location", "disease_cases", "population", "Cases", "E", "ID_year", "ID_spat"]

        future = DataFrame(columns=["time_period", "location", "population"], data=[["2020-03", "a", 50]])
        prediction = await runner.on_predict(BaseConfig(), {"workspace_dir": str(workspace)}, data, future)
        try:
            assert prediction["exit_code"] == 0, prediction["stderr"]
            assert prediction["content"].columns == ["time_period", "location", "population", "E", "ID_year", "ID_spat"]
        finally:
            shutil.rmtree(prediction["workspace_dir"], ignore_errors=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
