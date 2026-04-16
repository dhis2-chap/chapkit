"""Tests for optional on_validate_* callbacks on FunctionalModelRunner and ShellModelRunner."""

from __future__ import annotations

from typing import Any

import pytest
from geojson_pydantic import FeatureCollection

from chapkit.config import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner, ShellModelRunner, ValidationDiagnostic


class DummyConfig(BaseConfig):
    """Minimal config used for runner callback tests."""

    prediction_periods: int = 3
    min_rows: int = 5


async def _noop_train(config: DummyConfig, data: DataFrame, geo: FeatureCollection | None = None) -> Any:
    """Placeholder train callback so the runner can be constructed."""
    return {}


async def _noop_predict(
    config: DummyConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Placeholder predict callback so the runner can be constructed."""
    return future


@pytest.mark.asyncio
async def test_functional_runner_defaults_to_empty_diagnostics() -> None:
    """Without callbacks, FunctionalModelRunner returns empty validation diagnostics."""
    runner = FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict)
    config = DummyConfig(prediction_periods=3)
    data = DataFrame(columns=["rainfall"], data=[[1.0]])

    assert await runner.on_validate_train(config, data) == []
    assert await runner.on_validate_predict(config, data, data) == []


@pytest.mark.asyncio
async def test_functional_runner_calls_validate_train_callback() -> None:
    """Callback output is returned verbatim from on_validate_train."""

    async def on_validate_train(
        config: DummyConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        if len(data) < config.min_rows:
            return [
                ValidationDiagnostic(
                    severity="error",
                    code="too_few_rows",
                    message=f"need {config.min_rows}, got {len(data)}",
                    field="data",
                )
            ]
        return []

    runner = FunctionalModelRunner(
        on_train=_noop_train,
        on_predict=_noop_predict,
        on_validate_train=on_validate_train,
    )
    config = DummyConfig(min_rows=5)

    short = DataFrame(columns=["rainfall"], data=[[1.0]])
    diagnostics = await runner.on_validate_train(config, short)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "too_few_rows"

    enough = DataFrame(columns=["rainfall"], data=[[1.0]] * 5)
    assert await runner.on_validate_train(config, enough) == []


@pytest.mark.asyncio
async def test_functional_runner_calls_validate_predict_callback() -> None:
    """Callback output is returned verbatim from on_validate_predict."""

    async def on_validate_predict(
        config: DummyConfig,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        return [
            ValidationDiagnostic(
                severity="warning",
                code="runner_saw_predict",
                message="runner hook invoked",
                field=None,
            )
        ]

    runner = FunctionalModelRunner(
        on_train=_noop_train,
        on_predict=_noop_predict,
        on_validate_predict=on_validate_predict,
    )
    config = DummyConfig()
    historic = DataFrame(columns=["x"], data=[[1.0]])
    future = DataFrame(columns=["x"], data=[[2.0]])

    diagnostics = await runner.on_validate_predict(config, historic, future)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "runner_saw_predict"
    assert diagnostics[0].severity == "warning"


@pytest.mark.asyncio
async def test_shell_runner_defaults_to_empty_diagnostics() -> None:
    """Without callbacks, ShellModelRunner also returns empty diagnostics."""
    runner: ShellModelRunner[DummyConfig] = ShellModelRunner(
        train_command="true",
        predict_command="true",
    )
    config = DummyConfig()
    data = DataFrame(columns=["x"], data=[[1.0]])

    assert await runner.on_validate_train(config, data) == []
    assert await runner.on_validate_predict(config, data, data) == []


@pytest.mark.asyncio
async def test_shell_runner_calls_validate_callbacks() -> None:
    """ShellModelRunner delegates to Python validate callbacks when provided."""

    async def on_validate_train(
        config: DummyConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        return [
            ValidationDiagnostic(
                severity="error",
                code="shell_validate_train",
                message="called",
                field="data",
            )
        ]

    async def on_validate_predict(
        config: DummyConfig,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        return [
            ValidationDiagnostic(
                severity="error",
                code="shell_validate_predict",
                message="called",
                field="future",
            )
        ]

    runner: ShellModelRunner[DummyConfig] = ShellModelRunner(
        train_command="true",
        predict_command="true",
        on_validate_train=on_validate_train,
        on_validate_predict=on_validate_predict,
    )
    config = DummyConfig()
    data = DataFrame(columns=["x"], data=[[1.0]])

    train_diagnostics = await runner.on_validate_train(config, data)
    assert train_diagnostics[0].code == "shell_validate_train"

    predict_diagnostics = await runner.on_validate_predict(config, data, data)
    assert predict_diagnostics[0].code == "shell_validate_predict"
