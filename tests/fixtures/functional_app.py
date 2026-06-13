"""Minimal functional ML service factory used as a fixture by integration tests."""

from typing import Any

from fastapi import FastAPI
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner


class FunctionalAppConfig(BaseConfig):
    """Configuration for the functional fixture service."""

    prediction_periods: int = 3


async def on_train(
    config: FunctionalAppConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a trivial model that remembers the mean of disease_cases."""
    cases_index = data.columns.index("disease_cases")
    cases = [row[cases_index] for row in data.data]
    return {"mean_cases": sum(cases) / len(cases) if cases else 0.0}


async def on_predict(
    config: FunctionalAppConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Predict the training mean for every future row."""
    columns = [*future.columns, "sample_0"]
    rows = [[*row, model["mean_cases"]] for row in future.data]
    return DataFrame(columns=columns, data=rows)


def build_functional_app() -> FastAPI:
    """Build a minimal functional ML service exercising the train/predict workflow."""
    info = MLServiceInfo(
        id="functional-fixture-service",
        display_name="Functional Fixture Service",
        model_metadata=ModelMetadata(
            author="Test Suite",
            author_assessed_status=AssessedStatus.yellow,
            contact_email="test@example.com",
        ),
        period_type=PeriodType.monthly,
    )
    hierarchy = ArtifactHierarchy(
        name="functional_fixture",
        level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
    )
    runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)
    return MLServiceBuilder(
        info=info,
        config_schema=FunctionalAppConfig,
        hierarchy=hierarchy,
        runner=runner,
    ).build()
