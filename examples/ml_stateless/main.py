"""Stateless ML example: seasonal-baseline predictor without a train step.

This example demonstrates:
- A rule-based predictor that does NOT need a train step (no model is learned).
- Using FunctionalModelRunner with only on_predict.
- MLServiceBuilder auto-picking the stateless default hierarchy
  (level 0: ml_prediction, level 1: ml_prediction_workspace).
- POST /api/v1/ml/$predict with config_id in the body instead of artifact_id.
"""

import structlog
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class SeasonalBaselineConfig(BaseConfig):
    """Configuration for the seasonal-baseline predictor."""

    prediction_periods: int = 3
    target_column: str = "disease_cases"


async def on_predict(
    config: SeasonalBaselineConfig,
    model: None,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Predict each future period as the mean of the historic target column."""
    historic_df = historic.to_pandas()
    future_df = future.to_pandas()

    column = config.target_column
    baseline = float(historic_df[column].fillna(0).mean()) if column in historic_df.columns else 0.0

    future_df["sample_0"] = baseline

    log.info(
        "stateless_predictions_made",
        sample_count=len(future_df),
        baseline=baseline,
        column=column,
    )

    return DataFrame.from_pandas(future_df)


info = MLServiceInfo(
    id="seasonal-baseline-service",
    display_name="Seasonal Baseline Service",
    version="1.0.0",
    description="Rule-based stateless predictor that returns the historic mean as the forecast.",
    model_metadata=ModelMetadata(
        author="ML Team",
        author_assessed_status=AssessedStatus.yellow,
        contact_email="ml-team@example.com",
    ),
    period_type=PeriodType.monthly,
)

runner = FunctionalModelRunner(on_predict=on_predict)

app = MLServiceBuilder(
    info=info,
    config_schema=SeasonalBaselineConfig,
    runner=runner,
).build()


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
