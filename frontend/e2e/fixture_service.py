"""Minimal self-contained chapkit ML service used as the Playwright e2e fixture."""

import math
from typing import Any

from geojson_pydantic import FeatureCollection
from pydantic import Field

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import BaseModelRunner

TARGET = "disease_cases"


class DemoConfig(BaseConfig):
    """Configuration for the e2e demo model."""

    prediction_periods: int = Field(default=3, description="Periods to predict into the future")
    seasonal_amplitude: float = Field(default=0.4, description="Seasonal swing applied to predictions")
    region_seasonal: bool = Field(default=False, description="Include region-specific seasonal effects")


class DemoRunner(BaseModelRunner[DemoConfig]):
    """In-process runner so train/predict jobs complete without external tooling."""

    async def on_train(
        self,
        config: DemoConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Learn the mean disease_cases level."""
        cases = [float(v) for v in data.get_column(TARGET) if v is not None]
        mean = sum(cases) / len(cases) if cases else 1.0
        return {"mean": mean}

    async def on_predict(
        self,
        config: DemoConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Forecast a simple seasonal disease_cases series for each future row."""
        mean = float(model["mean"])
        periods = future.get_column("time_period")
        predictions: list[float] = []
        for period in periods:
            tail = str(period).split("-")[-1].lstrip("W")
            month = int(tail) if tail.isdigit() else 1
            seasonal = 1.0 + config.seasonal_amplitude * math.sin(2.0 * math.pi * month / 12.0)
            predictions.append(round(max(0.0, mean * seasonal), 2))
        columns = future.to_dict("list")
        columns[TARGET] = predictions
        return DataFrame.from_dict(columns)


info = MLServiceInfo(
    id="chapkit-e2e-fixture",
    display_name="Chapkit e2e fixture",
    version="0.1.0",
    description="Self-contained chapkit ML service used to drive the console e2e tests.",
    model_metadata=ModelMetadata(
        author="chapkit",
        author_assessed_status=AssessedStatus.green,
        organization="chapkit",
    ),
    period_type=PeriodType.monthly,
    required_covariates=["population", "rainfall", "mean_temperature"],
    min_prediction_periods=0,
    max_prediction_periods=12,
)

HIERARCHY = ArtifactHierarchy(
    name="e2e_fixture",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)

app = MLServiceBuilder(
    info=info,
    config_schema=DemoConfig,
    hierarchy=HIERARCHY,
    runner=DemoRunner(),
).build()
