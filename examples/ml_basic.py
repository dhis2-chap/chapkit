"""Basic ML example with LinearRegression for disease prediction.

This example demonstrates:
- Defining a config schema for ML parameters
- Creating train/predict functions using sklearn
- Using FunctionalModelRunner to wrap the functions
- Building a service with .with_ml() for train/predict endpoints

Run with: fastapi dev examples/ml_basic.py
"""

from typing import Any

import pandas as pd
import structlog
from geojson_pydantic import FeatureCollection
from servicekit.artifact import ArtifactHierarchy
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class DiseaseConfig(BaseConfig):
    """Configuration for disease prediction model."""

    # Add any model-specific parameters here if needed
    # For this simple example, we don't need any extra config


async def on_train(
    config: DiseaseConfig,
    data: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a linear regression model for disease prediction."""
    features = ["rainfall", "mean_temperature"]

    X = data[features]
    y = data["disease_cases"]
    y = y.fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    log.info("model_trained", features=features, coefficients=list(zip(features, model.coef_)))

    return model


async def on_predict(
    config: DiseaseConfig,
    model: Any,
    historic: pd.DataFrame,
    future: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> pd.DataFrame:
    """Make predictions using the trained model."""
    X = future[["rainfall", "mean_temperature"]]

    y_pred = model.predict(X)
    future["sample_0"] = y_pred

    log.info("predictions_made", sample_count=len(y_pred), mean_prediction=y_pred.mean())

    return future


# Create ML service info with metadata
info = MLServiceInfo(
    display_name="Disease Prediction ML Service",
    version="1.0.0",
    summary="ML service for disease prediction using weather data",
    description="Train and predict disease cases based on rainfall and temperature data using Linear Regression",
    author="ML Team",
    author_assessed_status=AssessedStatus.yellow,
    contact_email="ml-team@example.com",
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={0: "trained_model", 1: "predictions"},
)

# Create functional model runner
runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)

# Build the FastAPI application
app = (
    MLServiceBuilder(
        info=info,
        config_schema=DiseaseConfig,
        hierarchy=HIERARCHY,
        runner=runner,
    )
    .with_monitoring()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("ml_basic:app")
