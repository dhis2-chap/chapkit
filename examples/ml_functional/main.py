"""Basic ML example with LinearRegression for disease prediction.

This example demonstrates:
- Defining a config schema for ML parameters
- Creating train/predict functions using sklearn
- Using FunctionalModelRunner to wrap the functions
- Building a service with .with_ml() for train/predict endpoints
- Prometheus metrics and monitoring integration
"""

from typing import Any

import structlog
from geojson_pydantic import FeatureCollection
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class DiseaseConfig(BaseConfig):
    """Configuration for disease prediction model."""

    # Add any model-specific parameters here if needed
    # For this simple example, we don't need any extra config


async def on_train(
    config: DiseaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a linear regression model for disease prediction."""
    # Convert servicekit DataFrame to pandas for sklearn
    df = data.to_pandas()

    features = ["rainfall", "mean_temperature"]
    X = df[features]
    y = df["disease_cases"]
    y = y.fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    log.info("model_trained", features=features, coefficients=list(zip(features, model.coef_)))

    return model


async def on_predict(
    config: DiseaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Make predictions using the trained model."""
    # Convert to pandas for sklearn
    future_df = future.to_pandas()

    X = future_df[["rainfall", "mean_temperature"]]
    y_pred = model.predict(X)
    future_df["sample_0"] = y_pred

    log.info("predictions_made", sample_count=len(y_pred), mean_prediction=y_pred.mean())

    # Convert back to servicekit DataFrame
    return DataFrame.from_pandas(future_df)


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
    level_labels={0: "ml_training", 1: "ml_prediction"},
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

    run_app("main:app")
