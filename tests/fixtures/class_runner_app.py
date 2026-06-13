"""Class-based ML service factory used as a fixture by runner and validate integration tests."""

from typing import Any

from fastapi import FastAPI
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import BaseModelRunner, ValidationDiagnostic


class ClassRunnerConfig(BaseConfig):
    """Configuration for the class-based fixture runner."""

    prediction_periods: int = 3
    min_samples: int = 5
    normalize_features: bool = True


class FixtureModelRunner(BaseModelRunner[ClassRunnerConfig]):
    """Class-based runner exercising lifecycle hooks, shared state, and validation overrides."""

    def __init__(self) -> None:
        """Initialize runner with shared state."""
        self.target_name: str = "disease_cases"
        self.initialized: bool = False

    async def on_init(self) -> None:
        """Initialize runner before training or prediction."""
        self.initialized = True

    async def on_cleanup(self) -> None:
        """Clean up resources after training or prediction."""
        self.initialized = False

    async def on_train(
        self,
        config: ClassRunnerConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a trivial mean model, enforcing the config.min_samples threshold."""
        await self.on_init()
        try:
            if len(data) < config.min_samples:
                raise ValueError(f"Insufficient training data: {len(data)} < {config.min_samples}")

            target_index = data.columns.index(self.target_name)
            target = [row[target_index] for row in data.data]
            mean_target = sum(target) / len(target)
            return {
                "mean_target": mean_target,
                "normalized": config.normalize_features,
                "config": config.model_dump(),
            }
        finally:
            await self.on_cleanup()

    async def on_validate_train(
        self,
        config: ClassRunnerConfig,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> list[ValidationDiagnostic]:
        """Report a diagnostic when the training set is below config.min_samples."""
        diagnostics: list[ValidationDiagnostic] = []
        if len(data) < config.min_samples:
            diagnostics.append(
                ValidationDiagnostic.error(
                    code="insufficient_training_samples",
                    message=(
                        f"Training data has {len(data)} rows; at least {config.min_samples} are required by min_samples"
                    ),
                    field="data",
                )
            )
        return diagnostics

    async def on_predict(
        self,
        config: ClassRunnerConfig,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Predict the training mean for every future row."""
        await self.on_init()
        try:
            mean_target = model["mean_target"] if isinstance(model, dict) else 0.0
            columns = [*future.columns, "sample_0"]
            rows = [[*row, mean_target] for row in future.data]
            return DataFrame(columns=columns, data=rows)
        finally:
            await self.on_cleanup()


def build_class_runner_app() -> FastAPI:
    """Build a minimal class-based ML service exercising BaseModelRunner subclassing."""
    info = MLServiceInfo(
        id="class-runner-fixture-service",
        display_name="Class Runner Fixture Service",
        model_metadata=ModelMetadata(
            author="Test Suite",
            author_assessed_status=AssessedStatus.yellow,
            contact_email="test@example.com",
        ),
        period_type=PeriodType.monthly,
    )
    hierarchy = ArtifactHierarchy(
        name="class_runner_fixture",
        level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
    )
    return MLServiceBuilder(
        info=info,
        config_schema=ClassRunnerConfig,
        hierarchy=hierarchy,
        runner=FixtureModelRunner(),
    ).build()
