"""REST API router for ML train/predict operations."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, status
from opentelemetry.metrics import Counter
from servicekit.api.monitoring import get_meter
from servicekit.api.router import Router

from .manager import MLManager
from .schemas import (
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
    ValidateRequest,
    ValidationResponse,
)

# Lazily initialized counters (initialized after monitoring setup)
_train_counter: Counter | None = None
_predict_counter: Counter | None = None


def _get_counters() -> tuple[Counter, Counter]:
    """Get or create ML metrics counters (lazy initialization)."""
    global _train_counter, _predict_counter

    if _train_counter is None or _predict_counter is None:
        meter = get_meter("chapkit.ml")
        _train_counter = meter.create_counter(
            name="ml_train_jobs_total",
            description="Total number of ML training jobs submitted",
            unit="1",
        )
        _predict_counter = meter.create_counter(
            name="ml_predict_jobs_total",
            description="Total number of ML prediction jobs submitted",
            unit="1",
        )

    return _train_counter, _predict_counter


class MLRouter(Router):
    """Router with $train and $predict collection operations."""

    def __init__(
        self,
        prefix: str,
        tags: list[str],
        manager_factory: Any,
        sample_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ML router with manager factory."""
        self.manager_factory = manager_factory
        self.sample_metadata = sample_metadata or {}
        super().__init__(prefix=prefix, tags=tags, **kwargs)

    def _register_routes(self) -> None:
        """Register ML train and predict routes."""
        from typing import Literal

        from fastapi import HTTPException

        manager_factory = self.manager_factory
        sample_metadata = self.sample_metadata

        @self.router.get(
            "/$generate-sample-data",
            summary="Generate a sample train or predict payload",
            description=(
                "Return a ready-to-submit sample payload built with chapkit's synthetic data "
                "generator. Useful for trying out $train and $predict from the console."
            ),
        )
        async def generate_sample_data(
            kind: Literal["train", "predict"] = "train",
            config_id: str | None = None,
            num_locations: int = 5,
            num_periods: int = 50,
            num_features: int = 3,
            period_type: Literal["monthly", "weekly"] | None = None,
            geo_type: Literal["polygon", "point"] = "polygon",
            include_geo: bool | None = None,
            seed: int = -1,
        ) -> dict[str, Any]:
            """Build a sample train/predict payload from tunable data-generator parameters."""
            from chapkit.data.generator import TestDataGenerator

            # A negative seed means "fresh data each call"; a concrete seed is reproducible.
            generator = TestDataGenerator(seed=None if seed < 0 else seed)
            required_covariates = list(sample_metadata.get("required_covariates") or [])
            requires_geo = bool(sample_metadata.get("requires_geo", False))
            resolved_period = period_type or sample_metadata.get("period_type") or "monthly"
            want_geo = requires_geo if include_geo is None else include_geo
            geo = generator.generate_geo_data(num_features=num_locations, geo_type=geo_type) if want_geo else None

            if kind == "predict":
                historic, future = generator.generate_prediction_data(
                    num_locations=num_locations,
                    num_periods=max(1, num_periods),
                    num_features=num_features,
                    required_covariates=required_covariates,
                    period_type=resolved_period,
                )
                payload: dict[str, Any] = {"historic": historic, "future": future}
                if geo is not None:
                    payload["geo"] = geo
                return payload

            data = generator.generate_training_data(
                num_locations=num_locations,
                num_periods=max(1, num_periods),
                num_features=num_features,
                required_covariates=required_covariates,
                period_type=resolved_period,
            )
            payload = {"data": data}
            if config_id:
                payload["config_id"] = config_id
            if geo is not None:
                payload["geo"] = geo
            return payload

        @self.router.post(
            "/$train",
            response_model=TrainResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="Train model",
            description="Submit a training job to the scheduler",
        )
        async def train(
            request: TrainRequest,
            manager: MLManager = Depends(manager_factory),
        ) -> TrainResponse:
            """Train a model asynchronously and return job/artifact IDs."""
            try:
                response = await manager.execute_train(request)
                train_counter, _ = _get_counters()
                train_counter.add(1)
                return response
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )
            except RuntimeError as e:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(e),
                )

        @self.router.post(
            "/$predict",
            response_model=PredictResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="Make predictions",
            description="Submit a prediction job to the scheduler",
        )
        async def predict(
            request: PredictRequest,
            manager: MLManager = Depends(manager_factory),
        ) -> PredictResponse:
            """Make predictions asynchronously and return job/artifact IDs."""
            try:
                response = await manager.execute_predict(request)
                _, predict_counter = _get_counters()
                predict_counter.add(1)
                return response
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )
            except RuntimeError as e:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(e),
                )

        @self.router.post(
            "/$validate",
            response_model=ValidationResponse,
            status_code=status.HTTP_200_OK,
            summary="Validate a train or predict payload",
            description="Run framework and runner validations without executing.",
        )
        async def validate(
            request: ValidateRequest,
            manager: MLManager = Depends(manager_factory),
        ) -> ValidationResponse:
            """Return structured diagnostics for a train or predict payload."""
            try:
                return await manager.validate(request)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Validation failed: {type(exc).__name__}",
                )
