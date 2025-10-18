"""REST API router for ML train/predict operations."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, status
from opentelemetry.metrics import Counter
from servicekit.api.monitoring import get_meter
from servicekit.api.router import Router

from .manager import MLManager
from .schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse

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
        **kwargs: Any,
    ) -> None:
        """Initialize ML router with manager factory."""
        self.manager_factory = manager_factory
        super().__init__(prefix=prefix, tags=tags, **kwargs)

    def _register_routes(self) -> None:
        """Register ML train and predict routes."""
        from fastapi import HTTPException

        manager_factory = self.manager_factory

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
