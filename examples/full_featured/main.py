"""Complete showcase of all chapkit features in one service.

This example demonstrates EVERY feature available in chapkit:
- Health checks (with custom checks)
- System info endpoints
- Config management (typed validation)
- Artifacts (hierarchical storage)
- Config-artifact linking
- Job scheduling (async execution)
- Structured logging
- Monitoring (Prometheus metrics)
- Custom routers
- Startup/shutdown hooks
- Landing page
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from servicekit import Database
from servicekit.api import Router
from servicekit.api.routers.health import HealthState
from ulid import ULID

from chapkit import (
    BaseConfig,
    ConfigIn,
    ConfigManager,
    ConfigRepository,
)
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.api.dependencies import get_config_manager
from chapkit.artifact import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactRepository,
)

# ==================== Configuration Schema ====================


class MLPipelineConfig(BaseConfig):
    """Configuration schema for ML pipeline settings."""

    model_type: str = Field(description="Type of ML model (e.g., 'random_forest', 'xgboost')")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate for training")
    max_epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


# ==================== Artifact Hierarchy ====================


PIPELINE_HIERARCHY = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={
        0: "experiment",  # Top-level experiments
        1: "training",  # Training runs within experiment
        2: "evaluation",  # Evaluation results within training
        3: "prediction",  # Predictions within evaluation
    },
)


# ==================== Custom Health Check ====================


async def check_external_service() -> tuple[HealthState, str | None]:
    """Custom health check for external service dependency."""
    try:
        # Simulate checking external service (e.g., ML model API, S3, etc.)
        await asyncio.sleep(0.01)  # Simulate network call
        return (HealthState.HEALTHY, None)
    except Exception as e:
        return (HealthState.UNHEALTHY, f"External service unavailable: {str(e)}")


# ==================== Custom Router ====================


class StatsResponse(BaseModel):
    """Statistics response model."""

    total_configs: int = Field(description="Total number of configurations")
    total_artifacts: int = Field(description="Total number of artifacts")
    total_tasks: int = Field(description="Total number of task templates")
    service_version: str = Field(description="Service version")


class CustomStatsRouter(Router):
    """Custom router demonstrating extensibility."""

    def _register_routes(self) -> None:
        """Register routes for service statistics."""

        @self.router.get("", response_model=StatsResponse, summary="Get service statistics")
        async def get_stats(  # pyright: ignore[reportUnusedFunction]
            config_manager: Annotated[ConfigManager[MLPipelineConfig], Depends(get_config_manager)],
        ) -> StatsResponse:
            """Get comprehensive service statistics."""
            # Count configs
            configs = await config_manager.find_all()

            # Note: In real implementation, you'd inject other managers too
            # This is simplified to show the pattern

            return StatsResponse(
                total_configs=len(configs),
                total_artifacts=0,  # Would query ArtifactManager
                total_tasks=0,  # Would query TaskManager
                service_version="2.0.0",
            )


# ==================== Startup/Shutdown Hooks ====================


async def startup_hook(app: FastAPI) -> None:
    """Startup hook to seed example data."""
    print("🚀 Service starting up...")

    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        # Seed example config
        config_repo = ConfigRepository(session)
        config_manager = ConfigManager[MLPipelineConfig](config_repo, MLPipelineConfig)

        existing_configs = await config_manager.find_all()
        if len(existing_configs) == 0:
            await config_manager.save(
                ConfigIn[MLPipelineConfig](
                    id=ULID.from_str("01JCSEED00C0NF1GEXAMP1E001"),
                    name="production_pipeline",
                    data=MLPipelineConfig(
                        model_type="xgboost",
                        learning_rate=0.01,
                        max_epochs=500,
                        batch_size=64,
                        early_stopping=True,
                        random_seed=42,
                    ),
                )
            )
            print("  ✓ Seeded example config: production_pipeline")

        # Seed example artifact
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo, hierarchy=PIPELINE_HIERARCHY)

        existing_artifacts = await artifact_manager.find_all()
        if len(existing_artifacts) == 0:
            await artifact_manager.save(
                ArtifactIn(
                    id=ULID.from_str("01JCSEED00ART1FACTEXMP1001"),
                    data={
                        "experiment_name": "baseline_experiment",
                        "model_metrics": {"accuracy": 0.95, "f1_score": 0.93},
                        "dataset_info": {"train_size": 10000, "test_size": 2000},
                    },
                    parent_id=None,
                )
            )
            print("  ✓ Seeded example artifact: baseline_experiment")

    print("✅ Startup complete!\n")


async def shutdown_hook(app: FastAPI) -> None:
    """Shutdown hook for cleanup."""
    print("👋 Service shutting down...")
    # Perform cleanup if needed
    print("✅ Shutdown complete!")


# ==================== Service Configuration ====================


info = ServiceInfo(
    display_name="Complete Feature Showcase",
    summary="Comprehensive example demonstrating ALL chapkit features",
    version="2.0.0",
    description="""Complete Chapkit Feature Showcase.""",
    contact={"name": "Chapkit Team", "email": "support@example.com"},
    license_info={"name": "MIT"},
)

# ==================== Application Builder ====================

app = (
    ServiceBuilder(info=info)
    # Core features
    .with_health(
        checks={
            "external_service": check_external_service,
        },
        include_database_check=True,
    )
    .with_system()
    .with_logging()  # Structured logging with request tracing
    .with_monitoring()  # Prometheus metrics + OpenTelemetry
    .with_landing_page()  # Interactive homepage
    # Data management
    .with_config(MLPipelineConfig)
    .with_artifacts(
        hierarchy=PIPELINE_HIERARCHY,
        enable_config_linking=True,  # Enable config-artifact linking
    )
    # Execution
    .with_jobs(max_concurrency=5)  # Job scheduler
    # Extensibility
    .include_router(CustomStatsRouter.create(prefix="/api/v1/stats", tags=["statistics"]))
    # Lifecycle
    .on_startup(startup_hook)
    .on_shutdown(shutdown_hook)
    .build()
)

if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
