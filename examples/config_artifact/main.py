"""FastAPI service demonstrating config-artifact linking and hierarchies."""

from __future__ import annotations

import random
from typing import NotRequired, TypedDict

from fastapi import FastAPI
from pydantic import EmailStr
from servicekit import Database
from servicekit.api.routers.health import HealthState
from ulid import ULID

from chapkit import (
    BaseConfig,
    ConfigIn,
    ConfigManager,
    ConfigRepository,
)
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.artifact import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactRepository,
)


class ArtifactSeed(TypedDict):
    """Typed dictionary for seeding artifact hierarchies with parent-child relationships."""

    id: str
    data: dict[str, object]
    children: NotRequired[list["ArtifactSeed"]]


class ExperimentSeed(TypedDict):
    """Typed dictionary for seeding complete experiments with config and artifact tree."""

    config_id: str
    config_name: str
    config_payload: dict[str, object]
    root_artifact: ArtifactSeed


class ExperimentConfig(BaseConfig):
    """Configuration schema for machine learning experiment parameters."""

    prediction_periods: int = 3
    model: str
    learning_rate: float
    epochs: int
    batch_size: int


PIPELINE_HIERARCHY = ArtifactHierarchy(
    name="training_pipeline",
    level_labels={0: "train", 1: "predict", 2: "result"},
)

EXPERIMENTS: tuple[ExperimentSeed, ...] = (
    {
        "config_id": "01K72PWT05GEXK1S24AVKAZ9VE",
        "config_name": "experiment_alpha",
        "config_payload": {
            "model": "xgboost",
            "learning_rate": 0.05,
            "epochs": 50,
            "batch_size": 256,
        },
        "root_artifact": {
            "id": "01K72PWT05GEXK1S24AVKAZ9VF",
            "data": {"stage": "train", "dataset": "alpha_train.parquet"},
            "children": [
                {
                    "id": "01K72PWT05GEXK1S24AVKAZ9VG",
                    "data": {"stage": "predict", "run": "2024-01-05", "path": "alpha/preds_20240105.parquet"},
                    "children": [
                        {
                            "id": "01K72PWT05GEXK1S24AVKAZ9VH",
                            "data": {"stage": "result", "metrics": {"accuracy": 0.91, "f1": 0.88}},
                        }
                    ],
                },
                {
                    "id": "01K72PWT05GEXK1S24AVKAZ9VJ",
                    "data": {"stage": "predict", "run": "2024-01-12", "path": "alpha/preds_20240112.parquet"},
                },
            ],
        },
    },
    {
        "config_id": "01K72PWT05GEXK1S24AVKAZ9VK",
        "config_name": "experiment_beta",
        "config_payload": {
            "model": "lightgbm",
            "learning_rate": 0.02,
            "epochs": 80,
            "batch_size": 512,
        },
        "root_artifact": {
            "id": "01K72PWT05GEXK1S24AVKAZ9VM",
            "data": {"stage": "train", "dataset": "beta_train.parquet"},
            "children": [
                {
                    "id": "01K72PWT05GEXK1S24AVKAZ9VN",
                    "data": {"stage": "predict", "run": "2024-02-01", "path": "beta/preds_20240201.parquet"},
                    "children": [
                        {
                            "id": "01K72PWT05GEXK1S24AVKAZ9VP",
                            "data": {"stage": "result", "metrics": {"accuracy": 0.87, "f1": 0.84}},
                        }
                    ],
                }
            ],
        },
    },
)


class MLServiceInfo(ServiceInfo):
    """Extended service information with ML-specific metadata and configuration details."""

    author: str | None = None
    contact_email: EmailStr | None = None
    hierarchy: dict[str, object]
    configs: list[str]


async def create_artifact_tree(
    manager: ArtifactManager,
    seed: ArtifactSeed,
    parent_id: ULID | None = None,
) -> ULID:
    """Recursively creates an artifact tree from seed data with parent-child relationships."""
    artifact_id = ULID.from_str(seed["id"])
    artifact = await manager.save(
        ArtifactIn(
            id=artifact_id,
            parent_id=parent_id,
            data=seed["data"],
        )
    )

    for child in seed.get("children", []) or []:
        await create_artifact_tree(manager, child, parent_id=artifact.id)

    return artifact.id


async def seed_demo_data(app: FastAPI) -> None:
    """Startup hook that seeds the database with demo experiments and artifact trees."""
    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        config_repo = ConfigRepository(session)
        artifact_repo = ArtifactRepository(session)

        config_manager = ConfigManager[ExperimentConfig](config_repo, ExperimentConfig)
        artifact_manager = ArtifactManager(artifact_repo, hierarchy=PIPELINE_HIERARCHY)

        await artifact_manager.delete_all()
        await config_manager.delete_all()

        for experiment in EXPERIMENTS:
            payload = ExperimentConfig.model_validate(experiment["config_payload"])
            config = await config_manager.save(
                ConfigIn[ExperimentConfig](
                    id=ULID.from_str(experiment["config_id"]),
                    name=experiment["config_name"],
                    data=payload,
                )
            )

            root_id = await create_artifact_tree(artifact_manager, experiment["root_artifact"])
            await config_manager.link_artifact(config.id, root_id)


async def check_flaky_service() -> tuple[HealthState, str | None]:
    """Custom health check that randomly returns healthy, degraded, or unhealthy states."""
    outcome = random.choice([HealthState.HEALTHY, HealthState.DEGRADED, HealthState.UNHEALTHY])

    if outcome == HealthState.HEALTHY:
        return (HealthState.HEALTHY, None)
    elif outcome == HealthState.DEGRADED:
        return (HealthState.DEGRADED, "Service experiencing intermittent issues")
    else:
        return (HealthState.UNHEALTHY, "Service unavailable")


info = MLServiceInfo(
    display_name="Chapkit Config & Artifact Service",
    version="1.0.0",
    summary="Linked config and artifact CRUD example",
    author="Morten Hansen",
    contact_email="morten@dhis2.org",
    contact={"email": "morten@dhis2.org"},
    hierarchy={
        "name": PIPELINE_HIERARCHY.name,
        "level_labels": dict(PIPELINE_HIERARCHY.level_labels),
    },
    configs=[seed["config_name"] for seed in EXPERIMENTS],
)

app: FastAPI = (
    ServiceBuilder(info=info)
    .with_landing_page()
    .with_logging()
    .with_health(checks={"flaky_service": check_flaky_service})
    .with_system()
    .with_config(ExperimentConfig)
    .with_artifacts(hierarchy=PIPELINE_HIERARCHY, enable_config_linking=True)
    .on_startup(seed_demo_data)
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
