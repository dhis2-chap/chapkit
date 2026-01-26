"""FastAPI service with read-only artifact API, hierarchies, and non-JSON payloads."""

from __future__ import annotations

from typing import TypedDict

from fastapi import FastAPI
from servicekit import Database
from ulid import ULID

from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.artifact import ArtifactHierarchy, ArtifactIn, ArtifactManager, ArtifactRepository


class MockLinearModel:
    """Mock linear regression model for demonstrating non-JSON artifact payloads."""

    def __init__(self, *, coefficients: tuple[float, ...], intercept: float) -> None:
        """Initialize the model with coefficients and intercept."""
        self.coefficients = coefficients
        self.intercept = intercept

    def predict(self, features: tuple[float, ...]) -> float:
        """Predict output using linear combination of features and intercept."""
        return sum(w * x for w, x in zip(self.coefficients, features)) + self.intercept

    def __repr__(self) -> str:
        """Return string representation of the model with formatted coefficients and intercept."""
        coefs = ", ".join(f"{c:.3f}" for c in self.coefficients)
        return f"MockLinearModel(coefficients=({coefs}), intercept={self.intercept:.3f})"


class ArtifactSeed(TypedDict):
    """TypedDict for artifact seed data with id and payload."""

    id: str
    data: object


class StageSeed(TypedDict):
    """TypedDict for stage seed data with id, metadata, and child artifacts."""

    id: str
    data: dict[str, str]
    artifacts: list[ArtifactSeed]


class PipelineSeed(TypedDict):
    """TypedDict for pipeline seed data with label, root artifact, and stages."""

    label: str
    root: ArtifactSeed
    stages: list[StageSeed]


EXPERIMENT_HIERARCHY = ArtifactHierarchy(
    name="ml_experiment",
    level_labels={0: "experiment", 1: "stage", 2: "artifact"},
)

PIPELINES: tuple[PipelineSeed, ...] = (
    {
        "label": "experiment_alpha",
        "root": {
            "id": "01K72P5N5KCRM6MD3BRE4P07NB",
            "data": {"name": "experiment_alpha", "stage": "train", "status": "succeeded"},
        },
        "stages": [
            {
                "id": "01K72P5N5KCRM6MD3BRE4P07NC",
                "data": {"stage": "feature_engineering"},
                "artifacts": [
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07ND",
                        "data": {"kind": "dataset", "path": "alpha/features.parquet"},
                    },
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07NE",
                        "data": {"kind": "notebook", "path": "alpha/features.ipynb"},
                    },
                ],
            },
            {
                "id": "01K72P5N5KCRM6MD3BRE4P07NF",
                "data": {"stage": "model_training"},
                "artifacts": [
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07NG",
                        "data": {"kind": "model", "format": "pickle", "path": "alpha/model.pkl"},
                    },
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07NH",
                        "data": {"kind": "metrics", "format": "json", "path": "alpha/metrics.json"},
                    },
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07NJ",
                        "data": MockLinearModel(
                            coefficients=(0.42, 0.18, -0.07),
                            intercept=0.12,
                        ),
                    },
                ],
            },
        ],
    },
    {
        "label": "experiment_beta",
        "root": {
            "id": "01K72P5N5KCRM6MD3BRE4P07NK",
            "data": {"name": "experiment_beta", "stage": "train", "status": "running"},
        },
        "stages": [
            {
                "id": "01K72P5N5KCRM6MD3BRE4P07NM",
                "data": {"stage": "data_validation"},
                "artifacts": [
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07NN",
                        "data": {"kind": "report", "path": "beta/validation_report.html"},
                    },
                ],
            },
            {
                "id": "01K72P5N5KCRM6MD3BRE4P07NP",
                "data": {"stage": "batch_scoring"},
                "artifacts": [
                    {
                        "id": "01K72P5N5KCRM6MD3BRE4P07NQ",
                        "data": {"kind": "predictions", "format": "parquet", "path": "beta/preds.parquet"},
                    },
                    {
                        "id": "01K72P60ZNX2PJ6QJWZK7RMCRT",
                        "data": {"kind": "log", "path": "beta/batch.log"},
                    },
                ],
            },
        ],
    },
)


class PipelineMetadata(BaseConfig):
    """Configuration schema for pipeline metadata including owner and notification settings."""

    prediction_periods: int = 3
    owner: str
    stage: str
    notification_channel: str


class ArtifactServiceInfo(ServiceInfo):
    """Extended service info with additional fields for artifact service metadata."""

    author: str | None = None
    maintainer_contact: str | None = None
    hierarchy: dict[str, object]
    pipelines: list[str]
    non_json_payload: str


async def seed_artifacts(app: FastAPI) -> None:
    """Startup hook that seeds the database with predefined artifact hierarchies."""
    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        manager = ArtifactManager(ArtifactRepository(session), hierarchy=EXPERIMENT_HIERARCHY)
        await manager.delete_all()

        for pipeline in PIPELINES:
            root_seed = pipeline["root"]
            root = await manager.save(ArtifactIn(id=ULID.from_str(root_seed["id"]), data=root_seed["data"]))

            for stage in pipeline["stages"]:
                stage_node = await manager.save(
                    ArtifactIn(
                        id=ULID.from_str(stage["id"]),
                        parent_id=root.id,
                        data=stage["data"],
                    )
                )

                for artifact_seed in stage["artifacts"]:
                    await manager.save(
                        ArtifactIn(
                            id=ULID.from_str(artifact_seed["id"]),
                            parent_id=stage_node.id,
                            data=artifact_seed["data"],
                        )
                    )


info = ArtifactServiceInfo(
    display_name="Chapkit Artifact Service",
    summary="Artifact CRUD and tree operations example",
    author="Morten Hansen",
    maintainer_contact="morten@dhis2.org",
    contact={"email": "morten@dhis2.org"},
    hierarchy={
        "name": EXPERIMENT_HIERARCHY.name,
        "level_labels": dict(EXPERIMENT_HIERARCHY.level_labels),
    },
    pipelines=[pipeline["label"] for pipeline in PIPELINES],
    non_json_payload="MockLinearModel",
)

app: FastAPI = (
    ServiceBuilder(info=info)
    .with_landing_page()
    .with_health()
    .with_system()
    .with_config(PipelineMetadata)
    .with_artifacts(
        hierarchy=EXPERIMENT_HIERARCHY,
        allow_create=False,
        allow_update=False,
        allow_delete=False,
    )
    .on_startup(seed_artifacts)
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
