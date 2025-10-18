"""Example linking configs to artifact hierarchies with tree traversal."""

from __future__ import annotations

import asyncio
from typing import Any, NotRequired, TypedDict, cast

from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactRepository,
    BaseConfig,
    ConfigIn,
    ConfigManager,
    ConfigOut,
    ConfigRepository,
)


class ExperimentConfig(BaseConfig):
    """Experiment configuration with ML model hyperparameters."""

    model: str
    learning_rate: float
    epochs: int
    batch_size: int


class ArtifactNodeSeed(TypedDict):
    """TypedDict for seeding artifact tree nodes with optional children."""

    id: str
    data: dict[str, Any]
    children: NotRequired[list["ArtifactNodeSeed"]]


class ExperimentSeed(TypedDict):
    """TypedDict for seeding complete experiment with config and artifact tree."""

    config_id: str
    config_name: str
    config_payload: ExperimentConfig
    artifact_tree: ArtifactNodeSeed


PIPELINE_HIERARCHY = ArtifactHierarchy(
    name="training_pipeline",
    level_labels={0: "train", 1: "predict", 2: "result"},
)

EXPERIMENTS: tuple[ExperimentSeed, ...] = (
    {
        "config_id": "01K72PPTNHD992PP6MYHM5PP4V",
        "config_name": "experiment_alpha",
        "config_payload": ExperimentConfig(
            model="xgboost",
            learning_rate=0.05,
            epochs=50,
            batch_size=512,
        ),
        "artifact_tree": {
            "id": "01K72PPTNHD992PP6MYHM5PP4W",
            "data": {"stage": "train", "dataset": "alpha_train.parquet"},
            "children": [
                {
                    "id": "01K72PPTNHD992PP6MYHM5PP4X",
                    "data": {"stage": "predict", "run": "2024-01-05", "path": "alpha/preds_20240105.parquet"},
                    "children": [
                        {
                            "id": "01K72PPTNHD992PP6MYHM5PP4Y",
                            "data": {"stage": "result", "metrics": {"accuracy": 0.91, "f1": 0.88}},
                        }
                    ],
                },
                {
                    "id": "01K72PPTNHD992PP6MYHM5PP4Z",
                    "data": {"stage": "predict", "run": "2024-01-12", "path": "alpha/preds_20240112.parquet"},
                },
            ],
        },
    },
    {
        "config_id": "01K72PPTNHD992PP6MYHM5PP50",
        "config_name": "experiment_beta",
        "config_payload": ExperimentConfig(
            model="lightgbm",
            learning_rate=0.02,
            epochs=80,
            batch_size=256,
        ),
        "artifact_tree": {
            "id": "01K72PPTNHD992PP6MYHM5PP51",
            "data": {"stage": "train", "dataset": "beta_train.parquet"},
            "children": [
                {
                    "id": "01K72PPTNHD992PP6MYHM5PP52",
                    "data": {"stage": "predict", "run": "2024-02-01", "path": "beta/preds_20240201.parquet"},
                    "children": [
                        {
                            "id": "01K72PPTNHD992PP6MYHM5PP53",
                            "data": {"stage": "result", "metrics": {"accuracy": 0.87, "f1": 0.84}},
                        }
                    ],
                },
            ],
        },
    },
)


async def create_artifact_tree(
    manager: ArtifactManager,
    seed: ArtifactNodeSeed,
    parent_id: ULID | None = None,
) -> ULID:
    """Recursively create artifact tree from seed data."""
    node_id = ULID.from_str(seed["id"])
    artifact = await manager.save(
        ArtifactIn(
            id=node_id,
            parent_id=parent_id,
            data=seed["data"],
        )
    )

    for child in seed.get("children", []) or []:
        await create_artifact_tree(manager, child, parent_id=artifact.id)

    return artifact.id


def print_tree(node: ConfigOut[ExperimentConfig], tree_root_id: ULID, tree: dict[str, Any]) -> None:
    """Print artifact tree structure with config linkage."""

    def _walk(current: dict[str, Any], indent: int = 0) -> None:
        """Recursively walk and print tree nodes with indentation."""
        padding = "  " * indent
        level_label = current.get("level_label", f"level_{current.get('level', '?')}")
        node_id = current["id"]
        print(f"{padding}- {level_label}: {node_id}")
        data = current.get("data", {})
        if data:
            print(f"{padding}  data: {data}")
        if current.get("config"):
            cfg = current["config"]
            print(f"{padding}  linked config: {cfg['name']} -> {cfg['id']}")
        children = cast(list[dict[str, Any]], current.get("children") or [])
        for child in children:
            _walk(child, indent + 1)

    print(f"\nConfig '{node.name}' -> root artifact {tree_root_id}")
    _walk(tree)


async def main() -> None:
    """Demonstrate linking configs to artifact hierarchies with tree traversal."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    try:
        async with db.session() as session:
            config_repo = ConfigRepository(session)
            artifact_repo = ArtifactRepository(session)

            config_manager = ConfigManager[ExperimentConfig](config_repo, ExperimentConfig)
            artifact_manager = ArtifactManager(artifact_repo, hierarchy=PIPELINE_HIERARCHY, config_repo=config_repo)

            await config_manager.delete_all()
            await artifact_manager.delete_all()

            for experiment in EXPERIMENTS:
                config = await config_manager.save(
                    ConfigIn[ExperimentConfig](
                        id=ULID.from_str(experiment["config_id"]),
                        name=experiment["config_name"],
                        data=experiment["config_payload"],
                    )
                )

                root_id = await create_artifact_tree(artifact_manager, experiment["artifact_tree"])
                await config_manager.link_artifact(config.id, root_id)

                tree = await artifact_manager.build_tree(root_id)
                if tree:
                    print_tree(config, root_id, tree.model_dump(mode="python", warnings=False))

                if tree and tree.children:
                    child_id = tree.children[0].id
                    linked = await config_manager.get_config_for_artifact(child_id, artifact_repo)
                    if linked and linked.data:
                        print(
                            f"  ↳ Child artifact {child_id} inherits config '{linked.name}' (model={linked.data.model})"
                        )

    finally:
        await db.dispose()


if __name__ == "__main__":
    asyncio.run(main())
