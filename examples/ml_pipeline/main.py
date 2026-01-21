"""ML pipeline example: config, artifact hierarchies, and pandas dataframes."""

from __future__ import annotations

import asyncio
from typing import Iterable

import pandas as pd
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import (
    BaseConfig,
    ConfigIn,
    ConfigManager,
    ConfigOut,
    ConfigRepository,
)
from chapkit.artifact import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactRepository,
)
from chapkit.data import DataFrame

ML_CONFIG_ID = ULID.from_str("01K72P60ZNX2PJ6QJWZK7RMCRV")
TRAIN_ROOT_ID = ULID.from_str("01K72P5N5KCRM6MD3BRE4P07N5")
PREDICTION_ARTIFACT_IDS = [
    ULID.from_str("01K72P5N5KCRM6MD3BRE4P07N6"),
    ULID.from_str("01K72P5N5KCRM6MD3BRE4P07N7"),
]


class MLConfig(BaseConfig):
    """Machine learning model configuration with training hyperparameters."""

    prediction_periods: int = 3
    model_type: str
    learning_rate: float
    batch_size: int
    epochs: int


def _make_prediction_frame(run_name: str, probabilities: Iterable[float]) -> pd.DataFrame:
    """Create a DataFrame from prediction probabilities with binary classification."""
    scores = list(probabilities)
    return pd.DataFrame(
        {
            "run": [run_name] * len(scores),
            "prediction": [1 if p >= 0.5 else 0 for p in scores],
            "probability": scores,
        }
    )


async def main() -> None:
    """Demonstrate ML pipeline with config and hierarchical artifacts storing pandas DataFrames."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    hierarchy = ArtifactHierarchy(name="ml_pipeline", level_labels={0: "train", 1: "predict"})

    try:
        async with db.session() as session:
            config_repo = ConfigRepository(session)
            artifact_repo = ArtifactRepository(session)

            config_manager = ConfigManager[MLConfig](config_repo, MLConfig)
            artifact_manager = ArtifactManager(artifact_repo, hierarchy=hierarchy)

            config: ConfigOut[MLConfig] = await config_manager.save(
                ConfigIn[MLConfig](
                    id=ML_CONFIG_ID,
                    name="fraud_detector_v1",
                    data=MLConfig(model_type="xgboost", learning_rate=0.05, batch_size=512, epochs=30),
                )
            )
            print(f"Saved MLConfig '{config.name}' (model={config.data.model_type})")

            # Level 0: training dataset snapshot
            training_df = pd.DataFrame(
                {
                    "feature_a": [0.1, 0.7, 1.6, 0.4],
                    "feature_b": [3.5, 1.2, 0.4, 2.9],
                    "label": [0, 1, 1, 0],
                }
            )
            training_payload = DataFrame.from_pandas(training_df).model_dump()
            root = await artifact_manager.save(
                ArtifactIn(id=TRAIN_ROOT_ID, data={"stage": "train", "payload": training_payload})
            )
            print(f"Stored training artifact (ULID: {root.id})")

            # Level 1: predictions from different scoring batches
            prediction_runs = [
                ("batch_2024_01_05", [0.11, 0.83, 0.67]),
                ("batch_2024_01_12", [0.42, 0.91, 0.35]),
            ]

            for idx, (run_name, probabilities) in enumerate(prediction_runs):
                predictions_df = _make_prediction_frame(run_name, probabilities)
                prediction_payload = DataFrame.from_pandas(predictions_df).model_dump()
                child = await artifact_manager.save(
                    ArtifactIn(
                        id=PREDICTION_ARTIFACT_IDS[idx],
                        parent_id=root.id,
                        data={"stage": "predict", "run": run_name, "payload": prediction_payload},
                    )
                )
                print(f"  - Added prediction artifact '{run_name}' (ULID: {child.id})")

            # Rehydrate hierarchy to verify levels and data.
            tree = await artifact_manager.build_tree(root.id)
            if not tree:
                raise RuntimeError("Expected to rebuild artifact tree")

            def show_prediction(node_id: str, node_data: dict[str, object]) -> None:
                """Display prediction artifact as a DataFrame."""
                payload = node_data.get("payload")
                if not isinstance(payload, dict):
                    return
                df_schema = DataFrame.model_validate(payload)
                df = df_schema.to_pandas()
                print(f"\nPrediction artifact {node_id}:")
                print(df.to_string(index=False))

            print("\nHierarchy:")
            print(f"- {tree.level_label} (level {tree.level}) id={tree.id}")

            root_payload = tree.data.get("payload") if isinstance(tree.data, dict) else None
            if isinstance(root_payload, dict):
                train_df = DataFrame.model_validate(root_payload).to_pandas()
                print("\nTraining snapshot:")
                print(train_df.to_string(index=False))

            for node in tree.children or []:
                label = node.level_label or f"level_{node.level}"
                print(f"- {label} (level {node.level}) id={node.id}")
                if isinstance(node.data, dict):
                    show_prediction(str(node.id), node.data)

    finally:
        await db.dispose()


if __name__ == "__main__":
    asyncio.run(main())
