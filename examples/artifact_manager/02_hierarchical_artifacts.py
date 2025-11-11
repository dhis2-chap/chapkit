"""Hierarchical artifacts with parent-child relationships.

This example demonstrates:
- Creating parent training artifacts
- Creating child prediction artifacts
- Building artifact trees
- Navigating hierarchies
"""

import asyncio
from datetime import datetime

from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from chapkit.artifact import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactRepository,
    MLMetadata,
    MLPredictionArtifactData,
    MLTrainingArtifactData,
)
from chapkit.data import DataFrame


async def main() -> None:
    """Run the hierarchical artifact example."""
    # Setup database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    from chapkit.artifact.models import Artifact

    async with engine.begin() as conn:
        await conn.run_sync(Artifact.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]

    hierarchy = ArtifactHierarchy(
        name="ml_pipeline",
        level_labels={0: "training", 1: "prediction"},
    )

    async with async_session() as session:
        manager = ArtifactManager(ArtifactRepository(session), hierarchy=hierarchy)

        print("Hierarchical Artifact Example")
        print("=" * 60)

        # Train model
        import numpy as np

        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_train = np.array([3.0, 5.0, 7.0, 9.0])

        model = LinearRegression()
        model.fit(X_train, y_train)

        print("\n1. Trained model")

        # Create training artifact
        training_data_model = MLTrainingArtifactData(
            type="ml_training",
            metadata=MLMetadata(
                status="success",
                config_id="01CONFIG123...",
                started_at=datetime(2025, 10, 18, 10, 0, 0).isoformat(),
                completed_at=datetime(2025, 10, 18, 10, 5, 0).isoformat(),
                duration_seconds=300.0,
            ),
            content=model,
            content_type="application/x-pickle",
        )

        training_artifact = await manager.save(
            ArtifactIn(
                data={
                    "type": training_data_model.type,
                    "metadata": training_data_model.metadata.model_dump(),
                    "content": model,
                    "content_type": training_data_model.content_type,
                    "content_size": training_data_model.content_size,
                }
            )
        )

        print(f"   Created training artifact: {training_artifact.id} (level {training_artifact.level})")

        # Make multiple predictions (children)
        prediction_ids = []

        for i, X_test in enumerate([[[5.0, 6.0]], [[6.0, 7.0]], [[7.0, 8.0]]]):
            predictions = model.predict(X_test)

            pred_df = DataFrame(
                columns=["feature1", "feature2", "prediction"],
                data=[[X_test[0][0], X_test[0][1], float(predictions[0])]],
            )

            prediction_data_model = MLPredictionArtifactData(
                type="ml_prediction",
                metadata=MLMetadata(
                    status="success",
                    config_id="01CONFIG123...",
                    started_at=datetime(2025, 10, 18, 10, 10, i).isoformat(),
                    completed_at=datetime(2025, 10, 18, 10, 10, i + 1).isoformat(),
                    duration_seconds=1.0,
                ),
                content=pred_df.model_dump(),
                content_type="application/x-pandas-dataframe",
            )

            prediction_artifact = await manager.save(
                ArtifactIn(
                    parent_id=training_artifact.id,
                    data=prediction_data_model.model_dump(),
                )
            )

            prediction_ids.append(prediction_artifact.id)
            print(f"   Created prediction {i + 1}: {prediction_artifact.id} (level {prediction_artifact.level})")

        # Build artifact tree
        print(f"\n2. Building artifact tree from root {training_artifact.id}")

        tree = await manager.build_tree(training_artifact.id)

        if tree:
            print(f"   Root: {tree.id} ({tree.data['type']}) - level {tree.level}")
            print(f"   Children: {len(tree.children or [])}")

            if tree.children:
                for i, child in enumerate(tree.children):
                    child_type = child.data["type"]
                    pred_count = len(child.data["content"]["data"]) if "content" in child.data else 0
                    print(f"     {i + 1}. {child.id} ({child_type}) - level {child.level}, {pred_count} predictions")

        # List all artifacts
        print("\n3. All artifacts in hierarchy:")

        all_artifacts = await manager.find_all()
        for artifact in all_artifacts:
            artifact_type = artifact.data["type"]
            print(f"   Level {artifact.level}: {artifact.id} ({artifact_type})")
            if artifact.parent_id:
                print(f"              parent: {artifact.parent_id}")

        print("\n" + "=" * 60)
        print("Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
