"""Basic artifact creation and querying with typed schemas.

This example demonstrates:
- Creating ML training artifacts with typed schemas
- Storing structured metadata
- Querying artifacts
- Accessing typed data
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
    MLTrainingWorkspaceArtifactData,
)


async def main() -> None:
    """Run the basic artifact example."""
    # Setup database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    from chapkit.artifact.models import Artifact

    async with engine.begin() as conn:
        await conn.run_sync(Artifact.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]

    # Define hierarchy
    hierarchy = ArtifactHierarchy(
        name="ml_pipeline",
        level_labels={0: "training", 1: "prediction"},
    )

    async with async_session() as session:
        manager = ArtifactManager(ArtifactRepository(session), hierarchy=hierarchy)

        print("Basic Artifact Creation Example")
        print("=" * 60)

        # Train a simple model
        import numpy as np

        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_train = np.array([3.0, 5.0, 7.0])

        model = LinearRegression()
        model.fit(X_train, y_train)

        print(f"\n1. Trained model: y = {model.coef_[0]:.2f}*x1 + {model.coef_[1]:.2f}*x2 + {model.intercept_:.2f}")

        # Create typed training artifact data WITHOUT using model_dump on the model
        started_at = datetime(2025, 10, 18, 10, 0, 0)
        completed_at = datetime(2025, 10, 18, 10, 5, 30)

        training_data_model = MLTrainingWorkspaceArtifactData(
            type="ml_training_workspace",
            metadata=MLMetadata(
                status="success",
                config_id="01CONFIG123...",
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                duration_seconds=(completed_at - started_at).total_seconds(),
            ),
            content=model,  # Validate structure but don't serialize model yet
            content_type="application/x-pickle",
        )

        # Manually construct dict to preserve Python objects
        artifact_data = {
            "type": training_data_model.type,
            "metadata": training_data_model.metadata.model_dump(),
            "content": model,  # Keep as Python object for PickleType storage
            "content_type": training_data_model.content_type,
            "content_size": training_data_model.content_size,
        }

        # Save artifact
        training_artifact = await manager.save(ArtifactIn(data=artifact_data))

        print(f"\n2. Created training artifact: {training_artifact.id}")
        print(f"   Type: {training_artifact.data['type']}")
        print(f"   Status: {training_artifact.data['metadata']['status']}")
        print(f"   Duration: {training_artifact.data['metadata']['duration_seconds']}s")
        print(f"   Model type: {type(training_artifact.data['content']).__name__}")

        # Query the artifact back
        retrieved = await manager.find_by_id(training_artifact.id)

        if retrieved:
            print(f"\n3. Retrieved artifact: {retrieved.id}")

            # Access typed data
            assert retrieved.data["type"] == "ml_training_workspace"
            metadata = retrieved.data["metadata"]
            trained_model = retrieved.data["content"]

            print(f"   Config ID: {metadata['config_id']}")
            print(f"   Duration: {metadata['duration_seconds']}s")
            print(f"   Model coefficients: {trained_model.coef_}")
            print(f"   Model intercept: {trained_model.intercept_:.2f}")

            # Use the retrieved model
            X_new = np.array([[5.0, 6.0]])
            prediction = trained_model.predict(X_new)
            print(f"   Prediction for [5.0, 6.0]: {prediction[0]:.2f}")

        print("\n" + "=" * 60)
        print("Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
