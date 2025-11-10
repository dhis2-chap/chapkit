"""Generic artifacts with flexible metadata.

This example demonstrates:
- Creating generic artifacts with custom metadata
- Using GenericMetadata for flexible fields
- Storing arbitrary content
- Mixing different artifact types
"""

import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from chapkit.artifact import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactRepository,
    GenericArtifactData,
    GenericMetadata,
)


async def main() -> None:
    """Run the generic artifact example."""
    # Setup database
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    from chapkit.artifact.models import Artifact

    async with engine.begin() as conn:
        await conn.run_sync(Artifact.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    hierarchy = ArtifactHierarchy(
        name="project",
        level_labels={0: "project", 1: "document", 2: "version"},
    )

    async with async_session() as session:
        manager = ArtifactManager(ArtifactRepository(session), hierarchy=hierarchy)

        print("Generic Artifact Example")
        print("=" * 60)

        # Example 1: Project artifact
        print("\n1. Creating project artifact")

        project_data = GenericArtifactData(
            type="generic",
            metadata=GenericMetadata(
                project_name="Documentation",  # type: ignore[call-arg]
                owner="Engineering Team",  # type: ignore[call-arg]
                created_date=datetime.now().isoformat(),  # type: ignore[call-arg]
                tags=["docs", "api", "public"],  # type: ignore[call-arg]
            ),
            content={
                "description": "API documentation project",
                "repository": "https://github.com/org/api-docs",
                "status": "active",
            },
            content_type="application/json",
        )

        project = await manager.save(ArtifactIn(data=project_data.model_dump()))

        print(f"   Created project: {project.id}")
        print(f"   Project name: {project.data['metadata']['project_name']}")
        print(f"   Owner: {project.data['metadata']['owner']}")
        print(f"   Tags: {project.data['metadata']['tags']}")

        # Example 2: Document artifact (child of project)
        print("\n2. Creating document artifact")

        document_data = GenericArtifactData(
            type="generic",
            metadata=GenericMetadata(
                document_title="Authentication Guide",  # type: ignore[call-arg]
                author="Alice",  # type: ignore[call-arg]
                category="security",  # type: ignore[call-arg]
            ),
            content={
                "sections": 5,
                "word_count": 1200,
                "last_reviewed": "2025-10-15",
            },
            content_type="application/json",
        )

        document = await manager.save(ArtifactIn(parent_id=project.id, data=document_data.model_dump()))

        print(f"   Created document: {document.id}")
        print(f"   Title: {document.data['metadata']['document_title']}")
        print(f"   Author: {document.data['metadata']['author']}")
        print(f"   Parent: {document.parent_id}")

        # Example 3: Multiple versions (children of document)
        print("\n3. Creating version artifacts")

        versions = [
            {
                "version": "1.0.0",
                "changes": "Initial release",
                "published": "2025-10-01",
            },
            {
                "version": "1.1.0",
                "changes": "Added OAuth examples",
                "published": "2025-10-10",
            },
            {
                "version": "1.2.0",
                "changes": "Updated error handling section",
                "published": "2025-10-15",
            },
        ]

        for ver in versions:
            version_data = GenericArtifactData(
                type="generic",
                metadata=GenericMetadata(
                    version_number=ver["version"],  # type: ignore[call-arg]
                    release_date=ver["published"],  # type: ignore[call-arg]
                ),
                content={
                    "changes": ver["changes"],
                    "download_url": f"https://docs.example.com/v{ver['version']}.pdf",
                },
                content_type="application/json",
            )

            version_artifact = await manager.save(ArtifactIn(parent_id=document.id, data=version_data.model_dump()))

            print(f"   Version {ver['version']}: {version_artifact.id}")

        # View the hierarchy
        print("\n4. Viewing artifact hierarchy")

        tree = await manager.build_tree(project.id)

        if tree:
            print(f"   {tree.data['metadata']['project_name']} (level {tree.level})")

            if tree.children:
                for doc in tree.children:
                    doc_title = doc.data["metadata"]["document_title"]
                    print(f"     └── {doc_title} (level {doc.level})")

                    if doc.children:
                        for ver in doc.children:
                            ver_num = ver.data["metadata"]["version_number"]
                            changes = ver.data["content"]["changes"]
                            print(f"         └── v{ver_num}: {changes} (level {ver.level})")

        # Query by level
        print("\n5. Querying artifacts by level")

        all_artifacts = await manager.find_all()

        for level in [0, 1, 2]:
            level_artifacts = [a for a in all_artifacts if a.level == level]
            print(f"   Level {level} ({hierarchy.level_labels[level]}): {len(level_artifacts)} artifacts")

        print("\n" + "=" * 60)
        print("Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
