# Artifact Manager Examples

These examples demonstrate how to use `ArtifactManager` directly with typed artifact schemas for programmatic artifact operations.

## Examples

### 01_basic_artifacts.py

**Basic artifact creation and querying with typed schemas**

Shows how to:
- Create ML training artifacts with `MLTrainingWorkspaceArtifactData`
- Store trained models with structured metadata
- Query artifacts by ID
- Access and use stored models

```bash
python examples/artifact_manager/01_basic_artifacts.py
```

**Key Concepts:**
- Typed artifact schemas with Pydantic validation
- Preserving Python objects during storage
- Accessing metadata and content separately
- Using retrieved models for predictions

### 02_hierarchical_artifacts.py

**Hierarchical artifacts with parent-child relationships**

Shows how to:
- Create parent training artifacts (level 0)
- Create multiple child prediction artifacts (level 1)
- Build artifact trees
- Navigate hierarchies

```bash
python examples/artifact_manager/02_hierarchical_artifacts.py
```

**Key Concepts:**
- Parent-child relationships with `parent_id`
- Artifact levels and hierarchy labels
- Tree building with `build_tree()`
- Linking predictions to training runs

### 03_generic_artifacts.py

**Generic artifacts with flexible metadata**

Shows how to:
- Create generic artifacts with custom metadata
- Use `GenericMetadata` for flexible fields
- Store arbitrary JSON content
- Build multi-level hierarchies (project → document → version)

```bash
python examples/artifact_manager/03_generic_artifacts.py
```

**Key Concepts:**
- Flexible metadata with extra fields
- Generic content storage
- Document versioning pattern
- Querying by hierarchy level

## Common Patterns

### Creating Typed Artifacts

```python
from chapkit.artifact import MLTrainingWorkspaceArtifactData, MLMetadata

# Create typed data model
training_data_model = MLTrainingWorkspaceArtifactData(
    type="ml_training_workspace",
    metadata=MLMetadata(
        status="success",
        config_id="01CONFIG...",
        started_at="2025-10-18T10:00:00Z",
        completed_at="2025-10-18T10:05:00Z",
        duration_seconds=300.0
    ),
    content=trained_model,  # Python object
    content_type="application/x-pickle"
)

# Manually construct dict to preserve Python objects
artifact_data = {
    "type": training_data_model.type,
    "metadata": training_data_model.metadata.model_dump(),
    "content": trained_model,  # Keep as Python object
    "content_type": training_data_model.content_type,
    "content_size": training_data_model.content_size,
}

# Save artifact
artifact = await manager.save(ArtifactIn(data=artifact_data))
```

### Querying Artifacts

```python
# Get by ID
artifact = await manager.find_by_id(artifact_id)

# Get all artifacts
all_artifacts = await manager.find_all()

# Build tree from root
tree = await manager.build_tree(root_id)

# Access typed data
metadata = artifact.data["metadata"]
content = artifact.data["content"]
```

### Working with Hierarchies

```python
# Define hierarchy with level labels
hierarchy = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={
        0: "training",
        1: "prediction",
        2: "evaluation"
    }
)

# Create parent (level 0)
parent = await manager.save(ArtifactIn(data=parent_data))

# Create child (level 1)
child = await manager.save(ArtifactIn(
    parent_id=parent.id,
    data=child_data
))
```

## Use Cases

These patterns are useful for:

- **Batch ML Pipelines**: Store training runs and predictions
- **Jupyter Notebooks**: Interactive artifact exploration
- **CLI Tools**: Command-line artifact management
- **Testing**: Unit tests for artifact workflows
- **Data Migration**: Bulk import/export of artifacts
- **ETL Jobs**: Extract-transform-load with artifact storage
- **Experiment Tracking**: Track ML experiments and results
- **Version Control**: Document and code versioning

## Database Setup

All examples use in-memory SQLite for simplicity:

```python
engine = create_async_engine("sqlite+aiosqlite:///:memory:")

from chapkit.artifact.models import Artifact

async with engine.begin() as conn:
    await conn.run_sync(Artifact.metadata.create_all)
```

For production, use persistent storage:

```python
engine = create_async_engine("sqlite+aiosqlite:///./artifacts.db")
# or
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
```

## Typed Schemas

Chapkit provides three main artifact data schemas:

1. **MLTrainingWorkspaceArtifactData**: For trained ML models
   - Required metadata: status, config_id, timestamps, duration
   - Content: Trained model (Python object)
   - Content-Type: `application/x-pickle`

2. **MLPredictionArtifactData**: For prediction results
   - Required metadata: status, config_id, timestamps, duration
   - Content: Predictions DataFrame
   - Content-Type: `application/vnd.chapkit.dataframe+json`

3. **GenericArtifactData**: For custom artifacts
   - Flexible metadata: Any extra fields allowed
   - Content: Any JSON-serializable data
   - Content-Type: Configurable

## Important Notes

### Preserving Python Objects

When storing Python objects (like trained models), construct the artifact data dict manually instead of using `model_dump()`:

```python
# ✅ Correct - preserves Python object
artifact_data = {
    "type": data_model.type,
    "metadata": data_model.metadata.model_dump(),
    "content": python_object,  # Keep as-is
    ...
}

# ❌ Incorrect - serializes to dict
artifact_data = data_model.model_dump()  # Loses Python object
```

### Accessing Nested Data

Use nested dictionary access for metadata fields:

```python
# Access metadata
config_id = artifact.data["metadata"]["config_id"]
duration = artifact.data["metadata"]["duration_seconds"]

# Access content
model = artifact.data["content"]
predictions = artifact.data["content"]
```

## Related Examples

- `examples/artifact/` - FastAPI service with artifact API
- `examples/ml_functional/` - Full ML service with artifacts
- `examples/ml_class/` - Class-based ML workflows

## Documentation

See the documentation for more details:
- [Artifact Storage Guide](../../docs/guides/artifact-storage.md)
- [ML Workflows Guide](../../docs/guides/ml-workflows.md)
