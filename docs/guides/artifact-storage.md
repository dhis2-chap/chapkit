# Artifact Storage

Servicekit provides a hierarchical artifact storage system for managing ML models, datasets, experimental results, document versions, and any other structured data that needs parent-child relationships and tree-based organization.

## Quick Start

```python
from chapkit.artifact import ArtifactHierarchy, ArtifactManager, ArtifactIn, ArtifactRepository
from servicekit.api import BaseServiceBuilder, ServiceInfo
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

# Define artifact hierarchy
hierarchy = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={0: "experiment", 1: "model", 2: "predictions"}
)

# Create dependency for artifact manager
async def get_artifact_manager(session: AsyncSession = Depends(get_session)) -> ArtifactManager:
    return ArtifactManager(ArtifactRepository(session), hierarchy=hierarchy)

# Build service
app = (
    BaseServiceBuilder(info=ServiceInfo(display_name="My Service"))
    .with_health()
    .with_database("sqlite+aiosqlite:///./data.db")
    .build()
)

# Add artifact router
from chapkit.artifact import ArtifactRouter, ArtifactOut

artifact_router = ArtifactRouter.create(
    prefix="/api/v1/artifacts",
    tags=["Artifacts"],
    manager_factory=get_artifact_manager,
    entity_in_type=ArtifactIn,
    entity_out_type=ArtifactOut,
)
app.include_router(artifact_router)
```

**Run:** `fastapi dev your_file.py`

---

## Architecture

### Hierarchical Storage

Artifacts are organized in parent-child trees with configurable level labels:

```
Experiment (level 0, parent_id=None)
  ├─> Model V1 (level 1, parent_id=experiment_id)
  │    ├─> Predictions A (level 2, parent_id=model_v1_id)
  │    └─> Predictions B (level 2, parent_id=model_v1_id)
  └─> Model V2 (level 1, parent_id=experiment_id)
       └─> Predictions C (level 2, parent_id=model_v2_id)
```

**Benefits:**
- Complete lineage tracking
- Immutable versioning
- Multiple children per parent
- Flexible hierarchy depths

### Level Labels

Define semantic meaning for each hierarchy level:

```python
ArtifactHierarchy(
    name="document_versions",
    level_labels={
        0: "project",
        1: "document",
        2: "version"
    }
)
```

---

## Core Concepts

### ArtifactHierarchy

Defines the structure and validation rules for artifact trees.

```python
from chapkit.artifact import ArtifactHierarchy

hierarchy = ArtifactHierarchy(
    name="ml_pipeline",           # Unique identifier
    level_labels={                # Semantic labels
        0: "experiment",
        1: "trained_model",
        2: "predictions"
    }
)
```

**Properties:**
- `name`: Unique identifier for the hierarchy
- `level_labels`: Dict mapping level numbers to semantic labels
- Validates parent-child relationships
- Enforces level constraints

### Artifact

Base entity for hierarchical storage.

```python
from chapkit.artifact import ArtifactIn

artifact = ArtifactIn(
    parent_id=parent_artifact_id,  # Optional: link to parent
    data={"key": "value"},         # Any JSON-serializable data
)
```

**Fields:**
- `id`: ULID (auto-generated)
- `parent_id`: Optional parent artifact ID
- `level`: Hierarchy level (computed from parent)
- `data`: JSON-serializable dictionary
- `created_at`, `updated_at`: Timestamps

### ArtifactManager

Business logic layer for artifact operations.

```python
from chapkit.artifact import ArtifactManager, ArtifactRepository

manager = ArtifactManager(repository, hierarchy=hierarchy)

# Create root artifact
root = await manager.save(ArtifactIn(data={"experiment": "v1"}))

# Create child artifact
child = await manager.save(ArtifactIn(
    parent_id=root.id,
    data={"model": "trained"}
))

# Query tree
tree = await manager.build_tree(root.id)
```

---

## API Endpoints

### POST /api/v1/artifacts

Create new artifact.

**Request:**
```json
{
  "parent_id": "01PARENT123...",
  "data": {
    "experiment": "weather_prediction",
    "version": "1.0.0"
  }
}
```

**Response (201 Created):**
```json
{
  "id": "01ARTIFACT456...",
  "parent_id": "01PARENT123...",
  "level": 1,
  "data": {
    "experiment": "weather_prediction",
    "version": "1.0.0"
  },
  "created_at": "2025-10-18T12:00:00Z",
  "updated_at": "2025-10-18T12:00:00Z"
}
```

### GET /api/v1/artifacts

List all artifacts with pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Page size (default: 50)

**Response:**
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "size": 50,
  "pages": 2
}
```

### GET /api/v1/artifacts/{id}

Get artifact by ID.

### PUT /api/v1/artifacts/{id}

Update artifact data.

**Request:**
```json
{
  "data": {
    "updated": "value"
  }
}
```

### DELETE /api/v1/artifacts/{id}

Delete artifact (fails if artifact has children).

### POST /api/v1/artifacts/$tree

Get artifact tree structure.

**Request:**
```json
{
  "root_id": "01ARTIFACT456...",
  "max_depth": 3
}
```

**Response:**
```json
{
  "id": "01ARTIFACT456...",
  "level": 0,
  "data": {...},
  "children": [
    {
      "id": "01CHILD789...",
      "level": 1,
      "data": {...},
      "children": [...]
    }
  ]
}
```

### POST /api/v1/artifacts/$expand

Get artifact with all ancestors and descendants.

**Request:**
```json
{
  "artifact_id": "01ARTIFACT456..."
}
```

---

## Data Storage Patterns

### Storing DataFrames

Use `PandasDataFrame` schema for tabular data:

```python
from chapkit.artifact import ArtifactIn, PandasDataFrame
import pandas as pd

df = pd.DataFrame({
    "feature1": [1.0, 2.0, 3.0],
    "feature2": [4.0, 5.0, 6.0]
})

pandas_df = PandasDataFrame.from_dataframe(df)

artifact = await manager.save(ArtifactIn(
    data={"dataframe": pandas_df.model_dump()}
))

# Retrieve and convert back
artifact = await manager.find_by_id(artifact_id)
df = PandasDataFrame(**artifact.data["dataframe"]).to_dataframe()
```

### Storing Binary Data

Store large binary data externally and reference in artifact:

```python
# Save binary to external storage
blob_url = await save_to_s3(binary_data)

# Store reference in artifact
artifact = await manager.save(ArtifactIn(
    data={
        "blob_url": blob_url,
        "blob_size": len(binary_data),
        "blob_type": "model/pytorch"
    }
))
```

### Storing Model Objects

Use pickle for Python objects:

```python
import pickle
import base64

# Serialize model
model_bytes = pickle.dumps(model)
model_b64 = base64.b64encode(model_bytes).decode('utf-8')

artifact = await manager.save(ArtifactIn(
    data={
        "model": model_b64,
        "model_type": type(model).__name__,
        "model_size": len(model_bytes)
    }
))

# Deserialize
model_bytes = base64.b64decode(artifact.data["model"])
model = pickle.loads(model_bytes)
```

---

## Use Cases

### ML Model Lineage

```python
hierarchy = ArtifactHierarchy(
    name="ml_lineage",
    level_labels={
        0: "experiment",
        1: "trained_model",
        2: "predictions"
    }
)

# Track experiment
experiment = await manager.save(ArtifactIn(
    data={
        "experiment_name": "weather_forecast",
        "started_at": "2025-10-18T10:00:00Z"
    }
))

# Track trained model
model = await manager.save(ArtifactIn(
    parent_id=experiment.id,
    data={
        "model_type": "LinearRegression",
        "hyperparameters": {"alpha": 0.01},
        "metrics": {"rmse": 0.25}
    }
))

# Track predictions
predictions = await manager.save(ArtifactIn(
    parent_id=model.id,
    data={
        "prediction_date": "2025-10-20",
        "num_predictions": 1000
    }
))
```

### Document Versioning

```python
hierarchy = ArtifactHierarchy(
    name="documents",
    level_labels={
        0: "project",
        1: "document",
        2: "version"
    }
)

# Project
project = await manager.save(ArtifactIn(
    data={"name": "API Documentation"}
))

# Document
doc = await manager.save(ArtifactIn(
    parent_id=project.id,
    data={"title": "Authentication Guide"}
))

# Versions
v1 = await manager.save(ArtifactIn(
    parent_id=doc.id,
    data={"version": "1.0", "content": "..."}
))

v2 = await manager.save(ArtifactIn(
    parent_id=doc.id,
    data={"version": "1.1", "content": "..."}
))
```

### Dataset Versioning

```python
hierarchy = ArtifactHierarchy(
    name="datasets",
    level_labels={
        0: "dataset",
        1: "version",
        2: "partition"
    }
)

# Dataset
dataset = await manager.save(ArtifactIn(
    data={"name": "customer_data"}
))

# Version
version = await manager.save(ArtifactIn(
    parent_id=dataset.id,
    data={"version": "2025-10", "rows": 1000000}
))

# Partitions
train = await manager.save(ArtifactIn(
    parent_id=version.id,
    data={"partition": "train", "rows": 800000}
))

test = await manager.save(ArtifactIn(
    parent_id=version.id,
    data={"partition": "test", "rows": 200000}
))
```

---

## Testing

### Unit Tests

```python
import pytest
from chapkit.artifact import ArtifactManager, ArtifactRepository, ArtifactIn

@pytest.mark.asyncio
async def test_artifact_hierarchy(session):
    """Test parent-child relationships."""
    repo = ArtifactRepository(session)
    manager = ArtifactManager(repo, hierarchy=test_hierarchy)

    # Create parent
    parent = await manager.save(ArtifactIn(data={"level": "root"}))
    assert parent.level == 0
    assert parent.parent_id is None

    # Create child
    child = await manager.save(ArtifactIn(
        parent_id=parent.id,
        data={"level": "child"}
    ))
    assert child.level == 1
    assert child.parent_id == parent.id

    # Build tree
    tree = await manager.build_tree(parent.id)
    assert len(tree.children) == 1
    assert tree.children[0].id == child.id
```

### cURL Testing

```bash
# Create root artifact
ROOT_ID=$(curl -s -X POST http://localhost:8000/api/v1/artifacts \
  -H "Content-Type: application/json" \
  -d '{"data": {"experiment": "test"}}' | jq -r '.id')

# Create child artifact
CHILD_ID=$(curl -s -X POST http://localhost:8000/api/v1/artifacts \
  -H "Content-Type: application/json" \
  -d '{
    "parent_id": "'$ROOT_ID'",
    "data": {"model": "trained"}
  }' | jq -r '.id')

# Get tree
curl -X POST http://localhost:8000/api/v1/artifacts/\$tree \
  -H "Content-Type: application/json" \
  -d '{"root_id": "'$ROOT_ID'"}' | jq
```

---

## Production Considerations

### Database Size Management

Artifacts are stored in SQLite by default. Monitor database growth:

```bash
# Check database size
du -h data.db

# Count artifacts
sqlite3 data.db "SELECT COUNT(*) FROM artifacts;"
```

**Best Practices:**
- Implement retention policies
- Archive old artifacts externally
- Monitor BLOB storage growth
- Use PostgreSQL for large deployments

### Indexing

Default indexes on `id`, `parent_id`, `level`, `created_at`. Add custom indexes for queries:

```python
from sqlalchemy import Index

# Add index on data->>'experiment_name' for fast lookups
Index('ix_artifact_experiment', Artifact.data['experiment_name'].astext)
```

### Backup Strategy

```bash
# SQLite backup
sqlite3 data.db ".backup backup_$(date +%Y%m%d).db"

# Automated backups
0 2 * * * sqlite3 /app/data.db ".backup /backups/data_$(date +\%Y\%m\%d).db"
```

---

## Complete Example

See `examples/artifact_storage_api.py` for a complete working example with document versioning system.

## Next Steps

- **Task Execution:** Combine artifacts with tasks for ML pipelines
- **Job Scheduler:** Use background jobs for expensive artifact operations
- **Monitoring:** Track artifact creation rates with Prometheus
