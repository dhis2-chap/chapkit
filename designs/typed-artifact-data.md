# Design: Typed Artifact Data with Download Support

## Status
**Proposed** - November 2025

## Problem Statement

The current artifact system has several limitations that make it difficult to work with artifacts containing both JSON metadata and binary content:

### Current Architecture
- Artifacts store data in a single `data` column using SQLAlchemy's `PickleType`
- The `data` field accepts `Any` type, allowing arbitrary Python objects
- When retrieved via HTTP API, non-JSON-serializable objects are converted to metadata placeholders
- No way for external clients to download the actual binary content (trained models, datasets, etc.)

### Specific Issues

1. **No Binary Downloads**: ML models and other binary artifacts cannot be downloaded by external clients
2. **Untyped Data**: No validation or type safety for artifact data structures
3. **Metadata Loss**: Important metadata is hidden inside pickled objects, not queryable
4. **Poor API Experience**: Clients receive placeholder metadata instead of actual data
5. **No Content Separation**: Metadata and binary content are mixed together

### Example Problem

When storing a trained ML model:

```python
# What you store (internally)
artifact = Artifact(data=trained_sklearn_model)

# What API clients get back
{
  "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "data": {
    "_type": "LinearRegression",
    "_serialization_error": "Object is not JSON-serializable",
    "_repr": "LinearRegression(fit_intercept=True)"
  }
}
# ❌ Cannot download the actual model
```

## Proposed Solution

Introduce a typed artifact data system using Pydantic discriminated unions with a `type` field, standardize the structure for metadata and content, and add download endpoints.

### Key Design Principles

1. **Type Safety**: Use Pydantic discriminated unions for validation
2. **Backward Compatible**: No database migrations, support legacy data
3. **Simple Storage**: Keep using PickleType, add structure at application layer
4. **Clear Separation**: Distinguish metadata (JSON) from content (binary)
5. **Download Support**: Add endpoints for binary content retrieval

## Technical Design

### 1. Typed Artifact Data Schemas with Metadata/Content Separation

The key insight: **separate JSON-serializable metadata from binary content at storage time**, using **strongly-typed metadata schemas** with Python 3.13+ generics.

```python
# src/chapkit/artifact/data_schemas.py (NEW)

from typing import Annotated, Literal
from pydantic import BaseModel, Field

# Generic base class using Python 3.13+ syntax
class BaseArtifactData[MetadataT: BaseModel](BaseModel):
    """Base class for all artifact data types with typed metadata."""
    type: str = Field(description="Discriminator field for artifact type")
    metadata: MetadataT = Field(description="Strongly-typed JSON-serializable metadata")
    content: JsonSafe = Field(description="Content as Python object (bytes, DataFrame, models, etc.)")
    content_type: str | None = Field(default=None, description="MIME type for download responses")
    content_size: int | None = Field(default=None, description="Size of content in bytes")

    model_config = {"extra": "forbid"}

# Metadata schemas
class MLMetadata(BaseModel):
    """Metadata for ML artifacts (training and prediction)."""
    status: Literal["success", "failed"]
    config_id: str
    started_at: str  # ISO 8601 format
    completed_at: str
    duration_seconds: float

class GenericMetadata(BaseModel):
    """Free-form metadata for generic artifacts."""
    model_config = {"extra": "allow"}

# Concrete artifact types
class MLTrainingArtifactData(BaseArtifactData[MLMetadata]):
    """Schema for ML training artifact data with trained model.

    Content varies based on status:
    - success: Model object (sklearn, pytorch, etc.) or ZIP of training outputs
      - content_type: "application/zip" (most common)
    - failed: ZIP of entire temp directory for debugging
      - content_type: "application/zip"

    Note: Content is stored as Python object, PickleType handles DB serialization.
    """
    type: Literal["ml_training"] = "ml_training"
    metadata: MLMetadata

class MLPredictionArtifactData(BaseArtifactData[MLMetadata]):
    """Schema for ML prediction artifact data with results.

    Content varies based on status:
    - success: DataFrame with predictions or ZIP of prediction outputs
      - content_type: "text/csv", "application/x-parquet", "application/zip"
    - failed: ZIP of temp directory for debugging
      - content_type: "application/zip"

    Note: Content is stored as Python object, PickleType handles DB serialization.
    """
    type: Literal["ml_prediction"] = "ml_prediction"
    metadata: MLMetadata

class GenericArtifactData(BaseArtifactData[GenericMetadata]):
    """Schema for generic artifact data with free-form metadata."""
    type: Literal["generic"] = "generic"
    metadata: GenericMetadata

    # content: Any binary data
    # content_type: Any MIME type

# Discriminated union type
ArtifactData = Annotated[
    MLTrainingArtifactData | MLPredictionArtifactData | GenericArtifactData,
    Field(discriminator="type")
]
```

**Key Features:**
1. **Python 3.13+ generics**: `class BaseArtifactData[MetadataT: BaseModel]` for clean type parameterization
2. **Strongly-typed metadata**: Separate `Metadata` models with full validation
3. **Clear separation**: `metadata` (typed JSON) vs `content` (Python objects)
4. **Flexible content**: Store any Python object (bytes, DataFrame, models, dicts, etc.)
5. **Automatic serialization**: PickleType handles DB storage, JsonSafe handles API responses
6. **Type safety**: IDE autocomplete for `artifact.metadata.config_id`
7. **Status tracking**: Both successful and failed jobs create artifacts with `status` field
8. **Simple downloads**: Most downloads (99%) are ZIP files, just return the bytes

### 2. Artifact Status Pattern

Artifacts are created **after** job completion, with `status` indicating outcome:

**Success:**
- `metadata.status = "success"`
- `content` = serialized model/predictions in original format
- `content_type` = format-specific MIME type
- All required metadata fields populated

**Failed:**
- `metadata.status = "failed"`
- `content` = ZIP of entire temp directory for debugging/reproducibility
- `content_type = "application/zip"`
- Model metadata fields = None

**Workflow Validation (Future):**
```python
# Before running predict, validate training succeeded
training_artifact = await artifact_manager.find_by_id(training_artifact_id)
if training_artifact.metadata.status != "success":
    raise ValueError("Cannot predict with failed training artifact")

# Unpack ZIP and run prediction...
```

**Benefits:**
- Full reproducibility even for failures
- Debugging support (inspect logs, intermediate files)
- Audit trail of all job executions
- Workflow validation based on persistent state

### 3. Type Field Structure

The design uses a single `type` field at the root level:

- **`type`**: Discriminator for Pydantic unions (`"ml_training"`, `"ml_prediction"`, `"generic"`)

This allows:
- Pydantic to route to correct schema based on `type`
- ML logic to determine artifact type from `artifact.data["type"]`
- Simple, non-redundant design

### 4. Download Endpoints

Add new operations to the artifact router:

```python
# src/chapkit/artifact/router.py

from chapkit.data import DataFrame

@router.get("/{artifact_id}/$download")
async def download_artifact(artifact_id: str) -> Response:
    """Download artifact content as binary file.

    Most commonly used for ZIP files (99% of downloads).
    """
    artifact = await manager.find_by_id(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404)

    if not isinstance(artifact.data, dict):
        raise HTTPException(status_code=400, detail="Artifact has no downloadable content")

    content = artifact.data.get("content")
    if content is None:
        raise HTTPException(status_code=404, detail="Artifact has no content")

    content_type = artifact.data.get("content_type", "application/octet-stream")

    # Serialize content to bytes based on type
    if isinstance(content, bytes):
        # Most common case: ZIP files, PNG images, etc.
        binary = content
    elif isinstance(content, DataFrame):
        # Serialize DataFrame based on content_type
        if content_type == "text/csv":
            binary = content.to_csv().encode()
        elif content_type == "application/x-parquet":
            binary = content.to_parquet_bytes()
        else:
            binary = content.to_json().encode()
    else:
        raise ValueError(f"Cannot serialize content of type {type(content).__name__}")

    # Determine filename extension
    extension_map = {
        "application/zip": "zip",
        "text/csv": "csv",
        "application/x-parquet": "parquet",
        "application/json": "json",
        "image/png": "png",
    }
    ext = extension_map.get(content_type, "bin")

    return Response(
        content=binary,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=artifact_{artifact_id}.{ext}"
        }
    )

@router.get("/{artifact_id}/$metadata")
async def get_artifact_metadata(artifact_id: str) -> dict:
    """Get only JSON-serializable metadata, excluding binary content."""
    artifact = await manager.find_by_id(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404)

    if not isinstance(artifact.data, dict):
        return {}

    return artifact.data.get("metadata", {})
```

**Key Points:**
- 99% of downloads are ZIP files (bytes) - just return them
- DataFrame serialization using chapkit.data.DataFrame methods
- No complex serializer registry needed
- Raises error for unsupported content types (no pickle fallback)

### 5. Manager Methods

Add helper methods to ArtifactManager:

```python
# src/chapkit/artifact/manager.py

class ArtifactManager(BaseManager):

    def extract_metadata(self, artifact: Artifact) -> dict:
        """Extract JSON-serializable metadata from artifact data."""
        data = artifact.data

        if not isinstance(data, dict):
            return {}

        # Extract all JSON-serializable fields
        metadata = {}
        for key, value in data.items():
            try:
                json.dumps(value)  # Test if JSON-serializable
                metadata[key] = value
            except (TypeError, ValueError):
                # Skip non-JSON fields (like model objects)
                continue

        return metadata

    async def pre_save(self, entity: Artifact, data: ArtifactIn):
        """Validate artifact data if typed."""
        if isinstance(data.data, dict) and "type" in data.data:
            from chapkit.artifact.data_schemas import validate_artifact_data
            validate_artifact_data(data.data)

        await super().pre_save(entity, data)
```

### 6. Storage Format

No changes to database schema - continue using PickleType:

```python
# src/chapkit/artifact/models.py (NO CHANGE)

class Artifact(Entity):
    data: Mapped[Any] = mapped_column(PickleType(protocol=4), nullable=False)
    # ... rest unchanged
```

The typed structure is enforced at the application layer, pickled objects in the database remain transparent.

## Implementation Phases

### Phase 1: Core Schemas (Week 1)
- [ ] Create `src/chapkit/artifact/data_schemas.py`
- [ ] Implement `BaseArtifactData`, `MLTrainingArtifactData`, `MLPredictionArtifactData`, `GenericArtifactData`
- [ ] Add `validate_artifact_data()` helper
- [ ] Add tests for schema validation
- [ ] Add tests for discriminated union behavior

### Phase 2: ML Module Integration (Week 1)
- [ ] Update `ml/manager.py` to use `MLTrainingArtifactData`
- [ ] Update `ml/manager.py` to use `MLPredictionArtifactData`
- [ ] Update `ml/schemas.py` to re-export from `data_schemas`
- [ ] Update type checks to use `type` field
- [ ] Update ML tests

### Phase 3: Download Endpoints (Week 2)
- [ ] Add `GET /{id}/$download` endpoint to `artifact/router.py`
- [ ] Add `GET /{id}/$metadata` endpoint to `artifact/router.py`
- [ ] Add `extract_metadata()` method to `artifact/manager.py`
- [ ] Add tests for download endpoints
- [ ] Update Postman collection

### Phase 4: Examples and Documentation (Week 2)
- [ ] Update `examples/artifact/main.py` to demonstrate typed artifacts
- [ ] Add download examples to Postman collection
- [ ] Update docstrings throughout
- [ ] Add migration guide for users

### Phase 5: Optional Enhancements (Future)
- [ ] Add content-type detection for downloads
- [ ] Add compression support for large artifacts
- [ ] Add streaming support for very large artifacts
- [ ] Consider external blob storage for huge files

## API Examples

### Creating a Typed ML Training Artifact (Success with ZIP)

```python
import zipfile
from io import BytesIO
from datetime import UTC, datetime
from pathlib import Path
from chapkit.artifact.data_schemas import MLTrainingArtifactData, MLMetadata

# Most common case: ZIP entire training directory
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    # Add model file, training logs, plots, etc.
    for file_path in training_dir.rglob("*"):
        if file_path.is_file():
            zip_file.write(file_path, file_path.relative_to(training_dir))

zip_bytes = zip_buffer.getvalue()

# Create strongly-typed metadata
metadata = MLMetadata(
    config_id=str(config_id),
    started_at=started_at.isoformat(),
    completed_at=datetime.now(UTC).isoformat(),
    duration_seconds=42.5,
    status="success",
)

# Create typed artifact data
training_data = MLTrainingArtifactData(
    type="ml_training",
    metadata=metadata,  # Strongly-typed!
    content=zip_bytes,  # ZIP file as bytes
    content_type="application/zip",
    content_size=len(zip_bytes),
)

# Save artifact (PickleType handles DB serialization automatically)
artifact = await artifact_manager.save(
    ArtifactIn(data=training_data.model_dump())
)

# Type-safe access
print(f"Status: {training_data.metadata.status}")
print(f"Duration: {training_data.metadata.duration_seconds}s")  # IDE autocomplete!
print(f"Size: {training_data.content_size} bytes")
```

### Creating a Failed Training Artifact

```python
import zipfile
from io import BytesIO
from datetime import UTC, datetime
from chapkit.artifact.data_schemas import MLTrainingArtifactData, MLMetadata

# Zip entire temp directory for debugging
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    # Add all files from temp directory
    for file_path in temp_dir.rglob("*"):
        if file_path.is_file():
            zip_file.write(file_path, file_path.relative_to(temp_dir))

zip_bytes = zip_buffer.getvalue()

# Create metadata for failed training
metadata = MLMetadata(
    config_id=str(config_id),
    started_at=started_at.isoformat(),
    completed_at=datetime.now(UTC).isoformat(),
    duration_seconds=15.2,
    status="failed",
)

# Create artifact with zipped temp directory
training_data = MLTrainingArtifactData(
    type="ml_training",
    metadata=metadata,
    content=zip_bytes,  # ZIP of temp directory for debugging
    content_type="application/zip",
    content_size=len(zip_bytes),
)

# Save failed artifact for audit trail
artifact = await artifact_manager.save(
    ArtifactIn(data=training_data.model_dump())
)
```

### Retrieving Metadata vs Downloading Binary

```bash
# Get artifact with metadata (JSON response)
GET /api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAV
# Returns JSON with type, metadata dict, content_type
# (content is excluded or base64-encoded)

# Get only metadata (excludes binary content entirely)
GET /api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAV/$metadata
# Returns: {"config_id": "...", "status": "success", "started_at": "...", ...}

# Download binary content in ORIGINAL format
GET /api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAV/$download
# Returns: joblib bytes with Content-Type: application/x-joblib
# Filename: artifact_01ARZ3NDEKTSV4RRFFQ69G5FAV.joblib
```

### Using Downloaded Artifacts

```python
import zipfile
import requests
from io import BytesIO

# First, get artifact to check content_type
artifact_response = requests.get(
    "http://api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAV"
)
artifact = artifact_response.json()
content_type = artifact["data"]["content_type"]  # "application/zip"

# Download binary content
download_response = requests.get(
    "http://api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAV/$download"
)

# Most common case: Extract ZIP archive
if content_type == "application/zip":
    with zipfile.ZipFile(BytesIO(download_response.content)) as zip_file:
        # Extract model file
        model_bytes = zip_file.read("model.joblib")

        # Load the model
        import joblib
        model = joblib.loads(model_bytes)

        # Use the model
        predictions = model.predict(X_test)
```

### Creating a Generic Artifact with ZIP File

```python
import zipfile
from io import BytesIO
from chapkit.artifact.data_schemas import GenericArtifactData, GenericMetadata

# Create a ZIP file containing multiple files
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    zip_file.writestr("data.csv", csv_data)
    zip_file.writestr("metadata.json", json_metadata)
    zip_file.writestr("README.md", documentation)

zip_bytes = zip_buffer.getvalue()

# Create artifact with free-form metadata
artifact_data = GenericArtifactData(
    type="generic",
    metadata=GenericMetadata(
        description="Dataset package for rainfall prediction experiment",
        files=["data.csv", "metadata.json", "README.md"],
        created_by="data_pipeline_v2",
        dataset_version="2024-11-10",
    ),
    content=zip_bytes,
    content_type="application/zip",
    content_size=len(zip_bytes),
)

# Save artifact
artifact = await artifact_manager.save(
    ArtifactIn(
        name="rainfall-dataset-2024-11-10",
        data=artifact_data.model_dump(),
    )
)
```

### Downloading and Extracting ZIP Artifacts

```python
import zipfile
from io import BytesIO
import requests

# Get metadata to verify it's a ZIP
metadata_response = requests.get(
    "http://api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAZ/$metadata"
)
metadata = metadata_response.json()
print(f"Files in archive: {metadata['files']}")

# Download ZIP content
download_response = requests.get(
    "http://api/v1/artifacts/01ARZ3NDEKTSV4RRFFQ69G5FAZ/$download"
)

# Extract and use files
with zipfile.ZipFile(BytesIO(download_response.content)) as zip_file:
    # List contents
    print(zip_file.namelist())  # ['data.csv', 'metadata.json', 'README.md']

    # Extract specific file
    csv_data = zip_file.read("data.csv").decode()

    # Or extract all to directory
    zip_file.extractall("/tmp/dataset")
```

## Backward Compatibility

### Legacy Data Support

The design maintains backward compatibility:

1. **Untyped artifacts**: Generic artifacts accept any structure
2. **No migration**: Existing pickled data works without changes
3. **Gradual adoption**: Can mix typed and untyped artifacts

### Migration Strategy

```python
# Validation helper for artifact data
def validate_artifact_data(data: dict[str, Any]) -> BaseArtifactData:
    """Validate artifact data against appropriate schema."""

    # Determine artifact type
    artifact_type = data.get("type", "generic")

    # Validate with appropriate schema
    schema_map = {
        "ml_training": MLTrainingArtifactData,
        "ml_prediction": MLPredictionArtifactData,
        "generic": GenericArtifactData,
    }

    schema = schema_map.get(artifact_type, GenericArtifactData)
    return schema.model_validate(data)
```

### Breaking Changes

**None** - This design is fully backward compatible:
- No database schema changes
- No changes to existing API endpoints (only additions)
- Legacy artifacts continue to work (treated as generic)
- New artifacts use strongly-typed schemas

## Benefits

### For Developers
1. **Type Safety**: IDE autocomplete and type checking for artifact fields
2. **Validation**: Pydantic catches data errors at creation time
3. **Self-Documenting**: Schemas clearly define artifact structure
4. **Extensibility**: Easy to add new artifact types

### For API Users
1. **Download Support**: Can retrieve binary content (models, datasets)
2. **Better Metadata**: JSON responses show queryable metadata
3. **Clear Structure**: Discriminated types make artifact format explicit
4. **Flexibility**: Choose between metadata-only or full download

### For System
1. **No Migration**: No database changes required
2. **Backward Compatible**: Existing code works unchanged
3. **Gradual Adoption**: Can migrate incrementally
4. **Performance**: No storage overhead, same pickle format

## Alternatives Considered

### Alternative 1: Separate Metadata and Content Columns

Split `data` into two columns:
```python
metadata: Mapped[dict] = mapped_column(JSON)
content: Mapped[bytes | None] = mapped_column(LargeBinary)
```

**Rejected because:**
- Requires database migration
- Breaking change for existing data
- More complex to implement
- Overkill for current needs

### Alternative 2: Base64 Encoding in JSON Column

Store everything as JSON with base64-encoded binary:
```python
data: Mapped[dict] = mapped_column(JSON)
{
    "metadata": {...},
    "content": "base64_encoded_string"
}
```

**Rejected because:**
- 33% storage overhead from base64
- Still requires migration from PickleType
- Larger payloads over HTTP
- No clear benefit over pickle

### Alternative 3: External Blob Storage

Store binary content in S3/filesystem, metadata in database.

**Deferred because:**
- Significantly more complex
- Requires infrastructure setup
- Not needed for typical artifact sizes
- Can be added later if needed

## Open Questions

1. **Content-Type Detection**: Should we detect and return specific content types (e.g., `application/x-sklearn-model`)?
   - **Resolution**: Start with generic `application/octet-stream`, add specific types later if needed

2. **Compression**: Should we compress large artifacts?
   - **Resolution**: Not initially, can add transparent gzip compression in future

3. **Size Limits**: What's the maximum artifact size we support?
   - **Resolution**: Document 100MB soft limit, SQLite hard limit is 1GB

4. **Streaming**: Do we need streaming for very large downloads?
   - **Resolution**: Not initially, add if users request it

## Security Considerations

1. **Pickle Security**: Unpickling untrusted data can execute arbitrary code
   - **Mitigation**: Only system-created artifacts are pickled, not user uploads
   - **Future**: Consider safer serialization for user-uploaded content

2. **Download Access Control**: Who can download artifacts?
   - **Current**: Same permissions as GET endpoint (no additional restrictions)
   - **Future**: Could add download-specific permissions if needed

3. **Size-Based DoS**: Large downloads could overwhelm server
   - **Mitigation**: Document size limits, consider rate limiting in future

## Success Metrics

1. **Adoption**: ML module uses typed artifacts (100% of new code)
2. **Backward Compatibility**: All existing tests pass without modification
3. **Performance**: No measurable performance degradation (<5% overhead)
4. **API Usage**: Download endpoint used by external clients
5. **Type Safety**: Pydantic catches validation errors during development

## Timeline

- **Week 1**: Core schemas and ML module integration
- **Week 2**: Download endpoints and examples
- **Week 3**: Documentation and testing
- **Week 4**: Review, refinement, and merge

## References

- [Pydantic Discriminated Unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)
- [SQLAlchemy PickleType](https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.PickleType)
- [FastAPI Response Models](https://fastapi.tiangolo.com/tutorial/response-model/)
- [servicekit Repository Pattern](https://winterop-com.github.io/servicekit)

## Appendix: Complete Schema Examples

### ML Training Artifact

```json
{
  "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "name": "rf-classifier-training",
  "data": {
    "type": "ml_training",
    "metadata": {
      "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAX",
      "started_at": "2025-01-10T10:00:00Z",
      "completed_at": "2025-01-10T10:00:42Z",
      "duration_seconds": 42.5,
      "status": "success"
    },
    "content": "<bytes: ZIP file>",
    "content_type": "application/zip",
    "content_size": 1048576
  },
  "level": 0,
  "parent_id": null
}
```

### ML Prediction Artifact

```json
{
  "id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
  "name": "predictions-2025-01-10",
  "data": {
    "type": "ml_prediction",
    "metadata": {
      "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAX",
      "started_at": "2025-01-10T11:00:00Z",
      "completed_at": "2025-01-10T11:00:05Z",
      "duration_seconds": 5.2,
      "status": "success"
    },
    "content": "<DataFrame with predictions>",
    "content_type": "text/csv",
    "content_size": 50000
  },
  "level": 1,
  "parent_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"
}
```

**Note**: `content` is stored as a DataFrame Python object (PickleType handles DB serialization). On download via `/$download`, it's serialized to the format specified by `content_type` (CSV in this example).

### Generic Artifact

```json
{
  "id": "01ARZ3NDEKTSV4RRFFQ69G5FAY",
  "name": "experiment-metadata",
  "data": {
    "type": "generic",
    "metadata": {
      "experiment_name": "rainfall-prediction-v1",
      "dataset_path": "/data/rainfall_2024.csv",
      "notes": "Initial baseline experiment",
      "custom_field": "arbitrary value"
    },
    "content": null,
    "content_type": null,
    "content_size": null
  },
  "level": 0,
  "parent_id": null
}
```
