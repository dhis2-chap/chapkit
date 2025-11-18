# Artifact Size in API

**Status:** DRAFT
**Created:** 2025-11-18
**Target Version:** 0.11.0
**Related Design:** [workspace-artifact-storage.md](./workspace-artifact-storage.md)

---

## Summary

Add content_type and content_size as direct fields on the Artifact model and expose in API responses. Currently stored only in nested data dict.

**Current:** Stored in `artifact.data["content_type"]` and `artifact.data["content_size"]` (nested, not queryable)
**Proposed:** Add `content_type` and `content_size` as direct database columns and expose in API

---

## Problem Statement

Workspace artifacts (from workspace-artifact-storage.md) store size in metadata:
```python
{
    "type": "ml_training",
    "content": bytes,  # Workspace zip
    "content_type": "application/zip",
    "content_size": 314572800,  # 300MB - stored but not exposed
}
```

But API response doesn't include content metadata:
```python
# GET /api/v1/artifacts/{id}
{
    "id": "01H...",
    "parent_id": None,
    "level": 0,
    "data": {...},  # Includes content_type and content_size in nested data
    "created_at": "...",
    "updated_at": "..."
    # No top-level content_type or content_size fields
}
```

**Issues:**
1. Can't see artifact size/type without parsing data field
2. Can't query/filter artifacts by size or type (not in SQL columns)
3. Can't create database indexes for performance
4. Can't monitor storage usage via API
5. Can't display size or content type in UI/dashboards
6. Need to deserialize pickled data just to get metadata

---

## Goals

1. Add `content_type` and `content_size` as direct database columns on Artifact model
2. Expose fields in ArtifactOut schema
3. Enable SQL queries/filters on artifact size and type
4. Make artifact metadata accessible without deserializing data

---

## Design

### Artifact Model Changes

**File:** `src/chapkit/artifact/models.py`

Add direct columns for content metadata:

```python
# Current
class Artifact(Entity):
    id: Mapped[str] = mapped_column(primary_key=True)
    parent_id: Mapped[str | None]
    level: Mapped[int]
    data: Mapped[Any] = mapped_column(PickleType)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

# New
class Artifact(Entity):
    id: Mapped[str] = mapped_column(primary_key=True)
    parent_id: Mapped[str | None]
    level: Mapped[int]
    data: Mapped[Any] = mapped_column(PickleType)
    content_type: Mapped[str | None]  # NEW: Direct column
    content_size: Mapped[int | None]  # NEW: Direct column
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

**Benefits:**
- SQL queryable (WHERE content_size > 100000000)
- Indexable for performance
- No data deserialization needed
- Clear schema

### Schema Changes

**File:** `src/chapkit/artifact/schemas.py`

```python
# Current
class ArtifactOut(BaseModel):
    id: str
    parent_id: str | None
    level: int
    data: dict
    created_at: datetime
    updated_at: datetime

# New
class ArtifactOut(BaseModel):
    id: str
    parent_id: str | None
    level: int
    data: dict
    content_type: str | None  # NEW: Direct field
    content_size: int | None  # NEW: Direct field
    created_at: datetime
    updated_at: datetime
```

### Repository/Manager Changes

**Creating artifacts** - Always set fields from data:

```python
async def create(self, artifact_in: ArtifactIn) -> Artifact:
    """Create artifact with content metadata in direct fields."""
    artifact_data = artifact_in.data

    artifact = Artifact(
        id=str(ULID()),
        parent_id=artifact_in.parent_id,
        level=artifact_in.level or 0,
        data=artifact_data,
        content_type=artifact_data.get("content_type"),  # Always extract from data
        content_size=artifact_data.get("content_size"),  # Always extract from data
    )

    return await super().create(artifact)
```

**Note:** All artifact creation code (ML runner, etc.) should include content_type and content_size in data dict. Repository always extracts and sets direct fields.

**Converting to schema** - Map directly from fields:

```python
def to_schema(artifact: Artifact) -> ArtifactOut:
    """Convert ORM model to API schema."""
    return ArtifactOut(
        id=artifact.id,
        parent_id=artifact.parent_id,
        level=artifact.level,
        data=artifact.data,
        content_type=artifact.content_type,  # Direct field
        content_size=artifact.content_size,  # Direct field
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
```

### API Example

**Get artifact with content metadata:**
```http
GET /api/v1/artifacts/01H2PKW...

{
    "id": "01H2PKW...",
    "parent_id": null,
    "level": 0,
    "data": {
        "type": "ml_training",
        "content_type": "application/zip",
        "content_size": 314572800,
        ...
    },
    "content_type": "application/zip",  # NEW: Top-level field
    "content_size": 314572800,  # NEW: Top-level field
    "created_at": "2025-11-18T10:00:00Z",
    "updated_at": "2025-11-18T10:00:00Z"
}
```

---

## Implementation

**Files:**
- `src/chapkit/artifact/models.py` - Add content_type and content_size columns
- `src/chapkit/artifact/schemas.py` - Add content_type and content_size fields
- `src/chapkit/artifact/repository.py` - Set fields when creating artifacts

**Steps:**
1. Update Artifact model with new Mapped fields
2. Add content_type and content_size to ArtifactOut schema
3. Update repository create() to always extract and set fields from data dict
4. Ensure ML runner and other artifact creators include content_type and content_size in data
5. Update tests

---

## Testing

- Unit tests for schema with content_type and content_size fields
- Integration tests for API responses
- Test null values (for artifacts without content metadata)

---

## Success Criteria

- [ ] Artifact model has content_type and content_size columns
- [ ] content_type and content_size exposed in ArtifactOut schema
- [ ] Repository always extracts and sets fields when creating artifacts
- [ ] ML runner includes content_type and content_size in artifact data
- [ ] GET /api/v1/artifacts/{id} returns both fields
- [ ] GET /api/v1/artifacts returns both fields for all artifacts (for listing in modeling-app)
- [ ] Tests pass
- [ ] Documentation updated

---

## References

**Files:**
- `src/chapkit/artifact/models.py`
- `src/chapkit/artifact/schemas.py`
- `src/chapkit/artifact/repository.py`

**Related:** [workspace-artifact-storage.md](./workspace-artifact-storage.md) - Stores content_type and content_size in metadata
