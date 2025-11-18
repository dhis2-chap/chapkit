# Artifact Size in API

**Status:** DRAFT
**Created:** 2025-11-18
**Target Version:** 0.11.0
**Related Design:** [workspace-artifact-storage.md](./workspace-artifact-storage.md)

---

## Summary

Expose artifact content metadata in the artifact API responses. Currently content_type and content_size are stored in data but not surfaced.

**Current:** Stored in `artifact.data["content_type"]` and `artifact.data["content_size"]` but not in API schema
**Proposed:** Add `content_type` and `content_size` fields to ArtifactOut schema

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
2. Can't monitor storage usage via API
3. Can't display size or content type in UI/dashboards
4. Can't set proper Content-Type headers without parsing data

---

## Goals

1. Expose `content_type` and `content_size` as top-level fields in ArtifactOut schema
2. Make artifact content metadata visible in API responses without parsing nested data

---

## Design

### Schema Changes

**File:** `src/chapkit/artifact/schemas.py` (or wherever artifact schemas are)

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
    content_type: str | None  # NEW: Extracted from data["content_type"]
    content_size: int | None  # NEW: Extracted from data["content_size"]
    created_at: datetime
    updated_at: datetime
```

### Repository/Manager Changes

Extract content metadata when loading artifacts:

```python
def to_schema(artifact: Artifact) -> ArtifactOut:
    """Convert ORM model to API schema."""
    return ArtifactOut(
        id=artifact.id,
        parent_id=artifact.parent_id,
        level=artifact.level,
        data=artifact.data,
        content_type=artifact.data.get("content_type"),  # Extract from data
        content_size=artifact.data.get("content_size"),  # Extract from data
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
- `src/chapkit/artifact/schemas.py` - Add content_type and content_size fields
- `src/chapkit/artifact/repository.py` or `manager.py` - Extract fields when converting to schema

**Changes:**
1. Add `content_type: str | None` to ArtifactOut
2. Add `content_size: int | None` to ArtifactOut
3. Extract both from `artifact.data` when loading
4. Update tests

---

## Testing

- Unit tests for schema with content_type and content_size fields
- Integration tests for API responses
- Test null values (for artifacts without content metadata)

---

## Success Criteria

- [ ] content_type and content_size exposed in ArtifactOut schema
- [ ] GET /api/v1/artifacts/{id} returns both fields
- [ ] GET /api/v1/artifacts returns both fields for all artifacts
- [ ] Null values handled gracefully (old artifacts)
- [ ] Documentation updated
- [ ] Tests pass

---

## References

**Files:**
- `src/chapkit/artifact/schemas.py`
- `src/chapkit/artifact/repository.py` or `manager.py`

**Related:** [workspace-artifact-storage.md](./workspace-artifact-storage.md) - Stores content_type and content_size in metadata
