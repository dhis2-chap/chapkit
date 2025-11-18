# Artifact Size in API

**Status:** DRAFT
**Created:** 2025-11-18
**Target Version:** 0.11.0
**Related Design:** [workspace-artifact-storage.md](./workspace-artifact-storage.md)

---

## Summary

Expose artifact size in the artifact API responses. Currently size is stored in metadata but not surfaced.

**Current:** Size stored in `artifact.data["content_size"]` but not in API schema
**Proposed:** Add `size` field to ArtifactOut schema

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

But API response doesn't include size:
```python
# GET /api/v1/artifacts/{id}
{
    "id": "01H...",
    "parent_id": None,
    "level": 0,
    "data": {...},  # Includes content_size in nested data
    "created_at": "...",
    "updated_at": "..."
    # No top-level size field
}
```

**Issues:**
1. Can't see artifact size without parsing data field
2. Can't monitor storage usage via API
3. Can't display size in UI/dashboards

---

## Goals

1. Expose `size` as top-level field in ArtifactOut schema
2. Make artifact size visible in API responses without parsing nested data

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
    size: int | None  # NEW: Extracted from data["content_size"]
    created_at: datetime
    updated_at: datetime
```

### Repository/Manager Changes

Extract size when loading artifacts:

```python
def to_schema(artifact: Artifact) -> ArtifactOut:
    """Convert ORM model to API schema."""
    return ArtifactOut(
        id=artifact.id,
        parent_id=artifact.parent_id,
        level=artifact.level,
        data=artifact.data,
        size=artifact.data.get("content_size"),  # Extract from data
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )
```

### API Example

**Get artifact with size:**
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
    "size": 314572800,  # NEW: Top-level field
    "created_at": "2025-11-18T10:00:00Z",
    "updated_at": "2025-11-18T10:00:00Z"
}
```

---

## Implementation

**Files:**
- `src/chapkit/artifact/schemas.py` - Add size field
- `src/chapkit/artifact/repository.py` or `manager.py` - Extract size when converting to schema

**Changes:**
1. Add `size: int | None` to ArtifactOut
2. Extract from `artifact.data.get("content_size")` when loading
3. Update tests

---

## Testing

- Unit tests for schema with size field
- Integration tests for API responses
- Test null size (for artifacts without size)

---

## Success Criteria

- [ ] size exposed in ArtifactOut schema
- [ ] GET /api/v1/artifacts/{id} returns size
- [ ] GET /api/v1/artifacts returns size for all artifacts
- [ ] Null size handled gracefully (old artifacts)
- [ ] Documentation updated
- [ ] Tests pass

---

## References

**Files:**
- `src/chapkit/artifact/schemas.py`
- `src/chapkit/artifact/repository.py` or `manager.py`

**Related:** [workspace-artifact-storage.md](./workspace-artifact-storage.md) - Stores content_size in metadata
