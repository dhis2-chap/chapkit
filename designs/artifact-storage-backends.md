# Artifact Size in API

**Status:** DRAFT
**Created:** 2025-11-18
**Target Version:** 0.11.0
**Related Design:** [workspace-artifact-storage.md](./workspace-artifact-storage.md)

---

## Summary

Expose artifact content size in the artifact API responses. Currently size is stored in metadata but not surfaced.

**Current:** Size stored in `artifact.data["content_size"]` but not in API schema
**Proposed:** Add `content_size` field to ArtifactOut schema

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
    # No top-level content_size field
}
```

**Issues:**
1. Can't see artifact size without parsing data field
2. Can't sort/filter artifacts by size
3. Can't monitor storage usage via API
4. Can't display size in UI/dashboards

---

## Goals

1. Expose `content_size` as top-level field in ArtifactOut schema
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
    content_size: int | None  # NEW: Extracted from data
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
        content_size=artifact.data.get("content_size"),  # Extract from data
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
    "content_size": 314572800,  # NEW: Top-level field
    "created_at": "2025-11-18T10:00:00Z",
    "updated_at": "2025-11-18T10:00:00Z"
}
```

---

## Implementation

**Files:**
- `src/chapkit/artifact/schemas.py` - Add content_size field
- `src/chapkit/artifact/repository.py` or `manager.py` - Extract size when converting to schema

**Changes:**
1. Add `content_size: int | None` to ArtifactOut
2. Extract from `artifact.data.get("content_size")` when loading
3. Update tests

---

## Testing

- Unit tests for schema with content_size
- Integration tests for API responses
- Test null content_size (for artifacts without size)

---

## Success Criteria

- [ ] content_size exposed in ArtifactOut schema
- [ ] GET /api/v1/artifacts/{id} returns content_size
- [ ] GET /api/v1/artifacts returns content_size for all artifacts
- [ ] Null content_size handled gracefully (old artifacts)
- [ ] Documentation updated
- [ ] Tests pass

---

## References

**Files:**
- `src/chapkit/artifact/schemas.py`
- `src/chapkit/artifact/repository.py` or `manager.py`

**Related:** [workspace-artifact-storage.md](./workspace-artifact-storage.md) - Stores content_size in metadata
