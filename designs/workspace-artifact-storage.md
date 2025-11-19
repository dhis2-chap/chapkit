# Workspace Artifact Storage for ShellModelRunner

**Status:** DRAFT
**Created:** 2025-11-18
**Target Version:** 0.10.0
**Related Design:** [shell-runner-refactor.md](./shell-runner-refactor.md)

---

## Summary

Store ShellModelRunner training workspaces as zip artifacts to enable debugging, multi-file models, and proper failure handling. Predictions remain structured DataFrames.

**Key Changes:**
- Training: Zip entire workspace (model + logs + plots + metrics), store as bytes
- Prediction: Unzip workspace, extract model, run script, return DataFrame
- Failed/incomplete training: Block prediction with ValueError
- Workspace cleanup: Via existing artifact deletion API

**Scope:**
- ShellModelRunner only (FunctionalModelRunner unaffected)
- Breaking change: Pre-0.10.0 training artifacts incompatible

---

## Breaking Changes

**BREAKING:** ShellModelRunner training artifacts from <0.10.0 cannot be used for prediction after upgrade.

**Migration:**
1. Export critical predictions before upgrading (optional)
2. Upgrade to 0.10.0
3. Retrain all ShellModelRunner models
4. Delete old training artifacts to free storage (optional)

**Why:** Artifact content changes from pickled model object to workspace zip. No backward compatibility to minimize complexity.

---

## Implementation

### Phase 1: Core Workspace Storage

**Files:** `runner.py`, `manager.py`

**Training (ShellModelRunner.on_train()):**
```python
# After script execution, create workspace zip
import zipfile
import tempfile
from pathlib import Path

# Stream zip to temp file (not memory)
zip_file_path = Path(tempfile.mktemp(suffix=".zip"))
try:
    # Create zip with maximum compression
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zf.write(file_path, arcname)

    # Validate zip integrity
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise ValueError(f"Corrupted file in workspace zip: {bad_file}")

    # Read zip for storage
    workspace_content = zip_file_path.read_bytes()

finally:
    # Cleanup
    if zip_file_path.exists():
        zip_file_path.unlink()
    shutil.rmtree(temp_dir, ignore_errors=True)

return {
    "workspace_content": workspace_content,
    "content_type": "application/zip",
    "content_size": len(workspace_content),
    "exit_code": result.returncode,
    "stdout": result.stdout,
    "stderr": result.stderr,
}
```

**Prediction (ShellModelRunner.on_predict()):**
```python
# Extract workspace from training artifact
from io import BytesIO
import zipfile

zip_buffer = BytesIO(artifact_data["content"])
with zipfile.ZipFile(zip_buffer, 'r') as zf:
    zf.extractall(temp_dir)

# Write prediction data
historic_data.to_csv(temp_dir / "historic.csv", index=False)
future_data.to_csv(temp_dir / "future.csv", index=False)

# Run prediction script
# Script handles model loading from workspace (framework doesn't validate)
result = subprocess.run(command, cwd=temp_dir, capture_output=True, text=True)

# Read and return predictions
output_file = temp_dir / "predictions.csv"
if not output_file.exists():
    raise ValueError("Prediction script did not create predictions.csv")

return pd.read_csv(output_file)
```

### Phase 2: Failed Training Blocking

**File:** `manager.py`

```python
# In MLManager._predict_task():
training_metadata = training_artifact.data.get("metadata", {})
training_status = training_metadata.get("status", "unknown")

if training_status == "failed":
    exit_code = training_metadata.get("exit_code", "unknown")
    raise ValueError(
        f"Cannot predict using failed training artifact {artifact_id}. "
        f"Training script exited with code {exit_code}."
    )
```

### Phase 3: Testing & Documentation

- Unit tests for workspace storage, extraction, and failure blocking
- Integration tests with multi-file models (TensorFlow SavedModel)
- Update `docs/guides/ml-workflows.md` with workspace artifacts section
- Version bump to 0.10.0

---

## Artifact Schema

### Training Artifact
```python
{
    "type": "ml_training",
    "metadata": {
        "status": "success" | "failed",  # Reflects script exit code only
        "exit_code": int,
        "stdout": str,
        "stderr": str,
        "config_id": str,
        "started_at": str,
        "completed_at": str,
        "duration_seconds": float,
    },
    "content": bytes,             # Zip of entire workspace
    "content_type": "application/zip",
    "content_size": int,
}
```

**Status semantics:**
- `status="success"` means script exited with code 0
- `status="failed"` means script exited with non-zero code
- Status does NOT indicate whether model was created (script's responsibility)

### Prediction Artifact
```python
{
    "type": "ml_prediction",
    "metadata": {
        "status": "success" | "failed",  # Reflects script exit code only
        "config_id": str,
        "started_at": str,
        "completed_at": str,
        "duration_seconds": float,
    },
    "content": DataFrame,         # UNCHANGED
    "content_type": "application/vnd.chapkit.dataframe+json",
    "content_size": None,
}
```

---

## Design Decisions

1. **Training-only workspace storage** - Predictions remain DataFrames for queryability
2. **Full workspace zip** - Entire temp_dir (model, logs, plots, metrics, config)
3. **Compression level 9** - Maximum compression for minimal storage
4. **Stream to temp file** - Avoid in-memory buffering for large workspaces
5. **Zip integrity validation** - Prevent corrupted artifacts
6. **Failed training blocks prediction** - ValueError if status="failed" (exit_code != 0)
7. **No model validation** - Script handles model format/structure (framework agnostic)
8. **Status reflects script execution** - success/failed based on exit code, not model creation
9. **Cleanup via artifact API** - DELETE /api/v1/artifacts/{id} removes workspace
10. **No size limits** - Users manage workspace sizes (SQLite 2GB BLOB hard limit)
11. **Breaking change** - No backward compatibility for simplicity

---

## Storage Considerations

**Typical workspace sizes:**
- Simple models: 100KB - 5MB
- Deep learning: 10MB - 500MB
- With plots/logs: +10-50MB

**Limits:**
- SQLite BLOB hard limit: 2GB
- No enforced size restrictions
- Large workspaces (>500MB) impact database performance

**Memory:**
- Zip creation streams to temp file (low memory)
- Reading zip for storage loads full bytes into memory
- Large workspaces (>1GB) may cause memory pressure during artifact creation

**Monitoring:**
```sql
SELECT id,
       data->>'$.type' as type,
       ROUND(CAST(data->>'$.content_size' AS REAL) / 1024 / 1024, 2) as size_mb
FROM artifacts
WHERE data->>'$.type' = 'ml_training'
ORDER BY CAST(data->>'$.content_size' AS REAL) DESC;
```

---

## Testing

**Unit Tests (ShellModelRunner):**
- Workspace zip creation and structure validation
- Zip integrity validation (testzip)
- Workspace extraction during prediction
- Failed training blocking (status="failed", exit_code != 0)
- Corrupted zip handling
- temp_dir cleanup after zip creation
- Update test_shell_runner_cleanup_temp_files

**Integration Tests:**
- End-to-end train/predict with workspace artifacts
- Multi-file models (TensorFlow SavedModel directories)
- Failed training blocking via API (HTTP 400)
- Prediction script failure when model missing (natural failure)
- Large workspace (100MB+) performance
- Compression level 9 impact on training time

---

## Documentation Updates

**ML Workflows Guide (`docs/guides/ml-workflows.md`):**

Add section "Training Workspace Artifacts (ShellModelRunner v0.10.0+)":
- Workspace storage enables debugging and multi-file models
- Training artifacts store full workspace as zip
- Predictions remain DataFrames (queryable, unchanged)
- Failed/incomplete training blocks prediction
- Workspace cleanup via artifact deletion
- Storage considerations (sizes, limits, monitoring)
- Breaking change migration guide

**Postman Collection (`examples/ml_shell/postman_collection.json`):**

Add artifact exploration endpoints:
- `GET /api/v1/artifacts` - List all artifacts (filter by type, view training artifacts)
- `GET /api/v1/artifacts/{id}` - Download specific training artifact (workspace zip)

Demonstrates:
- Listing training artifacts after train operations
- Downloading workspace zips for local debugging
- Inspecting workspace contents (model, logs, plots, metrics)
- Artifact retrieval workflow

---

## Success Criteria

- [ ] Training creates workspace zip (compression level 9, streamed to temp file)
- [ ] Zip integrity validated before storage
- [ ] Prediction extracts workspace and runs script
- [ ] Failed training blocks prediction (status="failed" raises ValueError)
- [ ] Status reflects script exit code only (not model creation)
- [ ] temp_dir cleanup after zip creation
- [ ] All unit/integration tests pass
- [ ] test_shell_runner_cleanup_temp_files updated
- [ ] docs/guides/ml-workflows.md updated with migration guide
- [ ] examples/ml_shell/postman_collection.json updated with artifact endpoints
- [ ] Version bumped to 0.10.0

---

## Alternatives Considered

**Two Artifacts Per Operation:** Separate workspace and data artifacts. Rejected (complexity, double storage).

**Dual Content Schema:** Single artifact with `content: {predictions: DataFrame, workspace: bytes}`. Rejected (messy schema).

**Workspace for Training Only (SELECTED):** Training stores workspace zip, predictions store DataFrame. Balances debugging with queryability.

**Prediction Workspace Storage:** Store prediction workspaces too. Rejected (predictions are DataFrames for queryability).

---

## References

**Files:**
- `src/chapkit/ml/runner.py` (ShellModelRunner.on_train, on_predict)
- `src/chapkit/ml/manager.py` (MLManager._train_task, _predict_task)
- `src/chapkit/artifact/schemas.py` (MLTrainingArtifactData, MLPredictionArtifactData)
- `tests/test_ml_shell_runner.py`

**Related:** [shell-runner-refactor.md](./shell-runner-refactor.md)
