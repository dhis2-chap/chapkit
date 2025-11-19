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

    # Check if model file exists
    model_format = config.model_format
    model_path = temp_dir / f"model.{model_format}"
    model_exists = model_path.exists()

finally:
    # Cleanup
    if zip_file_path.exists():
        zip_file_path.unlink()
    shutil.rmtree(temp_dir, ignore_errors=True)

return {
    "workspace_content": workspace_content,
    "content_type": "application/zip",
    "content_size": len(workspace_content),
    "model_format": model_format,
    "model_exists": model_exists,
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

# Validate model exists
training_metadata = artifact_data.get("metadata", {})
model_format = training_metadata.get("model_format", "pkl")
model_exists = training_metadata.get("model_exists", False)

if not model_exists:
    raise ValueError("Training artifact contains no model file. Cannot predict.")

model_path = temp_dir / f"model.{model_format}"
if not model_path.exists():
    raise ValueError(f"Model file not found in workspace: model.{model_format}")

# Write prediction data and run script
historic_data.to_csv(temp_dir / "historic.csv", index=False)
future_data.to_csv(temp_dir / "future.csv", index=False)
result = subprocess.run(command, cwd=temp_dir, capture_output=True, text=True)

# Return DataFrame
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
        f"Training exited with code {exit_code}."
    )

if not training_metadata.get("model_exists", False):
    raise ValueError(
        f"Cannot predict using training artifact {artifact_id}. "
        f"No model file was created during training."
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
        "status": "success" | "failed",
        "exit_code": int,
        "stdout": str,
        "stderr": str,
        "config_id": str,
        "model_format": str,      # NEW: e.g., "pkl", "joblib", "h5"
        "model_exists": bool,     # NEW: model file present in workspace
        "started_at": str,
        "completed_at": str,
        "duration_seconds": float,
    },
    "content": bytes,             # Zip of entire workspace
    "content_type": "application/zip",
    "content_size": int,
}
```

### Prediction Artifact
```python
{
    "type": "ml_prediction",
    "metadata": {
        "status": "success" | "failed",
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
6. **Failed training blocks prediction** - ValueError if status="failed" or model_exists=False
7. **Cleanup via artifact API** - DELETE /api/v1/artifacts/{id} removes workspace
8. **No size limits** - Users manage workspace sizes (SQLite 2GB BLOB hard limit)
9. **Breaking change** - No backward compatibility for simplicity

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
- Metadata fields (model_format, model_exists)
- Failed training blocking (status="failed")
- Workspace-only training blocking (model_exists=False)
- Corrupted zip handling
- temp_dir cleanup after zip creation
- Update test_shell_runner_cleanup_temp_files

**Integration Tests:**
- End-to-end train/predict with workspace artifacts
- Multi-file models (TensorFlow SavedModel)
- Failed training blocking via API (HTTP 400)
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

---

## Success Criteria

- [ ] Training creates workspace zip (compression level 9, streamed to temp file)
- [ ] Zip integrity validated before storage
- [ ] Prediction extracts workspace and validates model file exists
- [ ] Failed training blocks prediction (ValueError)
- [ ] Workspace-only training blocks prediction (ValueError)
- [ ] Metadata includes model_format and model_exists
- [ ] temp_dir cleanup after zip creation
- [ ] All unit/integration tests pass
- [ ] test_shell_runner_cleanup_temp_files updated
- [ ] docs/guides/ml-workflows.md updated with migration guide
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
