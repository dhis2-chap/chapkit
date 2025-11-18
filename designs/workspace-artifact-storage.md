# Workspace Artifact Storage for ShellModelRunner

**Status:** DRAFT
**Created:** 2025-11-18
**Target Version:** 0.10.0
**Related Design:** [shell-runner-refactor.md](./shell-runner-refactor.md)

---

## Summary

Store training workspace as zip artifacts. Enables debugging, multi-file models, and blocks prediction on failed training. Predictions remain structured DataFrames.

**Key Changes:**
- Training: Zip entire temp_dir, store as bytes
- Prediction: Unzip training artifact, return DataFrame (current behavior)
- Failed training: Block prediction with ValueError
- Training workspace contains model + logs + plots + metrics
- Predictions remain queryable as structured data

---

## Problem Statement

Current limitations:
1. **Lost artifacts** - plots, logs, metrics created during training are deleted
2. **Single-file models only** - TensorFlow SavedModel directories not supported
3. **No debugging** - can't inspect what files existed during training
4. **No failure handling** - can predict using failed training artifacts
5. **Unknown sizes** - content_size always None

---

## Implementation

### Phase 1: Core Workspace Storage

**Files:** `runner.py`, `manager.py`

Training (runner.py):
```python
# After script execution:
import zipfile
from io import BytesIO

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(temp_dir)
            zf.write(file_path, arcname)

return {
    "content": zip_buffer.getvalue(),
    "content_type": "application/zip",
    "content_size": len(zip_buffer.getvalue()),
    "exit_code": result.returncode,
    "stdout": result.stdout,
    "stderr": result.stderr,
}
```

Prediction (runner.py):
```python
# Unzip training workspace to access model and training files
zip_buffer = BytesIO(artifact_data["content"])
with zipfile.ZipFile(zip_buffer, 'r') as zf:
    zf.extractall(temp_dir)

# Write current prediction data
historic_file = temp_dir / "historic.csv"
historic_data.to_csv(historic_file, index=False)

future_file = temp_dir / "future.csv"
future_data.to_csv(future_file, index=False)

# Run prediction script
result = subprocess.run(command, cwd=temp_dir, capture_output=True, text=True)

# Read predictions
output_file = temp_dir / "predictions.csv"
predictions = pd.read_csv(output_file)

# Return DataFrame (current behavior)
return predictions
```

### Phase 2: Failed Training Blocking

**File:** `manager.py`

```python
# In MLManager.predict():
training_metadata = training_artifact.data.get("metadata", {})
training_status = training_metadata.get("status", "unknown")

if training_status == "failed":
    exit_code = training_metadata.get("exit_code", "unknown")
    raise ValueError(
        f"Cannot predict using failed training artifact {artifact_id}. "
        f"Training exited with code {exit_code}."
    )
```

### Phase 3: Testing & Documentation

- Unit tests for workspace artifacts and failure blocking
- Integration tests with ml_shell example
- Update docs/guides/ml-workflows.md with workspace artifact section
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
        "started_at": str,
        "completed_at": str,
        "duration_seconds": float,
    },
    "content": bytes,  # Zip of entire workspace
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
    "content": DataFrame,  # Predictions (current behavior)
    "content_type": "application/vnd.chapkit.dataframe+json",
    "content_size": None,
}
```

---

## Design Decisions

1. **Workspace storage:** Training only (not prediction)
   - Training needs debugging (plots, logs, metrics, multi-file models)
   - Predictions are queryable DataFrames (current behavior preserved)
   - Training workspace available during prediction via unzip
2. **Zip contents:** Entire workspace (project files + outputs)
3. **Prediction artifacts:** DataFrame only (structured, queryable)
4. **Large artifacts:** SQLite only, document 2GB limit
5. **Failed training:** Create artifact but block prediction with ValueError

---

## Testing

### Unit Tests
- `test_train_creates_workspace_artifact()` - Verify zip storage
- `test_artifact_contains_all_workspace_files()` - Check contents
- `test_predict_unzips_workspace()` - Verify extraction
- `test_artifact_content_size_calculated()` - Size tracking
- `test_predict_returns_dataframe()` - Prediction artifact is DataFrame
- `test_predict_blocks_failed_training()` - Failure blocking
- `test_failed_training_metadata_complete()` - exit_code/stderr

### Integration Tests
- Train/predict end-to-end with workspace training artifact
- Multi-file model support (TensorFlow SavedModel)
- Failed training blocking via API (HTTP 400)
- Predictions queryable as DataFrame

---

## Documentation Updates

Add to `docs/guides/ml-workflows.md`:

**Section: "Training Workspace Artifacts"**
- Explain training artifacts store full workspace as zip
- Show multi-file model examples (TensorFlow, PyTorch)
- Document training artifacts (plots, logs, metrics)
- Explain failed training protection
- Note storage size (~500KB-200MB typical for training)
- SQLite 2GB limit
- Clarify predictions remain DataFrames (queryable, current behavior)

---

## Storage Monitoring

```sql
-- Check training artifact sizes
SELECT
    id,
    data->>'$.type' as type,
    data->>'$.content_size' as size_bytes,
    ROUND(CAST(data->>'$.content_size' AS REAL) / 1024 / 1024, 2) as size_mb
FROM artifacts
WHERE data->>'$.type' = 'ml_training'
ORDER BY CAST(data->>'$.content_size' AS REAL) DESC;
```

---

## Success Criteria

- [ ] Training creates workspace zip artifacts
- [ ] Prediction unzips training workspace correctly
- [ ] Prediction returns DataFrame (current behavior)
- [ ] Failed training blocks prediction
- [ ] content_size calculated accurately for training
- [ ] exit_code/stdout/stderr captured in training metadata
- [ ] >95% test coverage
- [ ] Documentation updated

---

## Alternatives Considered

### Option 1: Two Artifacts Per Operation
Store separate workspace and data artifacts for each operation. Training creates workspace artifact + model artifact, prediction creates workspace artifact + predictions artifact. Rejected due to complexity and double storage.

### Option 2: Dual Content Schema
Single artifact with `content: {predictions: DataFrame, workspace: bytes}`. Rejected due to messy schema mixing structured data with binary zip.

### Option 3: Workspace for Training Only (SELECTED)
Training stores workspace zip, predictions store DataFrame. Balances debugging capability with queryable predictions.

### Option 4: Both in One Artifact
Prediction artifact stores both `predictions: DataFrame` and `workspace: bytes` as separate fields. Could be reconsidered if prediction debugging becomes critical.

---

## References

**Files:**
- `src/chapkit/ml/runner.py`
- `src/chapkit/ml/manager.py`
- `src/chapkit/ml/schemas.py`
- `docs/guides/ml-workflows.md`
- `tests/test_ml_shell_runner.py`

**Related:** [shell-runner-refactor.md](./shell-runner-refactor.md)
