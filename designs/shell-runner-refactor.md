# ShellModelRunner Refactoring - Full Isolation Design

**Status:** Ready for Implementation
**Created:** 2025-11-18
**Branch:** `refactor/shell-runner-isolation`

---

## Quick Reference

**What:** Copy entire project directory to temp workspace for full isolation

**Why:** Enable relative imports, easy debugging, better error messages

**API Change:**
```python
# Simple - just 3 parameters!
runner = ShellModelRunner(
    train_command="python scripts/train.py --config {config_file} --data {data_file}",
    predict_command="python scripts/predict.py --config {config_file} --model {model_file} --output {output_file}",
    model_format="pickle",
)
```

**Key Implementation:** `self.project_root = Path.cwd()` → copy to temp → execute with `cwd=temp_dir`

---

## Executive Summary

This design proposes a major refactoring of the `ShellModelRunner` class to implement a **full isolation pattern** where all necessary files (scripts, libraries, dependencies) are copied to a temporary execution directory. This change addresses critical limitations in the current implementation and significantly improves debuggability.

**Key Changes:**
- Copy entire project directory (where main.py is located) to temp workspace
- Enable relative imports in scripts (e.g., `./lib.R`, `from .utils import ...`)
- Simplify debugging by creating self-contained execution snapshots
- Better error handling (full output, no truncation)
- Fix documentation: config format is YAML (not JSON)
- Remove unnecessary `SCRIPTS_DIR` pattern

**Note:** No migration needed - feature not yet in use.

---

## Implementation Phases

### Phase 1: Core Isolation [CRITICAL]

**Goal:** Implement full project directory copying with proper isolation.

**Files to Modify:**
- `src/chapkit/ml/runner.py` - Update `ShellModelRunner` class

**Tasks:**
1. ✅ **Update `__init__` signature**
   - Remove any old parameters
   - Keep only: `train_command`, `predict_command`, `model_format`
   - Add: `self.project_root = Path.cwd()`

2. ✅ **Implement `_prepare_workspace()` method**
   - Use `shutil.copytree()` to copy `self.project_root` to `temp_dir`
   - Add comprehensive ignore patterns:
     ```python
     ignore=shutil.ignore_patterns(
         '.venv', 'venv', '__pycache__', '*.pyc', '*.pyo',
         '*.egg-info', '.pytest_cache', '.mypy_cache', '.ruff_cache',
         'node_modules', '.git', '.gitignore', '.vscode', '.idea',
         '.DS_Store', 'build', 'dist', '*.so', '*.dylib',
     )
     ```
   - Log: `copied_project_directory` with src/dest paths

3. ✅ **Update `on_train()` method**
   - Call `_prepare_workspace(temp_dir)` BEFORE writing data files
   - Write data files (config.yml, data.csv, geo.json) to `temp_dir` root
   - Execute command with `cwd=temp_dir`
   - Variable substitution uses relative paths only

4. ✅ **Update `on_predict()` method**
   - Same pattern as `on_train()`
   - Copy workspace, write files, execute with `cwd=temp_dir`

5. ✅ **Remove truncated logging**
   - Change `stdout[:500]` to full `stdout`
   - Change `stderr[:500]` to full `stderr`

**Tests to Write:**
- `tests/test_ml_shell_runner.py`:
  - `test_copies_entire_project_directory()`
  - `test_ignores_venv_directory()`
  - `test_ignores_node_modules()`
  - `test_ignores_pycache()`
  - `test_project_structure_preserved()`
  - `test_uses_relative_paths()`

**Acceptance Criteria:**
- [ ] Project directory fully copied to temp workspace
- [ ] `.venv/` excluded from copy
- [ ] `node_modules/` excluded from copy
- [ ] Commands execute with `cwd=temp_dir`
- [ ] All existing tests still pass
- [ ] New tests pass

**Deliverable:** Working prototype with full isolation

---

### Phase 2: Examples & Templates Update

**Goal:** Update all examples and templates to use new simplified pattern.

**Files to Modify:**
- `examples/ml_shell/main.py`
- `src/chapkit/cli/templates/main_shell.py.jinja2`
- `src/chapkit/cli/templates/scripts/train_model.py.jinja2`
- `src/chapkit/cli/templates/scripts/predict_model.py.jinja2`

**Tasks:**

1. ✅ **Update `examples/ml_shell/main.py`** (Lines 12, 40-58)
   ```diff
   - SCRIPTS_DIR = Path(__file__).parent / "scripts"

   - train_command = f"python {SCRIPTS_DIR}/train_model.py ..."
   + train_command = "python scripts/train_model.py ..."

   - predict_command = f"python {SCRIPTS_DIR}/predict_model.py ..."
   + predict_command = "python scripts/predict_model.py ..."
   ```

2. ✅ **Add `examples/ml_shell/lib.py`**
   - Create shared utility module to demonstrate relative imports
   - Example functions: `validate_data()`, `preprocess_features()`

3. ✅ **Update `examples/ml_shell/scripts/train_model.py`**
   - Add: `from lib import validate_data, preprocess_features`
   - Demonstrate that relative imports work

4. ✅ **Update CLI template `main_shell.py.jinja2`** (Lines 12, 36-58)
   ```diff
   - # Get absolute path to scripts directory
   - SCRIPTS_DIR = Path(__file__).parent / "scripts"

   - train_command = f"python {SCRIPTS_DIR}/train_model.py ..."
   + train_command = "python scripts/train_model.py ..."
   ```

5. ✅ **Update script templates**
   - Ensure they work with relative paths
   - Add comments about relative imports capability

**Tests to Update:**
- `tests/test_example_ml_shell.py`:
  - Update all tests to work with new pattern
  - Add `test_relative_imports_work()`
  - Add `test_full_project_structure_available()`

**Acceptance Criteria:**
- [ ] All examples use simple string commands (no f-strings)
- [ ] No `SCRIPTS_DIR` variables anywhere
- [ ] `examples/ml_shell/` demonstrates relative imports
- [ ] CLI templates generate correct code
- [ ] All integration tests pass

**Deliverable:** Updated examples demonstrating new patterns

---

### Phase 3: Documentation

**Goal:** Update documentation to reflect new design.

**Files to Modify:**
- `docs/guides/ml-workflows.md`

**Tasks:**

1. ✅ **Fix config format** (Line 224)
   ```diff
   - `{config_file}` - JSON config file
   + `{config_file}` - YAML config file
   ```

2. ✅ **Add section: "Script Organization & Relative Imports"** (After line 274)
   - Show project structure
   - Demonstrate relative imports in Python/R/Julia
   - Explain full isolation pattern
   - Show debugging workflow

3. ✅ **Update ShellModelRunner examples**
   - Remove all `SCRIPTS_DIR` patterns
   - Show simple string commands
   - Emphasize simplicity

4. ✅ **Add debugging section**
   - How to read error messages with temp dir location
   - How temp directory contains full project snapshot

**Acceptance Criteria:**
- [ ] Documentation accurate (YAML not JSON)
- [ ] Relative imports documented with examples
- [ ] No mention of `SCRIPTS_DIR` pattern
- [ ] Debugging workflow clearly explained

**Deliverable:** Complete, accurate documentation

---

### Phase 4: Final Testing & Cleanup

**Goal:** Ensure everything works end-to-end.

**Tasks:**

1. ✅ **Run full test suite**
   ```bash
   make test
   make coverage
   ```

2. ✅ **Manual testing checklist**
   - [ ] `chapkit init --template shell test-project`
   - [ ] `cd test-project && uv sync`
   - [ ] Add relative import to script
   - [ ] `fastapi dev main.py`
   - [ ] Submit training job via API
   - [ ] Verify relative imports work
   - [ ] Verify `.venv/` not copied to temp dir
   - [ ] Check error messages show full output

3. ✅ **Performance check**
   - Measure directory copy overhead
   - Ensure reasonable for typical projects

4. ✅ **Update tests for full coverage**
   - Ensure >95% coverage on modified code
   - Add any missing edge case tests

**Acceptance Criteria:**
- [ ] All tests pass (`make test`)
- [ ] All linting passes (`make lint`)
- [ ] Coverage >95% on new code
- [ ] Manual testing checklist complete
- [ ] No performance regressions

**Deliverable:** Production-ready implementation

---

## Table of Contents

1. [Implementation Phases](#implementation-phases) **<-- START HERE**
2. [Problem Statement](#problem-statement)
3. [Goals](#goals)
4. [Core Design](#core-design-full-isolation-pattern)
5. [Technical Specification](#technical-specification)
6. [API Design Summary](#api-design-summary)
7. [Testing Strategy](#testing-strategy)
8. [Documentation Updates](#documentation-updates)
9. [Open Questions & Decisions](#open-questions--decisions)
10. [Success Criteria](#success-criteria)
11. [References](#references)

---

## Problem Statement

### Current Limitations

#### 1. No Support for Relative Imports
**Problem:** Scripts cannot use relative imports because they're executed in a temp directory without their surrounding context.

```python
# Current behavior:
# Script at: /app/scripts/train_model.py
# Tries to import: from .utils import preprocess
# Fails because only train_model.py exists in temp dir
```

**Impact:**
- Users must restructure code to avoid relative imports
- Can't use modular script organization
- R scripts can't use `source("./lib.R")`
- Julia scripts can't use `include("./utils.jl")`

#### 2. Hard to Debug Failures
**Problem:** When a script fails, only the individual input files exist in temp directory (which gets deleted).

**Current debugging workflow:**
1. Script fails with cryptic error
2. Temp directory already cleaned up
3. No way to reproduce exact execution environment
4. Must manually recreate files to test

**Desired workflow:**
1. Script fails
2. `tar -czf debug.tar.gz /tmp/chapkit_ml_train_XXXX/`
3. Share complete, reproducible environment
4. Colleague can examine all files, re-run command exactly

#### 3. Unnecessary SCRIPTS_DIR Pattern
**Problem:** Current examples use overly complex f-string pattern with `SCRIPTS_DIR`:

```python
SCRIPTS_DIR = Path(__file__).parent / "scripts"
train_command = f"python {SCRIPTS_DIR}/train_model.py --config {{config_file}} ..."
```

This is completely unnecessary. Users could simply write:
```python
train_command = "python scripts/train_model.py --config {config_file} ..."
```

**Impact:** Confusing, misleading examples that suggest complexity where none is needed.

#### 4. Documentation Inconsistency
**Problem:** Documentation says scripts receive "JSON config file" but implementation writes YAML.

**File:** `docs/guides/ml-workflows.md:224`
```markdown
- `{config_file}` - JSON config file  ← WRONG
```

**Actual:** `src/chapkit/ml/runner.py:123`
```python
config_file.write_text(yaml.safe_dump(config.model_dump(), indent=2))
```

**Impact:** User confusion, incorrect script implementations

#### 5. Missing Features
- Truncated error output (500 chars max)
- No output schema validation

---

## Goals

### Primary Goals
1. **Full Isolation:** Copy all necessary files to temp directory for self-contained execution
2. **Relative Imports:** Support modular script organization with relative paths
3. **Easy Debugging:** Create shareable execution snapshots
4. **Better Errors:** Full output, validation, clear messages

### Secondary Goals
1. Fix documentation inconsistencies
2. Improve test coverage for edge cases

### Non-Goals
- Change fundamental file-based approach (no gRPC/HTTP alternatives)
- Change data serialization formats (CSV/YAML/JSON remain)
- Change artifact storage mechanism
- Change job scheduling integration

---

## Core Design: Full Isolation Pattern

### Concept

Instead of creating individual files in a temp directory, **copy the entire project directory** (where main.py is located) to create a complete, isolated execution environment.

### Current Flow

```
1. Create temp dir: /tmp/chapkit_ml_train_ABC123/
2. Write individual files:
   - config.yml
   - data.csv
   - geo.json (optional)
3. Execute: python /app/scripts/train_model.py --config /tmp/.../config.yml ...
4. Script runs in isolation (no access to /app/scripts/lib.py)
5. Cleanup temp dir
```

### Proposed Flow

```
1. Create temp dir: /tmp/chapkit_ml_train_ABC123/
2. Copy entire project directory:
   - cp -r /app/* /tmp/chapkit_ml_train_ABC123/
   - Excludes: .venv, node_modules, __pycache__, .git, etc.
   - Now includes: scripts/, lib/, config files, everything!
3. Write generated data files to temp dir root:
   - /tmp/.../config.yml
   - /tmp/.../data.csv
   - /tmp/.../geo.json (optional)
4. Execute: python scripts/train_model.py --config config.yml --data data.csv ...
   (Note: relative paths within temp dir, cwd=temp_dir)
5. Scripts can import: from lib.utils import preprocess  ✓
6. Scripts can use: source("./lib/helpers.R")  ✓
7. On error: tar -czf debug.tar.gz /tmp/chapkit_ml_train_ABC123/
8. Cleanup temp dir (always delete)
```

### Directory Structure Example

**Before (Current):**
```
/tmp/chapkit_ml_train_ABC123/
├── config.yml
├── data.csv
└── model.pickle  (created by script)
```

**After (Proposed):**
```
/tmp/chapkit_ml_train_ABC123/
├── main.py              # Copied from project root
├── scripts/             # Copied from project
│   ├── train_model.py
│   ├── predict_model.py
│   └── lib.py          # ← Now available for relative imports!
├── lib/                 # Copied from project (if exists)
│   └── utils.py
├── config.yml           # Generated by runner
├── data.csv             # Generated by runner
├── geo.json             # Generated by runner (optional)
└── model.pickle         # Created by training script
```

**Note:** `.venv/`, `node_modules/`, `__pycache__/`, `.git/` are excluded from copying.

---

## Technical Specification

### New API Design

```python
class ShellModelRunner(BaseModelRunner[ConfigT]):
    def __init__(
        self,
        train_command: str,
        predict_command: str,
        model_format: str = "pickle",
    ) -> None:
        """Initialize shell runner with full isolation support.

        The runner automatically copies the entire project directory (where the
        ShellModelRunner is instantiated) to a temporary workspace, excluding
        .venv, node_modules, __pycache__, .git, and other build artifacts.

        Args:
            train_command: Command template for training (use relative paths)
            predict_command: Command template for prediction (use relative paths)
            model_format: File extension for model files (default: "pickle")
        """
```

**Usage:**
```python
runner = ShellModelRunner(
    train_command="python scripts/train_model.py --config {config_file} --data {data_file} --model {model_file}",
    predict_command="python scripts/predict_model.py --config {config_file} --model {model_file} --future {future_file} --output {output_file}",
    model_format="pickle",
)
```

**Key Simplifications:**
- No `script_dir` parameter - always copies from project root (CWD)
- No `timeout` parameter - ML jobs are async and can run for hours
- No `env` parameter - deferred to future enhancement
- No `preserve_on_error` parameter - always cleanup
- Project root is current working directory (`Path.cwd()`)

### Implementation Details

#### Determining Project Root

```python
def __init__(self, train_command: str, predict_command: str, ...) -> None:
    self.train_command = train_command
    self.predict_command = predict_command

    # Project root is current working directory
    # Users run: fastapi dev main.py (from project dir)
    # Docker sets WORKDIR to project root
    self.project_root = Path.cwd()

    logger.info("shell_runner_initialized", project_root=str(self.project_root))
```

#### File Copying Strategy

```python
def _prepare_workspace(self, temp_dir: Path) -> None:
    """Prepare isolated workspace with all necessary files."""

    # 1. Copy entire project directory to temp workspace
    shutil.copytree(
        self.project_root,
        temp_dir,
        ignore=shutil.ignore_patterns(
            # Python
            '.venv',
            'venv',
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '*.egg-info',
            '.pytest_cache',
            '.mypy_cache',
            '.ruff_cache',

            # JavaScript/Node
            'node_modules',

            # Version control
            '.git',
            '.gitignore',

            # IDEs
            '.vscode',
            '.idea',
            '.DS_Store',

            # Build artifacts
            'build',
            'dist',
            '*.so',
            '*.dylib',
        ),
        dirs_exist_ok=True,
    )
    logger.info("copied_project_directory", src=str(self.project_root), dest=str(temp_dir))

    # 2. Write generated data files (config, data, geo) to temp dir root
    # ... existing logic ...
```

**Note:** The venv is in PATH (e.g., `.venv/bin/python`), so we don't need to copy it. Scripts will use the same interpreter.

#### Command Template Adjustment

**Current (Absolute Paths):**
```python
SCRIPTS_DIR = Path(__file__).parent / "scripts"
command = f"python {SCRIPTS_DIR}/train.py --config {{config_file}} --data {{data_file}}"
# Template expands at instantiation to absolute path
```

**Proposed (Relative Paths):**
```python
command = "python scripts/train.py --config {config_file} --data {data_file}"
# Template uses relative paths
# Executed with cwd=temp_dir
# All project files available at: /tmp/xyz/
# Scripts can use: from scripts.lib import utils
```

**Key Change:** Commands use **relative paths** within temp directory, executed with `cwd=temp_dir`. The entire project structure is preserved.

#### Path Resolution

The runner provides variables for generated files:

```python
# Available in command templates (all relative to temp_dir):
# {config_file} = "config.yml"
# {data_file} = "data.csv"  (training only)
# {historic_file} = "historic.csv"  (prediction only)
# {future_file} = "future.csv"  (prediction only)
# {model_file} = "model.pkl"
# {output_file} = "predictions.csv"  (prediction only)
# {geo_file} = "geo.json"  (if provided, else empty string)
```

**Note:** Uses relative paths instead of the old f-string pattern with `SCRIPTS_DIR`.

#### Error Handling Enhancement

```python
if result["returncode"] != 0:
    error_msg = f"Script failed with exit code {result['returncode']}"

    # Full output (not truncated to 500 chars like before!)
    logger.error(
        "script_failed",
        command=command,
        exit_code=result["returncode"],
        stdout=result["stdout"],  # Full output
        stderr=result["stderr"],  # Full output
        temp_dir=str(temp_dir),
    )

    # Include stderr in exception for visibility
    raise RuntimeError(
        f"{error_msg}\n\n"
        f"Command: {command}\n"
        f"Working directory: {temp_dir}\n\n"
        f"Stderr:\n{result['stderr']}\n\n"
        f"Stdout:\n{result['stdout']}"
    )
```

**Note:** Temp directory is always cleaned up. Future enhancement could add a global debug mode to preserve temp dirs.

**Removed Feature:** Script validation is not needed - if the script doesn't exist in the project, copytree will succeed but the command will fail with a clear "file not found" error from the shell.

---

## API Design Summary

### Command Template Pattern

Use simple strings with relative paths:

```python
train_command = "python scripts/train.py --config {config_file} --data {data_file}"
predict_command = "python scripts/predict.py --config {config_file} --model {model_file} --future {future_file} --output {output_file}"
```

**No f-strings needed.** No `SCRIPTS_DIR` variable needed.

### Variable Substitution

All variables are relative paths within the temp directory:
- `{config_file}` = `"config.yml"`
- `{data_file}` = `"data.csv"`
- `{model_file}` = `"model.pkl"`
- etc.

### Relative Imports

Scripts can now use:

```python
# train_model.py
from scripts.lib import preprocess, validate
from lib.utils import load_config

# Or in R:
source("./lib.R")

# Or in Julia:
include("./utils.jl")
```

---

## Testing Strategy

### Unit Tests

**New tests in `tests/test_ml_shell_runner.py`:**

1. `test_shell_runner_copies_project_directory` - Verify full project copying
2. `test_shell_runner_ignores_venv` - Verify .venv excluded
3. `test_shell_runner_ignores_node_modules` - Verify node_modules excluded
4. `test_shell_runner_relative_imports` - Python relative imports work
5. `test_shell_runner_relative_paths` - Variable substitution uses relative paths
6. `test_shell_runner_project_structure_preserved` - Entire project structure intact

**Updated tests:**
1. Update all existing tests to use new API (remove f-strings with SCRIPTS_DIR)

### Integration Tests

**New test in `tests/test_example_ml_shell.py`:**

1. `test_shell_runner_with_library_module` - Train/predict with `from scripts.lib import utils`
2. `test_shell_runner_full_project_structure` - Verify entire project structure available in temp dir

### Manual Testing

**Checklist:**
- [ ] Python script with relative imports works
- [ ] R script with `source("./lib.R")` works (if R available)
- [ ] .venv is excluded from temp directory
- [ ] node_modules is excluded from temp directory
- [ ] Error message includes full stdout/stderr
- [ ] Can reproduce issue by copying project structure
- [ ] Long-running jobs (hours) work correctly

---

## Documentation Updates

### 1. ML Workflows Guide (`docs/guides/ml-workflows.md`)

**Changes:**

**Line 224 (Variable Substitution):**
```diff
- `{config_file}` - JSON config file
+ `{config_file}` - YAML config file
```

**Add new section after Line 274:**

```markdown
### Script Organization & Relative Imports

The ShellModelRunner automatically copies your entire project directory to an isolated workspace, enabling modular script organization and relative imports:

```python
# main.py
runner = ShellModelRunner(
    train_command="python scripts/train_model.py --config {config_file} --data {data_file} --model {model_file}",
    predict_command="python scripts/predict_model.py --config {config_file} --model {model_file} --future {future_file} --output {output_file}",
)
```

**Directory structure:**
```
your_project/
├── main.py
├── scripts/
│   ├── train_model.py
│   ├── predict_model.py
│   └── lib.py           # Shared utilities
└── lib/
    └── utils.py
```

**scripts/train_model.py can now use:**
```python
from scripts.lib import preprocess, validate  # Relative imports work!
from lib.utils import helper_function  # Access project modules!
```

**R scripts can use:**
```r
source("./scripts/lib.R")  # Relative paths work!
```

**Debugging:**
When a script fails, the error message includes the temp directory location and full output:
```bash
RuntimeError: Script failed with exit code 1

Command: python scripts/train_model.py --config config.yml --data data.csv
Working directory: /tmp/chapkit_ml_train_ABC123

Stderr:
<full error output>

Stdout:
<full script output>
```

You can then manually inspect or archive the temp directory before it's cleaned up.
```

### 2. CLI Template (`src/chapkit/cli/templates/main_shell.py.jinja2`)

**Update lines 36-58:**

```diff
# Training command template
train_command = (
-    f"python {SCRIPTS_DIR}/train_model.py "
+    "python scripts/train_model.py "
    "--config {config_file} "
    "--data {data_file} "
    "--model {model_file}"
)

# Prediction command template
predict_command = (
-    f"python {SCRIPTS_DIR}/predict_model.py "
+    "python scripts/predict_model.py "
    "--config {config_file} "
    "--model {model_file} "
    "--historic {historic_file} "
    "--future {future_file} "
    "--output {output_file}"
)

# Create shell model runner
runner: ShellModelRunner[{{ PROJECT_SLUG.replace('_', ' ').title().replace(' ', '') }}Config] = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
    model_format="pickle",
)
```

### 3. Example (`examples/ml_shell/main.py`)

**Update lines 40-58:**

```diff
train_command = (
-    f"python {SCRIPTS_DIR}/train_model.py --config {{config_file}} --data {{data_file}} --model {{model_file}}"
+    "python scripts/train_model.py --config {{config_file}} --data {{data_file}} --model {{model_file}}"
)

predict_command = (
-    f"python {SCRIPTS_DIR}/predict_model.py "
+    "python scripts/predict_model.py "
    "--config {config_file} "
    "--model {model_file} "
    "--historic {historic_file} "
    "--future {future_file} "
    "--output {output_file}"
)

runner: ShellModelRunner[DiseaseConfig] = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
    model_format="pickle",
)
```

**Add new file `examples/ml_shell/scripts/lib.py`:**
```python
"""Shared utilities for disease prediction model (demonstrates relative imports)."""

import pandas as pd

def validate_data(data: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that required columns exist in data."""
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def preprocess_features(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features with common transformations."""
    # Fill missing values
    data = data.fillna(0)
    # Add derived features
    data["temp_x_rainfall"] = data["mean_temperature"] * data["rainfall"]
    return data
```

**Update `examples/ml_shell/scripts/train_model.py`:**
```python
# Add at top:
from .lib import validate_data, preprocess_features  # Relative import!

# Use in training:
validate_data(data, ["rainfall", "mean_temperature", "humidity", "disease_cases"])
data = preprocess_features(data)
```

---

## Open Questions & Decisions

### 1. Config Format Documentation

**Question:** Should we support JSON config in addition to YAML?

**Decision:** No, keep YAML only. Just fix docs. YAML is:
- More human-readable
- Supports comments
- Standard for config files
- Already implemented

### 2. Absolute Path Support in Commands

**Question:** Should we still support absolute paths in command templates?

**Decision:** Yes, commands can use absolute paths if needed, but documentation will recommend relative paths for portability and consistency with the isolation pattern.

### 3. Ignore Patterns

**Question:** Should ignore patterns be configurable?

**Decision:** Not in initial implementation. Use sensible defaults (.venv, node_modules, etc.). Can add configurability later if needed.

### 4. Project Root Detection

**Question:** Use `inspect.currentframe()` or `Path.cwd()`?

**Decision:** Use `Path.cwd()` (current working directory). This is simpler and matches the standard usage pattern:
- Users run `fastapi dev main.py` from project directory
- Docker containers set `WORKDIR` to project root
- More straightforward than frame inspection

### 5. Cleanup on Failure

**Question:** Should temp directory be preserved on script failure for debugging?

**Decision:** Always cleanup in initial implementation. Future enhancement can add global debug mode or environment variable to preserve temp dirs (e.g., `CHAPKIT_DEBUG=1`).

---

## Future Enhancements (Out of Scope)

These are valuable but not part of initial refactor:

1. **Debug Mode:** Environment variable or global setting to preserve temp directories on failure
2. **Environment Variables:** `env` dict parameter for custom environment injection
3. **Configurable Ignore Patterns:** Allow users to specify additional files/dirs to exclude
4. **Streaming Output:** Real-time stdout/stderr monitoring
5. **Multiple Data Formats:** Parquet, Feather for large datasets
6. **Pluggable Serialization:** Support joblib, torch.save, etc.
7. **Resource Limits:** CPU/memory constraints
8. **Progress Callbacks:** Script can report progress percentage
9. **External Model Storage:** S3/Azure integration for large models

---

## Success Criteria

### Must Have
- [ ] Full project directory copying implemented
- [ ] Proper ignore patterns (.venv, node_modules, etc.)
- [ ] Relative imports work in Python/R/Julia scripts
- [ ] Error messages include full output (no truncation)
- [ ] SCRIPTS_DIR pattern removed from all examples
- [ ] All examples updated and working
- [ ] All tests passing with >95% coverage
- [ ] Documentation updated (YAML config, relative paths, no SCRIPTS_DIR)

### Nice to Have
- [ ] Output schema validation (sample_0 column check)
- [ ] Performance benchmarks (overhead of directory copying)
- [ ] Configurable ignore patterns

---

## References

### Related Files
- `src/chapkit/ml/runner.py` - Current implementation
- `docs/guides/ml-workflows.md` - ML workflows guide
- `examples/ml_shell/` - Reference example
- `src/chapkit/cli/templates/main_shell.py.jinja2` - CLI template
- `tests/test_ml_shell_runner.py` - Unit tests

### Related Issues
- Relative import support (this design)
- Config format documentation bug (fixed in this design)
- Timeout support (added in this design)

### Additional Context
- Deep dive analysis completed 2025-11-18
- Branch: `refactor/shell-runner-isolation`
- Breaking changes acceptable per user approval

---

## Next Steps

1. **Review this design document** with stakeholders
2. **Get approval** on API changes and migration strategy
3. **Begin Phase 1** implementation (core isolation)
4. **Iterate** based on feedback

---

**End of Design Document**
