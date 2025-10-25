# Design Document: Migration from pandas to servicekit.DataFrame

## Overview

This document outlines the plan to migrate `chapkit.ml.runner` from using pandas DataFrames directly to using servicekit's framework-agnostic DataFrame abstraction.

## Motivation

### Current State
- `runner.py` imports pandas directly and uses `pd.DataFrame` in type signatures
- Users of chapkit are forced to install pandas even if they prefer polars or other dataframe libraries
- Creates tight coupling between chapkit and pandas

### Desired State
- Use `servicekit.data.DataFrame` as the universal interchange format
- Users can choose their preferred dataframe library (pandas, polars, xarray, etc.)
- Chapkit remains framework-agnostic at the API boundary
- Smaller dependency footprint (pandas becomes optional)

## ServiceKit DataFrame Overview

ServiceKit provides a universal `DataFrame` class that:
- Acts as a framework-agnostic interchange format for tabular data
- Supports bidirectional conversion: `from_pandas()`, `to_pandas()`, `from_polars()`, `to_polars()`, `from_xarray()`
- Has built-in CSV read/write: `from_csv()`, `to_csv()`
- Uses lazy imports - pandas/polars only loaded when conversion methods are called
- Inherits from Pydantic BaseModel for validation and serialization
- Provides rich operations (filter, groupby, merge, pivot, etc.) without external dependencies

**Location:** `servicekit.data.DataFrame`

## Proposed Changes

### 1. File: `src/chapkit/ml/runner.py`

#### Import Changes
```python
# Remove
import pandas as pd

# Add
from servicekit.data import DataFrame
```

#### Type Signature Updates

**Type Aliases (lines 22-25):**
```python
# Before
type TrainFunction[ConfigT] = Callable[[ConfigT, pd.DataFrame, FeatureCollection | None], Awaitable[Any]]
type PredictFunction[ConfigT] = Callable[
    [ConfigT, Any, pd.DataFrame, pd.DataFrame, FeatureCollection | None], Awaitable[pd.DataFrame]
]

# After
type TrainFunction[ConfigT] = Callable[[ConfigT, DataFrame, FeatureCollection | None], Awaitable[Any]]
type PredictFunction[ConfigT] = Callable[
    [ConfigT, Any, DataFrame, DataFrame, FeatureCollection | None], Awaitable[DataFrame]
]
```

**BaseModelRunner (lines 42-61):**
```python
# Before
async def on_train(
    self,
    config: BaseConfig,
    data: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:

async def on_predict(
    self,
    config: BaseConfig,
    model: Any,
    historic: pd.DataFrame,
    future: pd.DataFrame,
    geo: FeatureCollection | None = None,
) -> pd.DataFrame:

# After
async def on_train(
    self,
    config: BaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:

async def on_predict(
    self,
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
```

**FunctionalModelRunner (lines 76-94):**
Same parameter and return type changes as BaseModelRunner.

**ShellModelRunner (lines 111-260):**
Same parameter and return type changes, plus implementation changes below.

#### Implementation Changes in ShellModelRunner

**`on_train()` method (line 127):**
```python
# Before
data.to_csv(data_file, index=False)

# After
data.to_csv(data_file)
```

**`on_predict()` method (lines 203-208, 253):**
```python
# Before (line 203)
historic.to_csv(historic_file, index=False)

# After
historic.to_csv(historic_file)

# Before (line 208)
future.to_csv(future_file, index=False)

# After
future.to_csv(future_file)

# Before (line 253)
predictions = pd.read_csv(output_file)

# After
predictions = DataFrame.from_csv(output_file)
```

### 2. Dependency Changes

**Remove pandas as direct dependency:**
```bash
uv remove pandas
```

**Note:** pandas becomes an optional dependency that users can install if they need pandas-specific features.

### 3. Files Requiring Updates

**Code:**
- `src/chapkit/ml/runner.py` - core implementation changes

**Tests:**
- Any tests in `tests/` that use `runner.py` will need to:
  - Import `from servicekit.data import DataFrame`
  - Convert test data: `DataFrame.from_pandas(test_df)` or create directly
  - Convert assertions: `result.to_pandas()` if comparing with pandas

**Documentation:**
- README examples showing DataFrame usage
- Migration guide for existing users
- Examples showing conversion patterns

**Examples:**
- Any example code in `examples/` using ML runners

## Migration Path for Users

Users currently using chapkit will need to adapt their code:

### Before
```python
import pandas as pd
from chapkit.ml import BaseModelRunner

class MyRunner(BaseModelRunner):
    async def on_train(self, config, data: pd.DataFrame, geo=None):
        # work with pandas directly
        return model

    async def on_predict(self, config, model, historic: pd.DataFrame,
                        future: pd.DataFrame, geo=None) -> pd.DataFrame:
        # work with pandas directly
        return predictions
```

### After (Option 1: Convert at boundaries)
```python
import pandas as pd
from servicekit.data import DataFrame
from chapkit.ml import BaseModelRunner

class MyRunner(BaseModelRunner):
    async def on_train(self, config, data: DataFrame, geo=None):
        # Convert to pandas if needed
        df = data.to_pandas()
        # work with pandas
        return model

    async def on_predict(self, config, model, historic: DataFrame,
                        future: DataFrame, geo=None) -> DataFrame:
        # Convert to pandas if needed
        hist_df = historic.to_pandas()
        future_df = future.to_pandas()
        # work with pandas
        pred_df = ...
        # Convert back
        return DataFrame.from_pandas(pred_df)
```

### After (Option 2: Use servicekit DataFrame directly)
```python
from servicekit.data import DataFrame
from chapkit.ml import BaseModelRunner

class MyRunner(BaseModelRunner):
    async def on_train(self, config, data: DataFrame, geo=None):
        # Work with DataFrame directly (no pandas needed)
        filtered = data.filter(lambda row: row[0] > 0)
        return model

    async def on_predict(self, config, model, historic: DataFrame,
                        future: DataFrame, geo=None) -> DataFrame:
        # Work with DataFrame operations
        return predictions
```

### After (Option 3: Use polars instead)
```python
import polars as pl
from servicekit.data import DataFrame
from chapkit.ml import BaseModelRunner

class MyRunner(BaseModelRunner):
    async def on_train(self, config, data: DataFrame, geo=None):
        # Convert to polars
        df = data.to_polars()
        # work with polars
        return model

    async def on_predict(self, config, model, historic: DataFrame,
                        future: DataFrame, geo=None) -> DataFrame:
        # Convert to polars
        hist_df = historic.to_polars()
        future_df = future.to_polars()
        # work with polars
        pred_df = ...
        # Convert back
        return DataFrame.from_polars(pred_df)
```

## Breaking Changes

This is a **breaking change** that will require users to update their code:

1. Type signatures change from `pd.DataFrame` to `DataFrame`
2. Users must explicitly convert if they want to use pandas/polars
3. pandas is no longer a required dependency

## Benefits

1. **Framework Agnostic:** Users choose pandas, polars, or other libraries
2. **Smaller Dependencies:** pandas becomes optional (pandas is large: ~50MB)
3. **Future Proof:** Easy to support new dataframe libraries as they emerge
4. **API Clarity:** Clear interchange format at API boundaries
5. **Consistency:** Aligns with servicekit's design philosophy

## Risks and Mitigations

### Risk: Breaking existing user code
**Mitigation:**
- Clear migration guide with examples
- Document all conversion patterns
- Consider deprecation period with warnings

### Risk: Performance overhead from conversions
**Mitigation:**
- Conversions are lazy - only happen when methods called
- Users can work with DataFrame directly for simple cases
- Servicekit DataFrame is optimized for common operations

### Risk: Missing pandas features
**Mitigation:**
- Users can still convert to pandas for advanced features
- Servicekit DataFrame provides rich operations for common cases
- Document which operations require pandas conversion

## Implementation Steps

1. Create feature branch: `feat/dataframe-migration`
2. Update `runner.py` with type and implementation changes
3. Update all tests to use DataFrame
4. Update documentation and examples
5. Add migration guide
6. Run full test suite
7. Create PR for review

## Testing Strategy

1. **Unit Tests:** Update existing runner tests to use DataFrame
2. **Integration Tests:** Test conversion workflows (pandas ↔ DataFrame ↔ polars)
3. **Shell Runner Tests:** Verify CSV round-trip works correctly
4. **Type Checking:** Ensure pyright passes with new types

## Documentation Updates

1. **README:** Update examples to use DataFrame
2. **Migration Guide:** Create guide for existing users
3. **API Documentation:** Update docstrings
4. **Examples:** Add examples showing pandas/polars conversions

## Timeline

1. Design review and approval: 1 day
2. Implementation: 1-2 days
3. Testing and documentation: 1 day
4. Review and iteration: 1-2 days

**Total: 4-6 days**

## Open Questions

None - design decisions confirmed with user:
- ✅ Use strict DataFrame types (no auto-conversion)
- ✅ Keep CSV format for ShellModelRunner
- ✅ Create design doc + branch + PR before implementation

## References

- ServiceKit DataFrame: `/Users/morten/dev/chap-sdk/servicekit/src/servicekit/data/dataframe.py`
- ServiceKit DataFrame Docs: `/Users/morten/dev/chap-sdk/servicekit/docs/guides/dataframe.md`
- Current Implementation: `/Users/morten/dev/chap-sdk/chapkit/src/chapkit/ml/runner.py`
