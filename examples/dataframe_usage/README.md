# DataFrame Usage Examples

Comprehensive examples demonstrating how to use servicekit's `DataFrame` class for data interchange, transformation, and API integration.

## Prerequisites

Install servicekit:

```bash
uv add servicekit
```

For library-specific examples, install the required library:

```bash
# For pandas
uv add pandas

# For Polars
uv add polars

# For xarray
uv add xarray numpy

# For API integration
uv add fastapi uvicorn
```

## Examples

### Core Features (No Extra Dependencies)

#### CSV Example
```bash
python csv_example.py
```

Shows how to:
- Read and write CSV files
- Convert to/from CSV strings
- Use custom delimiters (TSV)
- Handle headers and no-header files
- Round-trip CSV data

#### Data Inspection Example
```bash
python inspection_example.py
```

Shows how to:
- Use DataFrame properties (shape, size, empty, etc.)
- View data with head() and tail()
- Use negative indexing (pandas-style)
- Random sampling with sample()
- Reproducible sampling with random_state

#### Column Operations Example
```bash
python column_operations_example.py
```

Shows how to:
- Select specific columns
- Drop unwanted columns
- Rename columns
- Chain operations together
- Handle errors gracefully

#### Validation Example
```bash
python validation_example.py
```

Shows how to:
- Validate DataFrame structure
- Infer column data types
- Detect null values
- Handle mixed type columns
- Generate data quality reports

#### API Integration Example
```bash
python api_integration_example.py
```

Shows how to:
- Upload CSV data via API
- Validate data quality
- Download data as CSV
- Transform data with operations
- Generate data statistics
- Sample random data

Then visit http://localhost:8000/docs for interactive API documentation.

#### Analytics Example
```bash
python analytics_example.py
```

Shows how to:
- Access columns with [] syntax
- Get unique values
- Count value occurrences
- Sort DataFrames
- Iterate over rows
- Perform analytics with iteration

#### JSON Example
```bash
python json_example.py
```

Shows how to:
- Create DataFrame from JSON
- Export to JSON (records and columns formats)
- Round-trip JSON conversion
- Process API responses
- Handle mixed data types

#### Advanced Features Example
```bash
python advanced_features_example.py
```

Shows how to:
- Filter rows with predicates
- Transform columns with apply()
- Add new columns
- Drop rows by index
- Remove duplicate rows
- Fill missing values
- Concatenate DataFrames
- Generate statistical summaries
- Group by and aggregate
- Chain operations in pipelines
- Detect missing data with isna/notna
- Clean data with dropna
- Compare DataFrames with equals
- Create independent copies
- Count unique values with nunique

### Library Integration Examples

#### Pandas Example
```bash
python pandas_example.py
```

Shows how to:
- Convert pandas DataFrame to servicekit DataFrame
- Convert servicekit DataFrame back to pandas

#### Polars Example
```bash
python polars_example.py
```

Shows how to:
- Convert Polars DataFrame to servicekit DataFrame
- Convert servicekit DataFrame back to Polars

#### xarray Example
```bash
python xarray_example.py
```

Shows how to:
- Convert xarray DataArray (2D) to servicekit DataFrame
- Convert to pandas for further processing

## DataFrame Class

The `DataFrame` class provides a universal interchange format for tabular data. It stores data in a simple column-oriented structure:

```python
from chapkit.data import DataFrame

df = DataFrame(
    columns=["name", "age"],
    data=[["Alice", 25], ["Bob", 30]]
)
```

### Feature Overview

#### Creating DataFrames
- `DataFrame.from_dict(data)` - From dictionary (no dependencies)
- `DataFrame.from_records(records)` - From list of dicts (no dependencies)
- `DataFrame.from_csv(path/csv_string)` - From CSV (no dependencies)
- `DataFrame.from_json(json_string)` - From JSON array (no dependencies)
- `DataFrame.from_pandas(df)` - From pandas (requires pandas)
- `DataFrame.from_polars(df)` - From polars (requires polars)
- `DataFrame.from_xarray(da)` - From xarray (requires xarray)

#### Exporting DataFrames
- `df.to_dict(orient)` - To dictionary (no dependencies)
- `df.to_csv(path)` - To CSV file/string (no dependencies)
- `df.to_json(orient)` - To JSON string (no dependencies)
- `df.to_pandas()` - To pandas (requires pandas)
- `df.to_polars()` - To polars (requires polars)

#### Properties
- `df.shape` - (rows, columns) tuple
- `len(df)` - Number of rows
- `df.size` - Total elements (rows × columns)
- `df.empty` - True if no rows or columns
- `df.ndim` - Always 2 (2-dimensional)

#### Data Inspection
- `df.head(n)` - First n rows (default 5)
- `df.tail(n)` - Last n rows (default 5)
- `df.sample(n/frac, random_state)` - Random sample
- `for row in df` - Iterate over rows as dicts

#### Column Operations
- `df["column"]` - Get column values as list
- `df[["col1", "col2"]]` - Select multiple columns as DataFrame
- `df.get_column(name)` - Get column values
- `df.select(columns)` - Keep only specified columns
- `df.drop(columns)` - Remove specified columns
- `df.rename(mapper)` - Rename columns

#### Analytics
- `df.unique(column)` - Get unique values
- `df.value_counts(column)` - Count value occurrences
- `df.sort(by, ascending)` - Sort by column

#### Row Operations
- `df.filter(predicate)` - Filter rows with function
- `df.apply(func, column)` - Transform column values
- `df.add_column(name, values)` - Add new column
- `df.drop_rows(indices)` - Drop rows by index
- `df.drop_duplicates(subset)` - Remove duplicate rows
- `df.fillna(value/dict)` - Fill None values
- `df.concat(other)` - Concatenate DataFrames vertically

#### Statistical Analysis
- `df.describe()` - Generate summary statistics
- `df.groupby(column)` - Group by column values
  - `.count()` - Count rows per group
  - `.sum(column)` - Sum numeric column
  - `.mean(column)` - Average numeric column
  - `.min(column)` / `.max(column)` - Min/max values

#### Validation
- `df.validate_structure()` - Validate DataFrame integrity
- `df.infer_types()` - Infer column data types
- `df.has_nulls()` - Check for None values per column

#### Missing Data Operations
- `df.isna()` - Return DataFrame of booleans showing None locations
- `df.notna()` - Return DataFrame of booleans showing non-None locations
- `df.dropna(axis, how)` - Remove rows or columns with None values
- `df.fillna(value/dict)` - Fill None values

#### DataFrame Utilities
- `df.equals(other)` - Check if two DataFrames are identical
- `df.deepcopy()` - Create deep copy of DataFrame
- `df.nunique(column)` - Count unique values in column

### Type Support

Inferred types from `infer_types()`:
- `"int"` - All integers
- `"float"` - All floats (or mix of int/float)
- `"str"` - All strings
- `"bool"` - All booleans
- `"null"` - All None values
- `"mixed"` - Multiple different types

Each method will provide a helpful error message if a required library is not installed.

## Additional Resources

- See [DataFrame Guide](../../docs/guides/dataframe.md) for comprehensive documentation
- Visit the [examples directory](../) for more servicekit examples
