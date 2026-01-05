# DataFrame Data Interchange

Servicekit provides a universal DataFrame class for seamless data interchange between different data libraries (pandas, polars, xarray) and file formats (CSV, Parquet). It's designed to be lightweight, framework-agnostic, and easy to use in API services.

## Quick Start

### Basic Usage

```python
from chapkit.data import DataFrame

# Create from dictionary
df = DataFrame.from_dict({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "SF", "LA"]
})

# Inspect data
print(df.shape)  # (3, 3)
print(df.head(2))

# Convert to other libraries
pandas_df = df.to_pandas()
polars_df = df.to_polars()
```

### In FastAPI Services

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from chapkit.data import DataFrame

app = FastAPI()

@app.post("/data/$upload")
async def upload_csv(file: UploadFile):
    """Accept CSV upload and process."""
    content = await file.read()
    df = DataFrame.from_csv(csv_string=content.decode())

    # Process data
    df = df.select(["name", "age"]).head(100)

    return {"rows": df.shape[0], "columns": df.columns}

@app.get("/data/$download")
async def download_csv():
    """Export data as CSV."""
    df = get_data()  # Your data source
    csv_data = df.to_csv()
    return Response(content=csv_data, media_type="text/csv")
```

## Core Concepts

### Data Structure

DataFrame uses a simple columnar structure:

```python
df = DataFrame(
    columns=["name", "age"],
    data=[
        ["Alice", 25],
        ["Bob", 30]
    ]
)
```

- **columns**: List of column names (strings)
- **data**: List of rows, where each row is a list of values
- **Type-agnostic**: Values can be any Python type

### Design Principles

- **Lightweight**: No required dependencies beyond Pydantic
- **Framework-agnostic**: Works with any Python environment
- **Lazy imports**: Optional libraries loaded only when needed
- **Immutable**: Methods return new DataFrames (no in-place modification)
- **API consistency**: All methods follow `from_X()` / `to_X()` pattern

## Creating DataFrames

### From Dictionary

```python
# Column-oriented (dict of lists)
df = DataFrame.from_dict({
    "name": ["Alice", "Bob"],
    "age": [25, 30]
})
```

### From Records

```python
# Row-oriented (list of dicts)
df = DataFrame.from_records([
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30}
])
```

### From CSV

```python
# From file (with automatic type inference)
df = DataFrame.from_csv("data.csv")

# From string - values are automatically converted to int, float, bool, or None
csv_string = "name,age,active\nAlice,25,true\nBob,30,false"
df = DataFrame.from_csv(csv_string=csv_string)
# Result: [["Alice", 25, True], ["Bob", 30, False]]

# Custom delimiter
df = DataFrame.from_csv("data.tsv", delimiter="\t")

# Without header
df = DataFrame.from_csv("data.csv", has_header=False)
# Generates columns: col_0, col_1, ...

# Disable type inference (keep all values as strings)
df = DataFrame.from_csv("data.csv", infer_types=False)
```

### From Other Libraries

```python
# From pandas
import pandas as pd
pandas_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
df = DataFrame.from_pandas(pandas_df)

# From polars
import polars as pl
polars_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
df = DataFrame.from_polars(polars_df)

# From xarray (2D only)
import xarray as xr
data_array = xr.DataArray([[1, 2], [3, 4]])
df = DataFrame.from_xarray(data_array)
```

## Exporting DataFrames

### To CSV

```python
# To file
df.to_csv("output.csv")

# To string
csv_string = df.to_csv()

# Without header
df.to_csv("output.csv", include_header=False)

# Custom delimiter
df.to_csv("output.tsv", delimiter="\t")
```

### To Dictionary

```python
# As dict of lists (default)
data = df.to_dict(orient="list")
# {"name": ["Alice", "Bob"], "age": [25, 30]}

# As list of records
data = df.to_dict(orient="records")
# [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]

# As dict of dicts
data = df.to_dict(orient="dict")
# {"name": {0: "Alice", 1: "Bob"}, "age": {0: 25, 1: 30}}
```

### To Other Libraries

```python
# To pandas
pandas_df = df.to_pandas()

# To polars
polars_df = df.to_polars()
```

## Data Inspection

### Properties

```python
# Shape (rows, columns)
print(df.shape)  # (100, 5)

# Number of rows
print(df.shape[0])  # 100
print(len(df))  # 100 - can also use len()

# Number of columns
print(df.shape[1])  # 5

# Check if empty
print(df.empty)  # False

# Number of dimensions (always 2)
print(df.ndim)  # 2

# Total elements
print(df.size)  # 500 (100 * 5)
```

### Viewing Data

```python
# First 5 rows (default)
df.head()

# First n rows
df.head(10)

# Last 5 rows (default)
df.tail()

# Last n rows
df.tail(10)

# Negative indexing (pandas-style)
df.head(-3)  # All except last 3 rows
df.tail(-3)  # All except first 3 rows
```

### Random Sampling

```python
# Sample n rows
sample = df.sample(n=100)

# Sample fraction
sample = df.sample(frac=0.1)  # 10% of rows

# Reproducible sampling
sample = df.sample(n=50, random_state=42)
```

### Iteration

```python
# Iterate over rows as dictionaries
for row in df:
    print(row)  # {'name': 'Alice', 'age': 25}

# Get number of rows with len()
num_rows = len(df)

# Use in list comprehensions
names = [row['name'] for row in df]
```

## Column Operations

### Accessing Columns

```python
# Get column values as list
ages = df.get_column("age")  # [25, 30, 35]
ages = df["age"]  # Same using [] syntax

# Select multiple columns as DataFrame
df_subset = df[["name", "age"]]
df_subset = df.select(["name", "age"])  # Equivalent
```

### Selecting Columns

```python
# Select specific columns
df_subset = df.select(["name", "age"])

# Single column
df_single = df.select(["age"])
```

### Dropping Columns

```python
# Drop specific columns
df_clean = df.drop(["temp_column", "debug_field"])

# Drop multiple
df_clean = df.drop(["col1", "col2", "col3"])
```

### Renaming Columns

```python
# Rename specific columns
df_renamed = df.rename({
    "old_name": "new_name",
    "user_id": "id"
})

# Partial rename (other columns unchanged)
df_renamed = df.rename({"age": "years"})

# Alternative: use rename_columns() (same behavior)
df_renamed = df.rename_columns({"age": "years"})
```

The `rename_columns()` method is an alias for `rename()`, provided for improved code readability and discoverability.

## Validation and Type Inference

### Structure Validation

```python
# Validate DataFrame structure
try:
    df.validate_structure()
    print("DataFrame is valid")
except ValueError as e:
    print(f"Validation failed: {e}")

# Checks performed:
# - All rows have same length as columns
# - Column names are unique
# - No empty column names
```

### Type Inference

```python
# Infer column data types
types = df.infer_types()
print(types)
# {"age": "int", "name": "str", "score": "float"}

# Supported types:
# - "int": All integers
# - "float": All floats (or mix of int/float)
# - "str": All strings
# - "bool": All booleans
# - "null": All None values
# - "mixed": Multiple different types
```

### CSV Type Inference

When loading CSV files with `from_csv()`, values are automatically inferred and converted:

```python
csv_string = """name,age,score,active
Alice,25,95.5,true
Bob,30,87.0,false
,35,,yes"""

df = DataFrame.from_csv(csv_string=csv_string)

# Types are automatically inferred:
# - name: str (text values)
# - age: int (integer values)
# - score: float (decimal values)
# - active: bool (true/false/yes/no)

# Empty strings become None
print(df.data[2])  # [None, 35, None, True]
```

**Supported Type Conversions:**

| CSV Value | Python Type | Example |
|-----------|-------------|---------|
| Empty/whitespace | `None` | `""`, `"   "` |
| `true`, `false`, `yes`, `no` (case-insensitive) | `bool` | `"true"` -> `True` |
| Integer strings (no decimal) | `int` | `"42"` -> `42` |
| Decimal/scientific notation | `float` | `"3.14"`, `"1e10"` |
| Other text | `str` | `"hello"` -> `"hello"` |

**Type Promotion Rules:**
- Columns with mixed int/float values become `float`
- Columns with mixed types (e.g., int and str) remain `str`
- `None` values are compatible with any type

**Disabling Type Inference:**

```python
# Keep all values as strings (original behavior)
df = DataFrame.from_csv("data.csv", infer_types=False)
```

### Null Detection

```python
# Check for None values per column
nulls = df.has_nulls()
print(nulls)
# {"age": False, "email": True, "phone": True}

# Use for data quality checks
if any(nulls.values()):
    print("Warning: DataFrame contains null values")
```

## Sorting and Analytics

### Sorting

```python
# Sort by column (ascending)
df_sorted = df.sort("age")

# Sort descending
df_sorted = df.sort("score", ascending=False)

# None values always sort to the end
df_sorted = df.sort("nullable_column")
```

### Unique Values

```python
# Get unique values from a column
categories = df.unique("category")
# ['A', 'B', 'C'] - preserves order of first appearance

# Count unique values
num_unique = len(df.unique("category"))
```

### Value Counts

```python
# Count occurrences of each value
counts = df.value_counts("category")
# {'A': 3, 'B': 2, 'C': 1}

# Find most common value
most_common = max(counts, key=counts.get)

# Get distribution
total = len(df)
distribution = {k: v/total for k, v in counts.items()}
```

## JSON Support

### Creating from JSON

```python
# From JSON array of objects
json_data = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
df = DataFrame.from_json(json_data)

# From API response
import requests
response = requests.get("https://api.example.com/data")
df = DataFrame.from_json(response.text)
```

### Exporting to JSON

```python
# As array of objects (records format)
json_str = df.to_json(orient="records")
# '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'

# As object with arrays (columns format)
json_str = df.to_json(orient="columns")
# '{"name": ["Alice", "Bob"], "age": [25, 30]}'

# For API responses
from fastapi import Response

@app.get("/data")
async def get_data():
    df = get_dataframe()
    return Response(content=df.to_json(), media_type="application/json")
```

## Row Filtering and Transformation

### Filtering Rows

```python
# Filter with predicate function
adults = df.filter(lambda row: row['age'] >= 18)

# Multiple conditions
active_adults = df.filter(lambda row: row['age'] >= 18 and row['active'])

# Complex filtering
high_scorers = df.filter(lambda row: row['score'] > 90 or (row['score'] > 80 and row['bonus_eligible']))
```

### Applying Transformations

```python
# Transform column values
df_upper = df.apply(str.upper, 'name')

# Apply custom function
df_doubled = df.apply(lambda x: x * 2, 'price')

# Apply method
df_rounded = df.apply(round, 'price')
```

### Adding Columns

```python
# Add new column
total = [x + y for x, y in zip(df['price'], df['tax'])]
df_with_total = df.add_column('total', total)

# Chain column additions
df_enhanced = (
    df.add_column('total', totals)
      .add_column('formatted', formatted_values)
)
```

## Row Operations

### Dropping Rows

```python
# Drop rows by index
df_cleaned = df.drop_rows([0, 5, 10])

# Drop first row
df_no_header = df.drop_rows([0])

# Drop multiple rows
invalid_indices = [i for i, row in enumerate(df) if row['status'] == 'invalid']
df_valid = df.drop_rows(invalid_indices)
```

### Removing Duplicates

```python
# Remove duplicate rows (all columns)
df_unique = df.drop_duplicates()

# Remove duplicates by specific columns
df_unique_users = df.drop_duplicates(subset=['user_id'])

# Remove duplicates considering multiple columns
df_unique_pairs = df.drop_duplicates(subset=['category', 'product'])
```

### Filling Missing Values

```python
# Fill all None with single value
df_filled = df.fillna(0)

# Column-specific fill values
df_filled = df.fillna({
    'age': 0,
    'name': 'Unknown',
    'score': -1
})

# Partial filling (only specified columns)
df_partial = df.fillna({'age': 0})  # Other columns keep None
```

### Concatenating DataFrames

```python
# Stack DataFrames vertically
df1 = DataFrame.from_dict({'name': ['Alice'], 'age': [25]})
df2 = DataFrame.from_dict({'name': ['Bob'], 'age': [30]})
combined = df1.concat(df2)

# Combine multiple DataFrames
dfs = [df1, df2, df3]
result = dfs[0]
for df in dfs[1:]:
    result = result.concat(df)
```

## Reshaping Operations

The `melt()` method transforms DataFrames from wide format (many columns) to long format (fewer columns, more rows). This is essential for preparing data for analysis, visualization, or API interchange.

### Understanding Wide vs Long Format

**Wide Format:** Multiple measurement columns
```python
# Example: Student grades across subjects
df_wide = DataFrame.from_dict({
    'name': ['Alice', 'Bob', 'Charlie'],
    'math': [90, 78, 95],
    'science': [85, 92, 89],
    'history': [88, 81, 93]
})
# name    | math | science | history
# Alice   | 90   | 85      | 88
# Bob     | 78   | 92      | 81
# Charlie | 95   | 89      | 93
```

**Long Format:** Single measurement column with category identifier
```python
# Melt to long format
df_long = df_wide.melt(
    id_vars=['name'],
    value_vars=['math', 'science', 'history'],
    var_name='subject',
    value_name='score'
)
# name    | subject | score
# Alice   | math    | 90
# Alice   | science | 85
# Alice   | history | 88
# Bob     | math    | 78
# ...
```

### Basic melt() Usage

```python
from chapkit.data import DataFrame

# Create wide format data
df = DataFrame.from_dict({
    'product': ['Widget', 'Gadget'],
    'q1_sales': [1000, 800],
    'q2_sales': [1100, 850],
    'q3_sales': [1200, 900]
})

# Melt to long format
melted = df.melt(
    id_vars=['product'],           # Columns to keep as identifiers
    value_vars=['q1_sales', 'q2_sales', 'q3_sales'],  # Columns to unpivot
    var_name='quarter',            # Name for variable column
    value_name='sales'             # Name for value column
)

# Result:
# product | quarter   | sales
# Widget  | q1_sales  | 1000
# Widget  | q2_sales  | 1100
# Widget  | q3_sales  | 1200
# Gadget  | q1_sales  | 800
# Gadget  | q2_sales  | 850
# Gadget  | q3_sales  | 900
```

### melt() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_vars` | `list[str] \| None` | `None` | Columns to keep as identifiers (not melted) |
| `value_vars` | `list[str] \| None` | `None` | Columns to unpivot (if None, uses all non-id columns) |
| `var_name` | `str` | `"variable"` | Name for the column containing former column names |
| `value_name` | `str` | `"value"` | Name for the column containing values |

### Common Use Cases

#### Survey/Questionnaire Data

```python
# Wide format: each question is a column
survey = DataFrame.from_dict({
    'respondent_id': [1, 2, 3],
    'age': [25, 30, 35],
    'q1_rating': [5, 4, 5],
    'q2_rating': [4, 4, 5],
    'q3_rating': [5, 3, 5]
})

# Melt to long format for analysis
responses = survey.melt(
    id_vars=['respondent_id', 'age'],
    value_vars=['q1_rating', 'q2_rating', 'q3_rating'],
    var_name='question',
    value_name='rating'
)

# Now easy to analyze: average rating per question
avg_by_question = responses.groupby('question').mean('rating')
```

#### Time Series Data

```python
# Wide format: each month is a column
sales = DataFrame.from_dict({
    'region': ['North', 'South', 'East'],
    'product': ['Widget', 'Widget', 'Widget'],
    'jan': [1000, 1200, 900],
    'feb': [1100, 1300, 950],
    'mar': [1200, 1400, 1000]
})

# Melt to time series format
time_series = sales.melt(
    id_vars=['region', 'product'],
    value_vars=['jan', 'feb', 'mar'],
    var_name='month',
    value_name='sales'
)

# Now can analyze trends over time
total_by_month = time_series.groupby('month').sum('sales')
```

#### API Data Standardization

```python
# API returns different metrics as columns
sensor_data = DataFrame.from_dict({
    'sensor_id': ['s1', 's2'],
    'location': ['room_a', 'room_b'],
    'temp_c': [22.5, 23.1],
    'humidity_pct': [45, 48],
    'pressure_kpa': [101.3, 101.2]
})

# Standardize to key-value format
metrics = sensor_data.melt(
    id_vars=['sensor_id', 'location'],
    value_vars=['temp_c', 'humidity_pct', 'pressure_kpa'],
    var_name='metric_type',
    value_name='metric_value'
)

# Easier to store/process uniform metric records
```

### Advanced Patterns

#### Combining melt() with groupby()

```python
# Wide format sales data
df = DataFrame.from_dict({
    'region': ['North', 'North', 'South', 'South'],
    'product': ['Widget', 'Gadget', 'Widget', 'Gadget'],
    'q1': [1000, 800, 1200, 900],
    'q2': [1100, 850, 1300, 950]
})

# Melt then aggregate
melted = df.melt(
    id_vars=['region', 'product'],
    value_vars=['q1', 'q2'],
    var_name='quarter',
    value_name='sales'
)

# Total sales by region
region_totals = melted.groupby('region').sum('sales')

# Average sales by product
product_avg = melted.groupby('product').mean('sales')
```

#### Filtering After melt()

```python
# Melt then filter for specific conditions
melted = df.melt(id_vars=['product'], value_vars=['q1', 'q2', 'q3', 'q4'])

# Only keep quarters with sales > 1000
high_sales = melted.filter(lambda row: row['value'] > 1000)

# Group filtered results
summary = high_sales.groupby('product').count()
```

### melt() Design Notes

- **Stdlib only:** No external dependencies, pure Python implementation
- **Immutable:** Returns new DataFrame, original unchanged
- **None values:** Preserved during transformation
- **Column order:** Results maintain row order and value_vars order
- **Validation:** Raises `KeyError` for non-existent columns, `ValueError` for name conflicts

### pivot() - Long to Wide Format

The `pivot()` method is the inverse of `melt()` - it transforms data from long format to wide format by spreading row values into columns.

#### Basic pivot() Usage

```python
from chapkit.data import DataFrame

# Long format data
df_long = DataFrame.from_dict({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'metric': ['sales', 'profit', 'sales', 'profit'],
    'value': [1000, 200, 1100, 220]
})

# Pivot to wide format
df_wide = df_long.pivot(index='date', columns='metric', values='value')

# Result:
# date    | profit | sales
# 2024-01 | 200    | 1000
# 2024-02 | 220    | 1100
```

#### pivot() Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `index` | `str` | Column to use as row index in result |
| `columns` | `str` | Column whose unique values become new columns |
| `values` | `str` | Column containing values to fill pivoted table |

#### Pivot Use Cases

**Report Generation:**
```python
# Student grades in long format
df_long = DataFrame.from_dict({
    'student': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob'],
    'subject': ['math', 'science', 'history', 'math', 'science', 'history'],
    'score': [90, 85, 88, 78, 92, 81]
})

# Pivot for report card
report = df_long.pivot(index='student', columns='subject', values='score')
# student | history | math | science
# Alice   | 88      | 90   | 85
# Bob     | 81      | 78   | 92
```

**Time Series Restructuring:**
```python
# API returns time series in long format
time_series = DataFrame.from_dict({
    'week': [1, 1, 2, 2],
    'day': ['mon', 'tue', 'mon', 'tue'],
    'hours': [8, 7, 9, 8]
})

# Pivot for weekly view
weekly = time_series.pivot(index='week', columns='day', values='hours')
# week | mon | tue
# 1    | 8   | 7
# 2    | 9   | 8
```

#### Combining melt() and pivot()

These operations are inverses - you can round-trip data:

```python
# Start with wide format
df_wide = DataFrame.from_dict({
    'id': [1, 2],
    'a': [10, 20],
    'b': [30, 40]
})

# Melt to long format
df_long = df_wide.melt(id_vars=['id'], value_vars=['a', 'b'])
# id | variable | value
# 1  | a        | 10
# 1  | b        | 30
# 2  | a        | 20
# 2  | b        | 40

# Pivot back to wide format
df_restored = df_long.pivot(index='id', columns='variable', values='value')
# id | a  | b
# 1  | 10 | 30
# 2  | 20 | 40
```

#### pivot() Design Notes

- **Duplicate detection:** Raises `ValueError` if index/column combinations are duplicated
- **Sparse data:** Missing combinations filled with `None`
- **Column ordering:** Result columns are sorted alphabetically
- **Validation:** Raises `KeyError` for non-existent column names
- **Stdlib only:** No external dependencies

### transpose() - Swap Rows and Columns

The `transpose()` method swaps rows and columns (matrix transpose). The first column becomes column headers, and column names become the first column values.

#### Basic transpose() Usage

```python
from chapkit.data import DataFrame

# Metrics as rows
df = DataFrame.from_dict({
    'metric': ['revenue', 'profit', 'growth'],
    '2023': [1000, 200, 0.10],
    '2024': [1200, 250, 0.20]
})

# Transpose to have metrics as columns
df_t = df.transpose()

# Result:
# index | revenue | profit | growth
# 2023  | 1000    | 200    | 0.10
# 2024  | 1200    | 250    | 0.20
```

#### Transpose Use Cases

**Report Formatting:**
```python
# API returns quarterly metrics by region
data = DataFrame.from_dict({
    'region': ['North', 'South', 'East', 'West'],
    'Q1': [100, 200, 150, 180],
    'Q2': [110, 210, 160, 190],
    'Q3': [120, 220, 170, 200],
    'Q4': [130, 230, 180, 210]
})

# Transpose for quarterly view
quarterly = data.transpose()
# index | North | South | East | West
# Q1    | 100   | 200   | 150  | 180
# Q2    | 110   | 210   | 160  | 190
# ...
```

**Rotating Time Series:**
```python
# Monthly sales by product (wide format)
monthly = DataFrame.from_dict({
    'product': ['Widget', 'Gadget', 'Tool'],
    'jan': [100, 80, 60],
    'feb': [110, 85, 65],
    'mar': [120, 90, 70]
})

# Transpose to have products as columns
by_month = monthly.transpose()
# index | Widget | Gadget | Tool
# jan   | 100    | 80     | 60
# feb   | 110    | 85     | 65
# mar   | 120    | 90     | 70
```

**Preparing for Visualization:**
```python
# Data with entities as rows
entities = DataFrame.from_dict({
    'entity': ['Team A', 'Team B', 'Team C'],
    'score': [85, 92, 78],
    'rank': [2, 1, 3]
})

# Transpose for chart libraries expecting columns as series
chart_data = entities.transpose()
# index | Team A | Team B | Team C
# score | 85     | 92     | 78
# rank  | 2      | 1      | 3
```

#### Combining with Other Operations

**Transpose + Filter + Transpose:**
```python
# Start with wide format
df = DataFrame.from_dict({
    'id': [1, 2, 3],
    'metric_a': [100, 200, 300],
    'metric_b': [10, 20, 30],
    'metric_c': [5, 15, 25]
})

# Transpose to work with metrics as rows
df_t = df.transpose()

# Filter for specific IDs (now columns)
# Then transpose back
filtered = df_t.filter(lambda row: row['1'] > 50)
result = filtered.transpose()
```

**Transpose for Aggregation:**
```python
# Quarterly data by product
df = DataFrame.from_dict({
    'product': ['Widget', 'Gadget'],
    'Q1': [100, 80],
    'Q2': [110, 85],
    'Q3': [120, 90],
    'Q4': [130, 95]
})

# Transpose and melt for time series analysis
df_t = df.transpose()
# Now can use melt() or groupby() on transposed data
```

#### transpose() Design Notes

- **First column as index:** First column values become new column names (converted to strings)
- **Column names as data:** Original column names (except first) become first column values
- **Immutable:** Returns new DataFrame, original unchanged
- **None values:** Preserved during transformation
- **Empty handling:** Empty DataFrames return empty result
- **Round-trip:** Transposing twice restores original structure (with potential column name changes)
- **Stdlib only:** No external dependencies

## Combining DataFrames

### merge() - Database-Style Joins

The `merge()` method combines two DataFrames using SQL-like join operations (inner, left, right, outer).

#### Basic merge() Usage

```python
from chapkit.data import DataFrame

# Users data
users = DataFrame.from_dict({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

# Orders data
orders = DataFrame.from_dict({
    'user_id': [1, 1, 2],
    'amount': [100, 150, 200]
})

# Join to get user names with orders
result = orders.merge(users, on='user_id', how='left')
# user_id | amount | name
# 1       | 100    | Alice
# 1       | 150    | Alice
# 2       | 200    | Bob
```

#### merge() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `DataFrame` | - | DataFrame to merge with |
| `on` | `str \| list[str]` | `None` | Column(s) to join on (must exist in both) |
| `how` | `"inner" \| "left" \| "right" \| "outer"` | `"inner"` | Type of join to perform |
| `left_on` | `str \| list[str]` | `None` | Column(s) from left DataFrame to join on |
| `right_on` | `str \| list[str]` | `None` | Column(s) from right DataFrame to join on |
| `suffixes` | `tuple[str, str]` | `("_x", "_y")` | Suffixes for overlapping column names |

#### Join Types

**Inner Join (default):** Only rows with matching keys in both DataFrames
```python
left = DataFrame.from_dict({'key': [1, 2, 3], 'left_val': ['a', 'b', 'c']})
right = DataFrame.from_dict({'key': [1, 2, 4], 'right_val': ['x', 'y', 'z']})

result = left.merge(right, on='key', how='inner')
# key | left_val | right_val
# 1   | a        | x
# 2   | b        | y
```

**Left Join:** All rows from left, matched rows from right (None if no match)
```python
result = left.merge(right, on='key', how='left')
# key | left_val | right_val
# 1   | a        | x
# 2   | b        | y
# 3   | c        | None
```

**Right Join:** All rows from right, matched rows from left (None if no match)
```python
result = left.merge(right, on='key', how='right')
# key | left_val | right_val
# 1   | a        | x
# 2   | b        | y
# 4   | None     | z
```

**Outer Join:** All rows from both DataFrames
```python
result = left.merge(right, on='key', how='outer')
# key | left_val | right_val
# 1   | a        | x
# 2   | b        | y
# 3   | c        | None
# 4   | None     | z
```

#### Common Merge Patterns

**Enriching Data from Another Service:**
```python
# Orders from one service
orders = DataFrame.from_dict({
    'order_id': [101, 102, 103],
    'user_id': [1, 2, 1],
    'amount': [50, 75, 100]
})

# User details from another service
users = DataFrame.from_dict({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'tier': ['gold', 'silver', 'gold']
})

# Enrich orders with user data
enriched = orders.merge(users, on='user_id', how='left')
# order_id | user_id | amount | name  | tier
# 101      | 1       | 50     | Alice | gold
# 102      | 2       | 75     | Bob   | silver
# 103      | 1       | 100    | Alice | gold
```

**Different Column Names:**
```python
products = DataFrame.from_dict({
    'product_id': [1, 2, 3],
    'product_name': ['Widget', 'Gadget', 'Tool']
})

sales = DataFrame.from_dict({
    'item_id': [1, 2, 1],
    'quantity': [10, 5, 8]
})

# Join on different column names
result = sales.merge(
    products,
    left_on='item_id',
    right_on='product_id',
    how='left'
)
```

**Multiple Join Keys:**
```python
# Join on multiple columns for composite keys
result = df1.merge(df2, on=['region', 'product'], how='inner')
```

**Handling Column Conflicts:**
```python
# Both DataFrames have 'value' column
left = DataFrame.from_dict({'key': [1, 2], 'value': [10, 20]})
right = DataFrame.from_dict({'key': [1, 2], 'value': [100, 200]})

# Use suffixes to distinguish
result = left.merge(right, on='key', suffixes=('_old', '_new'))
# key | value_old | value_new
# 1   | 10        | 100
# 2   | 20        | 200
```

#### merge() Design Notes

- **One-to-many joins:** Cartesian product of matching rows
- **None handling:** None values in keys can match None
- **Validation:** Raises `KeyError` for non-existent columns, `ValueError` for invalid parameters
- **Performance:** Uses dict-based lookup for efficient joins
- **Stdlib only:** No external dependencies

## Missing Data Operations

### Detecting Missing Values

```python
# Check for None values (returns DataFrame of booleans)
is_null = df.isna()
print(is_null.data)  # [[False, True], [True, False], ...]

# Check for non-None values
not_null = df.notna()

# Check if columns have None values
nulls = df.has_nulls()
print(nulls)  # {'age': False, 'email': True}
```

### Removing Missing Values

```python
# Drop rows with any None values (default)
clean_df = df.dropna()

# Drop rows only if all values are None
df.dropna(axis=0, how='all')

# Drop columns with any None values
df.dropna(axis=1, how='any')

# Drop columns only if all values are None
df.dropna(axis=1, how='all')
```

### Filling Missing Values

```python
# Fill all None with a single value
df.fillna(0)

# Fill with column-specific values
df.fillna({
    'age': 0,
    'name': 'Unknown',
    'score': -1
})
```

### Complete Data Cleaning Pipeline

```python
# Clean data by removing bad rows and filling missing values
clean_df = (
    df.dropna(axis=1, how='all')     # Remove empty columns
    .fillna({'age': 0, 'name': ''})  # Fill remaining None
    .filter(lambda row: row['age'] >= 0)  # Remove invalid rows
)
```

## DataFrame Utilities

### Comparing DataFrames

```python
# Check if two DataFrames are identical
df1 = DataFrame.from_dict({'a': [1, 2], 'b': [3, 4]})
df2 = DataFrame.from_dict({'a': [1, 2], 'b': [3, 4]})

assert df1.equals(df2)  # True

# Order matters
df3 = DataFrame.from_dict({'a': [2, 1], 'b': [4, 3]})
assert not df1.equals(df3)  # False
```

### Copying DataFrames

```python
# Create independent copy
df_copy = df.deepcopy()

# Modifications to copy don't affect original
df_copy.data[0][0] = 'modified'
assert df.data[0][0] != 'modified'
```

### Counting Unique Values

```python
# Count unique values in a column
unique_count = df.nunique('category')
print(f"Found {unique_count} unique categories")

# Get the actual unique values
unique_values = df.unique('category')
print(f"Categories: {unique_values}")

# Count occurrences of each value
value_counts = df.value_counts('status')
print(value_counts)  # {'active': 10, 'inactive': 5}
```

## Statistical Analysis

### Summary Statistics

```python
# Generate statistical summary
stats = df.describe()

# Results include: count, mean, std, min, 25%, 50%, 75%, max
# Non-numeric columns show None for statistics
print(stats.get_column('age'))  # [5, 32.5, 4.2, 25, 28, 31, 36, 45]
print(stats.get_column('stat'))  # ['count', 'mean', 'std', ...]
```

### Group By Operations

The `groupby()` method provides SQL-like aggregation capabilities. It returns a `GroupBy` helper object that builds groups internally and provides aggregation methods.

**How it works:**
- Groups rows by unique values in the specified column
- Filters out rows where the grouping column is None
- Provides aggregation methods that return new DataFrames
- Uses eager evaluation (groups are built immediately)
- Uses only Python stdlib (statistics module for mean)

**Available aggregation methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `count()` | Count rows per group | DataFrame with `[group_col, 'count']` |
| `sum(col)` | Sum numeric column per group | DataFrame with `[group_col, 'col_sum']` |
| `mean(col)` | Average numeric column per group | DataFrame with `[group_col, 'col_mean']` |
| `min(col)` | Minimum value per group | DataFrame with `[group_col, 'col_min']` |
| `max(col)` | Maximum value per group | DataFrame with `[group_col, 'col_max']` |

**Basic usage:**

```python
from chapkit.data import DataFrame

# Sample sales data
df = DataFrame(
    columns=['region', 'product', 'sales', 'quantity'],
    data=[
        ['North', 'Widget', 1000, 10],
        ['North', 'Gadget', 1500, 15],
        ['South', 'Widget', 800, 8],
        ['South', 'Gadget', 1200, 12],
        ['North', 'Widget', 1100, 11],
    ]
)

# Count rows per group
region_counts = df.groupby('region').count()
# Returns: DataFrame({'region': ['North', 'South'], 'count': [3, 2]})

# Sum sales by region
total_sales = df.groupby('region').sum('sales')
# Returns: DataFrame({'region': ['North', 'South'], 'sales_sum': [3600, 2000]})

# Average quantity by product
avg_qty = df.groupby('product').mean('quantity')
# Returns: DataFrame({'product': ['Widget', 'Gadget'], 'quantity_mean': [9.67, 13.5]})

# Find min/max sales by region
min_sales = df.groupby('region').min('sales')
max_sales = df.groupby('region').max('sales')
```

**Advanced patterns:**

```python
# Multiple aggregations on same grouping
grouped = df.groupby('region')
summary = DataFrame(
    columns=['region', 'count', 'total_sales', 'avg_sales'],
    data=[
        [
            group,
            grouped.count().filter(lambda r: r['region'] == group)[0]['count'],
            grouped.sum('sales').filter(lambda r: r['region'] == group)[0]['sales_sum'],
            grouped.mean('sales').filter(lambda r: r['region'] == group)[0]['sales_mean'],
        ]
        for group in df.unique('region')
    ]
)

# Combine with filtering
high_sales = df.filter(lambda r: r['sales'] > 1000)
summary = high_sales.groupby('region').count()

# Chain with other operations
df.groupby('product').sum('sales').sort('sales_sum', reverse=True).head(5)
```

**Design notes:**
- Groups are built eagerly when `groupby()` is called
- Aggregation methods filter out None values automatically
- All methods return new DataFrame instances (immutable pattern)
- Uses stdlib only (no pandas/numpy dependencies)
- Raises KeyError if column not found

## Common Patterns

### Data Pipeline

```python
# Read, process, write
df = (
    DataFrame.from_csv("input.csv")
    .select(["name", "age", "score"])
    .rename({"score": "grade"})
    .filter(lambda row: row['age'] >= 18)
    .drop_duplicates(subset=['name'])
    .fillna({'grade': 0})
    .head(1000)
)
df.to_csv("output.csv")

# Advanced pipeline with transformations
df = (
    DataFrame.from_csv("sales.csv")
    .drop(['internal_id', 'debug_flag'])
    .rename({'product_name': 'product', 'sale_amount': 'amount'})
    .filter(lambda row: row['amount'] > 0)
    .apply(str.upper, 'product')
    .drop_duplicates(subset=['order_id'])
    .sort('amount', ascending=False)
)
```

### API Data Validation

```python
from fastapi import FastAPI, HTTPException
from chapkit.data import DataFrame

@app.post("/data/$validate")
async def validate_data(file: UploadFile):
    """Validate uploaded CSV data."""
    content = await file.read()
    df = DataFrame.from_csv(csv_string=content.decode())

    # Validate structure
    try:
        df.validate_structure()
    except ValueError as e:
        raise HTTPException(400, f"Invalid structure: {e}")

    # Check required columns
    required = ["user_id", "timestamp", "value"]
    missing = set(required) - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    # Check for nulls
    nulls = df.has_nulls()
    if any(nulls.get(col, False) for col in required):
        raise HTTPException(400, "Required columns contain null values")

    # Return metadata
    return {
        "rows": df.shape[0],
        "columns": df.columns,
        "types": df.infer_types(),
        "sample": df.head(5).to_dict(orient="records")
    }
```

### Data Transformation

```python
def clean_dataframe(df: DataFrame) -> DataFrame:
    """Clean and standardize DataFrame."""
    # Remove unnecessary columns
    df = df.drop(["temp", "debug"])

    # Rename for consistency
    df = df.rename({
        "user_name": "name",
        "user_age": "age"
    })

    # Validate
    df.validate_structure()

    return df
```

### Format Conversion

```python
# CSV to Pandas
df = DataFrame.from_csv("data.csv")
pandas_df = df.to_pandas()

# Pandas to CSV
df = DataFrame.from_pandas(pandas_df)
df.to_csv("output.csv")

# Cross-library conversion
polars_df = DataFrame.from_pandas(pandas_df).to_polars()
```

## API Response Formats

### JSON Response

```python
from fastapi import FastAPI
from chapkit.data import DataFrame

@app.get("/data")
async def get_data():
    """Return data as JSON."""
    df = get_dataframe()
    return df.to_dict(orient="records")
```

### CSV Download

```python
from fastapi.responses import Response

@app.get("/data/export.csv")
async def download_csv():
    """Download data as CSV."""
    df = get_dataframe()
    csv_data = df.to_csv()

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=data.csv"
        }
    )
```

### Paginated Response

```python
from pydantic import BaseModel

class PaginatedData(BaseModel):
    """Paginated DataFrame response."""
    total: int
    page: int
    page_size: int
    data: list[dict]

@app.get("/data", response_model=PaginatedData)
async def get_paginated_data(page: int = 1, page_size: int = 100):
    """Return paginated data."""
    df = get_dataframe()

    # Calculate pagination
    total = df.shape[0]
    start = (page - 1) * page_size

    # Get page using head/tail
    if start + page_size < total:
        page_df = df.tail(-start).head(page_size)
    else:
        page_df = df.tail(-start)

    return PaginatedData(
        total=total,
        page=page,
        page_size=page_size,
        data=page_df.to_dict(orient="records")
    )
```

## Error Handling

### Import Errors

DataFrames with optional dependencies raise clear errors:

```python
try:
    df.to_pandas()
except ImportError as e:
    print(e)
    # "pandas is required for to_pandas(). Install with: uv add pandas"
```

### Validation Errors

```python
# Column not found
try:
    df.select(["nonexistent"])
except KeyError as e:
    print(e)  # "Column 'nonexistent' not found in DataFrame"

# Invalid structure
try:
    df = DataFrame(columns=["a", "b"], data=[[1, 2, 3]])
    df.validate_structure()
except ValueError as e:
    print(e)  # "Row 0 has 3 values, expected 2"
```

### CSV Errors

```python
# File not found
try:
    df = DataFrame.from_csv("missing.csv")
except FileNotFoundError as e:
    print(e)  # "File not found: missing.csv"

# Invalid parameters
try:
    df = DataFrame.from_csv()  # Neither path nor csv_string
except ValueError as e:
    print(e)  # "Either path or csv_string must be provided"
```

## Performance Considerations

### When to Use DataFrame

**Good use cases:**
- API data interchange (JSON ↔ DataFrame ↔ CSV)
- Small to medium datasets (<100k rows)
- Format conversion between libraries
- Data validation and inspection
- Prototyping and development

**Not recommended for:**
- Large datasets (>1M rows) - use pandas/polars directly
- Heavy data transformations - use specialized libraries
- Production analytics - use pandas/polars/DuckDB
- High-performance computing - use NumPy/pandas

### Memory Efficiency

```python
# DataFrame stores data as list of lists (row-oriented)
# For large datasets, convert to columnar format:
pandas_df = df.to_pandas()  # More efficient for operations

# For very large files, consider streaming:
# - Read in chunks with pandas
# - Process incrementally
# - Use DataFrame for API boundaries only
```

## Testing with DataFrame

### Test Data Creation

```python
import pytest
from chapkit.data import DataFrame

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return DataFrame.from_dict({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95, 87, 92]
    })

def test_data_processing(sample_data):
    """Test data processing pipeline."""
    result = sample_data.select(["name", "score"])
    assert result.shape == (3, 2)
    assert "id" not in result.columns
```

### CSV Round-Trip Testing

```python
def test_csv_roundtrip(tmp_path):
    """Test CSV export and import."""
    # Create test data
    original = DataFrame.from_dict({
        "name": ["Alice", "Bob"],
        "age": [25, 30]
    })

    # Write to file
    csv_file = tmp_path / "test.csv"
    original.to_csv(csv_file)

    # Read back
    restored = DataFrame.from_csv(csv_file)

    # Verify (note: CSV makes all values strings)
    assert restored.columns == original.columns
    assert restored.shape == original.shape
```

## Best Practices

### Recommended Practices

- **Validate early**: Call `validate_structure()` after data ingestion
- **Check types**: Use `infer_types()` to understand your data
- **Handle nulls**: Use `has_nulls()` to detect data quality issues
- **Immutable pattern**: Chain operations without modifying originals
- **Small data**: Use DataFrame for API boundaries, not heavy processing
- **Clear errors**: Let ImportError guide users to install dependencies
- **CSV for interchange**: Use CSV for human-readable data exchange

### Avoid

- **Large datasets**: Don't use for >100k rows (use pandas/polars instead)
- **Heavy operations**: Don't use for joins, aggregations, complex queries
- **In-place modification**: Don't try to modify DataFrames (they're immutable)
- **Type assumptions**: CSV imports make all values strings
- **Missing validation**: Always validate data from external sources

## Dependencies

### Core (Required)
- **pydantic**: For data validation and serialization

### Optional (No Dependencies)
- **CSV support**: Uses Python stdlib `csv` module
- **Data inspection**: Uses Python stdlib `random` module
- **Column operations**: Pure Python (no dependencies)
- **Validation**: Pure Python (no dependencies)

### Optional (With Dependencies)
Install as needed:

```bash
# For pandas support
uv add pandas

# For polars support
uv add polars

# For xarray support
uv add xarray

# For PyArrow/Parquet support (future)
uv add 'servicekit[arrow]'
```

## Examples

### Example Files

See `examples/` directory:
- Basic DataFrame operations
- API integration patterns
- CSV upload/download
- Data validation workflows

### Interactive Session

```python
from chapkit.data import DataFrame

# Create sample data
df = DataFrame.from_dict({
    "product": ["Apple", "Banana", "Cherry", "Date"],
    "price": [1.2, 0.5, 3.0, 2.5],
    "stock": [100, 150, 80, 60]
})

# Explore
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(df.head(2))

# Analyze
print(f"Types: {df.infer_types()}")
print(f"Has nulls: {df.has_nulls()}")

# Transform
expensive = df.select(["product", "price"])
print(expensive.to_dict(orient="records"))

# Export
expensive.to_csv("expensive_items.csv")
```

## Troubleshooting

### ImportError for Optional Libraries

**Problem**: Getting ImportError when using pandas/polars methods.

**Solution**:
```bash
# Install required library
uv add pandas  # or polars, xarray
```

### CSV Values All Strings

**Problem**: After `from_csv()`, all values are strings.

**Solution**: CSV format doesn't preserve types. Either:
1. Convert manually after import
2. Use `infer_types()` to detect types
3. Use Parquet for type preservation (future feature)

### Column Not Found Errors

**Problem**: `KeyError: Column 'x' not found`.

**Solution**:
```python
# Check available columns
print(df.columns)

# Case-sensitive matching
df.select(["Name"])  # Error if column is "name"
```

### Memory Issues with Large Data

**Problem**: Running out of memory with large DataFrames.

**Solution**: DataFrame is designed for small-medium data. For large datasets:
```python
# Use pandas directly for large data
import pandas as pd
pandas_df = pd.read_csv("large_file.csv", chunksize=10000)

# Or use polars for better performance
import polars as pl
polars_df = pl.read_csv("large_file.csv")
```

## Next Steps

- **Learn More**: See other guides for integrating DataFrame with APIs
- **Contribute**: Submit PRs for new format support (Parquet, Arrow, etc.)
- **Examples**: Check `examples/` directory for real-world usage

For related features, see:
- [Servicekit Repository](https://github.com/winterop-com/servicekit) - Building services with servicekit
