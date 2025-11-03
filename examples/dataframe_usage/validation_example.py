"""Example showing DataFrame validation and type inference."""

from chapkit.data import DataFrame

# Create sample data with various types
df = DataFrame.from_dict(
    {
        "user_id": [1, 2, 3, 4, 5],
        "username": ["alice", "bob", "charlie", "diana", "eve"],
        "age": [25, 30, 35, 28, 42],
        "score": [95.5, 87.0, 92.3, 88.5, 91.0],
        "active": [True, True, False, True, True],
        "notes": ["Good", None, "Excellent", None, "Average"],
    }
)

print("=== DataFrame Structure Validation ===")
try:
    df.validate_structure()
    print("✓ DataFrame structure is valid")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
print()

print("=== Type Inference ===")
types = df.infer_types()
print("Inferred column types:")
for col, dtype in types.items():
    print(f"  {col}: {dtype}")
print()

print("=== Null Detection ===")
nulls = df.has_nulls()
print("Columns with null values:")
for col, has_null in nulls.items():
    status = "✓ Has nulls" if has_null else "✗ No nulls"
    print(f"  {col}: {status}")
print()

# Create DataFrame with mixed types
mixed_df = DataFrame.from_dict(
    {
        "mixed_col": [1, "hello", 3.14, True, None],
        "int_float_col": [1, 2, 3.5, 4.0, 5],
        "all_null_col": [None, None, None, None, None],
    }
)

print("=== Mixed Type Detection ===")
mixed_types = mixed_df.infer_types()
print("Column types in mixed DataFrame:")
for col, dtype in mixed_types.items():
    print(f"  {col}: {dtype}")
print()

# Validation errors
print("=== Validation Error Examples ===")

# Duplicate columns
try:
    invalid_df = DataFrame(columns=["a", "a", "b"], data=[[1, 2, 3]])
    invalid_df.validate_structure()
except ValueError as e:
    print(f"✓ Duplicate columns caught: {e}")

# Unequal row lengths
try:
    invalid_df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4, 5]])
    invalid_df.validate_structure()
except ValueError as e:
    print(f"✓ Unequal row length caught: {e}")

# Empty column name
try:
    invalid_df = DataFrame(columns=["a", "", "c"], data=[[1, 2, 3]])
    invalid_df.validate_structure()
except ValueError as e:
    print(f"✓ Empty column name caught: {e}")
print()

# Data quality report
print("=== Data Quality Report ===")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"Structure valid: {'Yes' if df else 'N/A'}")
print()

print("Column Summary:")
for col in df.columns:
    dtype = types[col]
    has_null = nulls[col]
    null_count = sum(1 for row in df.data if row[df.columns.index(col)] is None)

    print(f"  {col}:")
    print(f"    Type: {dtype}")
    print(f"    Nulls: {null_count}/{df.shape[0]} ({null_count / df.shape[0] * 100:.1f}%)")
print()

# Filtering columns with nulls
print("=== Columns Requiring Attention ===")
null_columns = [col for col, has_null in nulls.items() if has_null]
if null_columns:
    print(f"Columns with null values: {', '.join(null_columns)}")
    print("Consider:")
    print("  - Filling nulls with default values")
    print("  - Dropping rows with nulls")
    print("  - Marking as optional fields")
else:
    print("✓ No null values found - data is complete")
