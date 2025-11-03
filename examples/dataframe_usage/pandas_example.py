"""Comprehensive example showing DataFrame usage with pandas."""

import numpy as np
import pandas as pd

from chapkit.data import DataFrame

print("=== Basic Conversion ===")
# Create a pandas DataFrame
pdf = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["New York", "London", "Paris"]})

print("Original pandas DataFrame:")
print(pdf)
print()

# Convert to servicekit DataFrame
sk_df = DataFrame.from_pandas(pdf)

print("Servicekit DataFrame:")
print(f"Columns: {sk_df.columns}")
print(f"Shape: {sk_df.shape}")
print(f"Data: {sk_df.data}")
print()

# Convert back to pandas
pdf_back = sk_df.to_pandas()

print("Converted back to pandas:")
print(pdf_back)
print(f"DataFrames equal: {pdf.equals(pdf_back)}")
print()

print("=== Handling Different Data Types ===")
pdf_types = pd.DataFrame(
    {
        "integers": [1, 2, 3],
        "floats": [1.5, 2.5, 3.5],
        "strings": ["a", "b", "c"],
        "booleans": [True, False, True],
        "dates": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    }
)

print("Pandas with various types:")
print(pdf_types.dtypes)
print()

sk_df_types = DataFrame.from_pandas(pdf_types)
print("Servicekit DataFrame columns:")
print(sk_df_types.columns)
print("Inferred types:")
types = sk_df_types.infer_types()
for col, dtype in types.items():
    print(f"  {col}: {dtype}")
print()

print("=== Handling None/NaN Values ===")
pdf_nulls = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": ["x", None, "z", "w"], "c": [1.0, 2.0, np.nan, 4.0]})

print("Pandas with NaN/None:")
print(pdf_nulls)
print()

sk_df_nulls = DataFrame.from_pandas(pdf_nulls)
print("Null detection in servicekit DataFrame:")
nulls = sk_df_nulls.has_nulls()
for col, has_null in nulls.items():
    print(f"  {col}: {'Has nulls' if has_null else 'No nulls'}")
print()

print("=== Working with Pandas Operations ===")
# Demonstrate that you can use pandas operations, convert to servicekit, then back
pdf_calc = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

# Do pandas operations
pdf_calc["sum"] = pdf_calc["x"] + pdf_calc["y"]
pdf_calc["product"] = pdf_calc["x"] * pdf_calc["y"]

print("Pandas DataFrame after calculations:")
print(pdf_calc)
print()

# Convert to servicekit
sk_df_calc = DataFrame.from_pandas(pdf_calc)

# Use servicekit operations
filtered = sk_df_calc.sort("sum", ascending=False)
top_3 = filtered.head(3)

print("Top 3 by sum (using servicekit operations):")
for row in top_3:
    print(f"  x={row['x']}, y={row['y']}, sum={row['sum']}, product={row['product']}")
print()

print("=== Round-trip Verification ===")
original = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]})

# Round-trip
intermediate = DataFrame.from_pandas(original)
restored = intermediate.to_pandas()

print("Original and restored are equal:", original.equals(restored))
print("Original dtypes:", original.dtypes.to_dict())
print("Restored dtypes:", restored.dtypes.to_dict())

# Export variables for testing
df = pdf
df_back = pdf_back
