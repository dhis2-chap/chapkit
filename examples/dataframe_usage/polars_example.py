"""Comprehensive example showing DataFrame usage with Polars."""

import polars as pl

from chapkit.data import DataFrame

print("=== Basic Conversion ===")
# Create a Polars DataFrame
pldf = pl.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["New York", "London", "Paris"]})

print("Original Polars DataFrame:")
print(pldf)
print()

# Convert to servicekit DataFrame
sk_df = DataFrame.from_polars(pldf)

print("Servicekit DataFrame:")
print(f"Columns: {sk_df.columns}")
print(f"Shape: {sk_df.shape}")
print(f"Data: {sk_df.data}")
print()

# Convert back to Polars
pldf_back = sk_df.to_polars()

print("Converted back to Polars:")
print(pldf_back)
print(f"DataFrames equal: {pldf.equals(pldf_back)}")
print()

print("=== Handling Different Data Types ===")
pldf_types = pl.DataFrame(
    {
        "integers": [1, 2, 3],
        "floats": [1.5, 2.5, 3.5],
        "strings": ["a", "b", "c"],
        "booleans": [True, False, True],
    }
)

print("Polars DataFrame types:")
print(pldf_types.dtypes)
print()

sk_df_types = DataFrame.from_polars(pldf_types)
print("Servicekit DataFrame inferred types:")
types = sk_df_types.infer_types()
for col, dtype in types.items():
    print(f"  {col}: {dtype}")
print()

print("=== Handling None Values ===")
pldf_nulls = pl.DataFrame({"a": [1, None, 3, 4], "b": ["x", None, "z", "w"], "c": [1.0, 2.0, None, 4.0]})

print("Polars with None values:")
print(pldf_nulls)
print()

sk_df_nulls = DataFrame.from_polars(pldf_nulls)
print("Null detection in servicekit DataFrame:")
nulls = sk_df_nulls.has_nulls()
for col, has_null in nulls.items():
    print(f"  {col}: {'Has nulls' if has_null else 'No nulls'}")
print()

print("=== Working with Polars Operations ===")
# Demonstrate that you can use Polars operations, convert to servicekit, then back
pldf_calc = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

# Do Polars operations (expressions)
pldf_calc = pldf_calc.with_columns(
    [(pl.col("x") + pl.col("y")).alias("sum"), (pl.col("x") * pl.col("y")).alias("product")]
)

print("Polars DataFrame after calculations:")
print(pldf_calc)
print()

# Convert to servicekit
sk_df_calc = DataFrame.from_polars(pldf_calc)

# Use servicekit operations
sorted_df = sk_df_calc.sort("product", ascending=False)
top_3 = sorted_df.head(3)

print("Top 3 by product (using servicekit operations):")
for row in top_3:
    print(f"  x={row['x']}, y={row['y']}, sum={row['sum']}, product={row['product']}")
print()

print("=== Polars Lazy Operations ===")
# Show that you can work with lazy Polars DataFrames too
lazy_df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

# Apply lazy transformations
lazy_result = lazy_df.filter(pl.col("a") > 2).select([pl.col("a"), pl.col("b")])

# Collect to eager DataFrame
eager_result = lazy_result.collect()

print("Polars lazy DataFrame (collected):")
print(eager_result)
print()

# Convert to servicekit
sk_from_lazy = DataFrame.from_polars(eager_result)
print("Converted from Polars lazy result:")
print(f"Shape: {sk_from_lazy.shape}")
print(f"Data: {sk_from_lazy.data}")
print()

print("=== Round-trip Verification ===")
original = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]})

# Round-trip
intermediate = DataFrame.from_polars(original)
restored = intermediate.to_polars()

print("Original and restored are equal:", original.equals(restored))
print("Original schema:", original.schema)
print("Restored schema:", restored.schema)

# Export variables for testing
df = pldf
df_back = pldf_back
