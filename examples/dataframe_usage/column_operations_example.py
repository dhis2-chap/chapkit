"""Example showing DataFrame column operations."""

from chapkit.data import DataFrame

# Create sample data
df = DataFrame.from_dict(
    {
        "id": [1, 2, 3, 4, 5],
        "first_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "last_name": ["Smith", "Jones", "Brown", "Wilson", "Davis"],
        "age": [25, 30, 35, 28, 42],
        "email": [
            "alice@example.com",
            "bob@example.com",
            "charlie@example.com",
            "diana@example.com",
            "eve@example.com",
        ],
        "temp_field": ["x", "y", "z", "a", "b"],
        "debug_col": [1, 2, 3, 4, 5],
    }
)

print("=== Original DataFrame ===")
print(f"Columns: {df.columns}")
print(f"Shape: {df.shape}")
print()

print("=== Selecting Columns ===")
# Select specific columns
user_info = df.select(["id", "first_name", "last_name", "email"])
print(f"Selected columns: {user_info.columns}")
print(f"Shape: {user_info.shape}")
print("\nFirst 2 rows:")
for row in user_info.head(2).to_dict(orient="records"):
    print(f"  {row}")
print()

# Select single column
ages = df.select(["age"])
print(f"Single column selection: {ages.columns}")
print(f"Ages: {ages.to_dict(orient='list')['age']}")
print()

print("=== Dropping Columns ===")
# Drop temporary/debug columns
clean_df = df.drop(["temp_field", "debug_col"])
print(f"After dropping temp columns: {clean_df.columns}")
print(f"Shape: {clean_df.shape}")
print()

# Drop multiple at once
minimal = df.drop(["temp_field", "debug_col", "last_name"])
print(f"Minimal columns: {minimal.columns}")
print()

print("=== Renaming Columns ===")
# Rename for clarity
renamed = df.rename({"first_name": "name", "last_name": "surname", "temp_field": "status"})
print(f"Renamed columns: {renamed.columns}")
print()

# Chain operations
result = (
    df.drop(["temp_field", "debug_col"])
    .rename({"first_name": "name", "last_name": "surname"})
    .select(["id", "name", "surname", "age"])
)

print("=== Chained Operations ===")
print("Pipeline: drop -> rename -> select")
print(f"Final columns: {result.columns}")
print(f"Final shape: {result.shape}")
print("\nResult:")
for row in result.head(3).to_dict(orient="records"):
    print(f"  {row}")
print()

print("=== Error Handling ===")
try:
    df.select(["nonexistent_column"])
except KeyError as e:
    print(f"✓ Caught error: {e}")

try:
    df.drop(["missing_column"])
except KeyError as e:
    print(f"✓ Caught error: {e}")

try:
    df.rename({"missing": "new_name"})
except KeyError as e:
    print(f"✓ Caught error: {e}")

try:
    df.rename({"first_name": "last_name"})  # Would create duplicate
except ValueError as e:
    print(f"✓ Caught error: {e}")
