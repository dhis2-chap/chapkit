"""Example showing DataFrame JSON support."""

import json

from chapkit.data import DataFrame

print("=== Creating DataFrame from JSON ===")

# JSON array of objects (most common API format)
json_data = """[
    {"name": "Alice", "age": 25, "city": "NYC"},
    {"name": "Bob", "age": 30, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]"""

df = DataFrame.from_json(json_data)
print(f"Created DataFrame with shape: {df.shape}")
print(f"Columns: {df.columns}")
print()

print("=== Exporting to JSON (records format) ===")
# Default: array of objects
json_records = df.to_json()
print(json_records)
print()

print("=== Exporting to JSON (columns format) ===")
# Column-oriented: object with arrays
json_columns = df.to_json(orient="columns")
print(json_columns)
print()

print("=== Round-trip Conversion ===")
# Convert to JSON and back
original = DataFrame.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]})
json_str = original.to_json()
restored = DataFrame.from_json(json_str)

print(f"Original: {original.data}")
print(f"JSON: {json_str}")
print(f"Restored: {restored.data}")
print(f"Data preserved: {original.data == restored.data}")
print()

print("=== Simulating API Response ===")


# Simulate fetching data from API
def fetch_api_data() -> str:
    """Simulate API response."""
    return """[
        {"product": "Laptop", "price": 999.99, "stock": 15},
        {"product": "Mouse", "price": 29.99, "stock": 150},
        {"product": "Keyboard", "price": 79.99, "stock": 50}
    ]"""


# Process API response
api_response = fetch_api_data()
df_products = DataFrame.from_json(api_response)

print("Products from API:")
for row in df_products:
    print(f"  {row['product']}: ${row['price']} ({row['stock']} in stock)")
print()

print("=== Filter and Export ===")
# Filter products in stock and export
in_stock: list[dict] = []
for row in df_products:
    if row["stock"] > 20:
        in_stock.append(row)

if in_stock:
    # Create new DataFrame with filtered data
    df_filtered = DataFrame.from_records(in_stock)
    filtered_json = df_filtered.to_json()
    print("Products with >20 in stock:")
    print(filtered_json)
    print()

print("=== JSON with Different Data Types ===")
# JSON handles various types
mixed_data = """[
    {"name": "Test", "count": 42, "price": 19.99, "active": true},
    {"name": "Demo", "count": 0, "price": 0.99, "active": false}
]"""

df_mixed = DataFrame.from_json(mixed_data)
print("Types inferred from JSON:")
types = df_mixed.infer_types()
for col, dtype in types.items():
    print(f"  {col}: {dtype}")
print()

print("=== Pretty JSON for Humans ===")
# Export with indentation for readability
df_small = DataFrame.from_dict({"name": ["Alice", "Bob"], "score": [95, 87]})
pretty_json = json.dumps(df_small.to_dict(orient="records"), indent=2)
print("Pretty formatted JSON:")
print(pretty_json)
