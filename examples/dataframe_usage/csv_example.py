"""Example showing DataFrame CSV read/write capabilities."""

from pathlib import Path

from chapkit.data import DataFrame

# Create sample data
df = DataFrame.from_dict(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 42],
        "city": ["NYC", "SF", "LA", "Chicago", "Boston"],
        "score": [95.5, 87.0, 92.3, 88.5, 91.0],
    }
)

print("Original DataFrame:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(df.head(3).to_dict(orient="records"))
print()

# Write to CSV file
output_file = Path("output.csv")
df.to_csv(output_file)
print(f"✓ Wrote to {output_file}")
print()

# Read back from CSV
df_from_file = DataFrame.from_csv(output_file)
print("Read from CSV file:")
print(f"Shape: {df_from_file.shape}")
print(f"Columns: {df_from_file.columns}")
print()

# CSV to string
csv_string = df.to_csv()
print("CSV as string:")
print(csv_string)
print()

# Read from CSV string
df_from_string = DataFrame.from_csv(csv_string=csv_string)
print("Read from CSV string:")
print(f"Shape: {df_from_string.shape}")
print()

# Custom delimiter (TSV)
tsv_file = Path("output.tsv")
df.to_csv(tsv_file, delimiter="\t")
print(f"✓ Wrote TSV to {tsv_file}")
print()

df_tsv = DataFrame.from_csv(tsv_file, delimiter="\t")
print("Read from TSV:")
print(f"Columns: {df_tsv.columns}")
print()

# Without header
no_header_file = Path("output_no_header.csv")
df.to_csv(no_header_file, include_header=False)
print(f"✓ Wrote CSV without header to {no_header_file}")
print()

df_no_header = DataFrame.from_csv(no_header_file, has_header=False)
print("Read CSV without header (auto-generated columns):")
print(f"Columns: {df_no_header.columns}")
print()

# Cleanup
output_file.unlink()
tsv_file.unlink()
no_header_file.unlink()
print("✓ Cleaned up example files")
