"""Example showing DataFrame data inspection methods."""

from chapkit.data import DataFrame

# Create sample data
df = DataFrame.from_dict(
    {
        "product": ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon"],
        "price": [1.20, 0.50, 3.00, 2.50, 4.00, 2.00, 2.50, 3.50, 1.80, 0.80],
        "stock": [100, 150, 80, 60, 40, 90, 120, 70, 110, 130],
        "category": ["Fruit", "Fruit", "Fruit", "Fruit", "Berry", "Fruit", "Fruit", "Melon", "Fruit", "Citrus"],
    }
)

print("=== DataFrame Properties ===")
print(f"Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Size (total elements): {df.size}")
print(f"Is empty: {df.empty}")
print(f"Dimensions: {df.ndim}")
print()

print("=== Head and Tail ===")
print("First 3 rows:")
head_df = df.head(3)
for row in head_df.to_dict(orient="records"):
    print(f"  {row}")
print()

print("Last 3 rows:")
tail_df = df.tail(3)
for row in tail_df.to_dict(orient="records"):
    print(f"  {row}")
print()

print("All except last 2 rows (head with negative):")
except_last = df.head(-2)
print(f"  Shape: {except_last.shape}")
print()

print("All except first 2 rows (tail with negative):")
except_first = df.tail(-2)
print(f"  Shape: {except_first.shape}")
print()

print("=== Random Sampling ===")
print("Random sample of 3 rows:")
sample_df = df.sample(n=3, random_state=42)
for row in sample_df.to_dict(orient="records"):
    print(f"  {row}")
print()

print("Random 20% sample:")
frac_sample = df.sample(frac=0.2, random_state=42)
print(f"  Sampled {frac_sample.shape[0]} rows (20% of {df.shape[0]})")
print()

print("=== Reproducible Sampling ===")
sample1 = df.sample(n=5, random_state=123)
sample2 = df.sample(n=5, random_state=123)
print(f"Same random seed produces same sample: {sample1.data == sample2.data}")
print()

print("=== Data Summary ===")
print(f"Products: {', '.join(df.select(['product']).head(5).to_dict(orient='list')['product'])}, ...")
print(f"Price range: ${min(df.to_dict(orient='list')['price'])} - ${max(df.to_dict(orient='list')['price'])}")
print(f"Total stock: {sum(df.to_dict(orient='list')['stock'])} units")
