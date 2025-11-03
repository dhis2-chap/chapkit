"""Example showing DataFrame analytics and column access methods."""

from chapkit.data import DataFrame

# Create sample sales data
df = DataFrame.from_dict(
    {
        "product": ["Apple", "Banana", "Cherry", "Apple", "Banana", "Apple", "Cherry", "Banana", "Apple", "Cherry"],
        "category": ["Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit"],
        "price": [1.20, 0.50, 3.00, 1.20, 0.50, 1.20, 3.00, 0.50, 1.20, 3.00],
        "quantity": [10, 25, 5, 15, 30, 8, 12, 20, 18, 7],
        "store": ["A", "A", "B", "B", "A", "C", "A", "B", "C", "C"],
    }
)

print("=== DataFrame Info ===")
print(f"Total rows: {len(df)}")
print(f"Shape: {df.shape}")
print()

print("=== Column Access ===")
# Get single column as list
products = df.get_column("product")
print(f"Products: {products[:5]}...")

# Get multiple columns as DataFrame
prices_and_quantities = df.select(["product", "price", "quantity"])
print(f"Price/Qty shape: {prices_and_quantities.shape}")
print()

print("=== Unique Values ===")
unique_products = df.unique("product")
print(f"Unique products: {unique_products}")

unique_stores = df.unique("store")
print(f"Unique stores: {unique_stores}")
print()

print("=== Value Counts ===")
product_counts = df.value_counts("product")
print("Product sales count:")
for product, count in product_counts.items():
    print(f"  {product}: {count} transactions")
print()

store_counts = df.value_counts("store")
print("Transactions per store:")
for store, count in store_counts.items():
    print(f"  Store {store}: {count} transactions")
print()

print("=== Sorting ===")
# Sort by price (ascending)
sorted_by_price = df.sort("price")
print("Cheapest products:")
for row in sorted_by_price.head(3):
    print(f"  {row['product']}: ${row['price']}")
print()

# Sort by quantity (descending)
sorted_by_qty = df.sort("quantity", ascending=False)
print("Largest quantity orders:")
for row in sorted_by_qty.head(3):
    print(f"  {row['product']}: {row['quantity']} units")
print()

print("=== Iteration ===")
print("All transactions:")
for i, row in enumerate(df):
    if i >= 3:  # Show first 3
        print(f"  ... and {len(df) - 3} more")
        break
    print(f"  {row['product']} at Store {row['store']}: {row['quantity']} units @ ${row['price']}")
print()

print("=== Analytics with Iteration ===")
# Calculate total revenue per product
revenue_by_product: dict[str, float] = {}
for row in df:
    product = row["product"]
    revenue = row["price"] * row["quantity"]
    revenue_by_product[product] = revenue_by_product.get(product, 0.0) + revenue

print("Total revenue by product:")
for product in df.unique("product"):
    print(f"  {product}: ${revenue_by_product[product]:.2f}")
print()

# Find best selling product
best_seller = max(revenue_by_product, key=lambda k: revenue_by_product[k])
print(f"Best selling product: {best_seller} (${revenue_by_product[best_seller]:.2f})")
print()

print("=== Finding Specific Data ===")
# Find unique stores selling Apples using iteration
apple_stores: list[str] = []
for row in df:
    if row["product"] == "Apple" and row["store"] not in apple_stores:
        apple_stores.append(row["store"])
print(f"Stores selling Apples: {sorted(apple_stores)}")
