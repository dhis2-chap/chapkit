"""Comprehensive example showing DataFrame usage with xarray."""

import numpy as np
import xarray as xr

from chapkit.data import DataFrame

print("=== Basic 2D DataArray Conversion ===")
# Create a 2D xarray DataArray with named coordinates
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
da = xr.DataArray(data, dims=["row", "col"], coords={"row": ["A", "B", "C"], "col": ["x", "y", "z"]})

print("Original xarray DataArray:")
print(da)
print()

# Convert to servicekit DataFrame (goes through pandas internally)
sk_df = DataFrame.from_xarray(da)

print("Servicekit DataFrame:")
print(f"Columns: {sk_df.columns}")
print(f"Shape: {sk_df.shape}")
print(f"Data: {sk_df.data}")
print()

# Convert to pandas
pdf = sk_df.to_pandas()

print("Converted to pandas:")
print(pdf)
print()

print("=== Working with Climate Data Example ===")
# Simulate temperature data over time and location
temps = np.random.randn(5, 3) * 5 + 20  # Temperature around 20°C
times = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
locations = ["NYC", "LA", "Chicago"]

temp_da = xr.DataArray(
    temps, dims=["time", "location"], coords={"time": times, "location": locations}, attrs={"units": "celsius"}
)

print("Temperature DataArray:")
print(temp_da)
print()

# Convert to servicekit DataFrame
temp_df = DataFrame.from_xarray(temp_da)

print("Temperature as DataFrame:")
print(f"Columns (locations): {temp_df.columns}")
print(f"Rows (time points): {temp_df.shape[0]}")
print()

# Use servicekit operations
print("Temperature statistics per location:")
for col in temp_df.columns:
    temps_col = temp_df.get_column(col)
    print(f"  {col}: min={min(temps_col):.1f}, max={max(temps_col):.1f}, avg={sum(temps_col) / len(temps_col):.1f}")
print()

print("=== Multidimensional to Tabular ===")
# Show how xarray's multidimensional data becomes tabular
pressure_data = np.array([[1013, 1012, 1014], [1015, 1013, 1012]])
pressure_da = xr.DataArray(
    pressure_data,
    dims=["day", "station"],
    coords={"day": ["Mon", "Tue"], "station": ["Station1", "Station2", "Station3"]},
)

print("Pressure DataArray (2D):")
print(pressure_da)
print()

pressure_df = DataFrame.from_xarray(pressure_da)
print("As tabular DataFrame:")
print(f"Columns: {pressure_df.columns}")
for i, row in enumerate(pressure_df):
    day = pressure_da.coords["day"].values[i]
    print(f"  {day}: {row}")
print()

print("=== Type Handling ===")
# xarray preserves numpy dtypes
float_data = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
int_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

float_da = xr.DataArray(float_data, dims=["x", "y"], coords={"y": ["col1", "col2", "col3"]})
int_da = xr.DataArray(int_data, dims=["x", "y"], coords={"y": ["col1", "col2", "col3"]})

float_df = DataFrame.from_xarray(float_da)
int_df = DataFrame.from_xarray(int_da)

print("Float DataArray converted:")
print(f"  Inferred types: {float_df.infer_types()}")

print("Int DataArray converted:")
print(f"  Inferred types: {int_df.infer_types()}")
print()

print("=== Integration with Pandas ===")
# Show that xarray -> servicekit -> pandas works
original_da = xr.DataArray(
    np.array([[10, 20], [30, 40], [50, 60]]), dims=["time", "sensor"], coords={"sensor": ["temp", "humidity"]}
)

# Convert through servicekit
intermediate = DataFrame.from_xarray(original_da)
final_pdf = intermediate.to_pandas()

print("Final pandas DataFrame from xarray:")
print(final_pdf)
print(f"Shape preserved: {final_pdf.shape} (from {original_da.shape})")

# Export variables for testing
df = pdf
