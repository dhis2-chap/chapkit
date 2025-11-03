"""Universal DataFrame schema for chapkit services."""

import csv
import io
import json
import random
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel


class DataFrame(BaseModel):
    """Universal interchange format for tabular data from pandas, polars, xarray, and other libraries."""

    columns: list[str]
    data: list[list[Any]]

    @classmethod
    def from_pandas(cls, df: Any) -> Self:
        """Create schema from pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for from_pandas(). Install with: uv add pandas") from None

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

        return cls(
            columns=df.columns.tolist(),
            data=df.values.tolist(),
        )

    @classmethod
    def from_polars(cls, df: Any) -> Self:
        """Create schema from Polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required for from_polars(). Install with: uv add polars") from None

        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(df)}")

        return cls(
            columns=df.columns,
            data=[list(row) for row in df.rows()],
        )

    @classmethod
    def from_xarray(cls, da: Any) -> Self:
        """Create schema from xarray DataArray (2D only)."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for from_xarray(). Install with: uv add xarray") from None

        if not isinstance(da, xr.DataArray):
            raise TypeError(f"Expected xarray DataArray, got {type(da)}")

        if len(da.dims) != 2:
            raise ValueError(f"Only 2D DataArrays supported, got {len(da.dims)} dimensions")

        # Convert to pandas then use from_pandas
        pdf = da.to_pandas()
        return cls.from_pandas(pdf)

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> Self:
        """Create schema from dictionary of columns."""
        if not data:
            return cls(columns=[], data=[])

        columns = list(data.keys())
        num_rows = len(next(iter(data.values())))

        if not all(len(vals) == num_rows for vals in data.values()):
            raise ValueError("All columns must have the same length")

        rows = [[data[col][i] for col in columns] for i in range(num_rows)]

        return cls(columns=columns, data=rows)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> Self:
        """Create schema from list of records (row-oriented)."""
        if not records:
            return cls(columns=[], data=[])

        columns = list(records[0].keys())
        data = [[record[col] for col in columns] for record in records]

        return cls(columns=columns, data=data)

    @classmethod
    def from_csv(
        cls,
        path: str | Path | None = None,
        *,
        csv_string: str | None = None,
        delimiter: str = ",",
        has_header: bool = True,
        encoding: str = "utf-8",
    ) -> Self:
        """Create DataFrame from CSV file or string."""
        # Validate mutually exclusive parameters
        if path is None and csv_string is None:
            raise ValueError("Either path or csv_string must be provided")
        if path is not None and csv_string is not None:
            raise ValueError("path and csv_string are mutually exclusive")

        # Read CSV data
        if path is not None:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with path_obj.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
        else:
            # csv_string is not None
            string_io = io.StringIO(csv_string)
            reader = csv.reader(string_io, delimiter=delimiter)
            rows = list(reader)

        # Handle empty CSV
        if not rows:
            return cls(columns=[], data=[])

        # Extract columns and data
        if has_header:
            columns = rows[0]
            data = rows[1:]
        else:
            # Generate column names
            num_cols = len(rows[0]) if rows else 0
            columns = [f"col_{i}" for i in range(num_cols)]
            data = rows

        return cls(columns=columns, data=data)

    def to_pandas(self) -> Any:
        """Convert schema to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_pandas(). Install with: uv add pandas") from None

        return pd.DataFrame(self.data, columns=self.columns)

    def to_polars(self) -> Any:
        """Convert schema to Polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required for to_polars(). Install with: uv add polars") from None

        return pl.DataFrame(self.data, schema=self.columns, orient="row")

    def to_dict(self, orient: Literal["dict", "list", "records"] = "dict") -> Any:
        """Convert schema to dictionary with specified orient (dict, list, or records)."""
        if orient == "dict":
            return {col: {i: self.data[i][j] for i in range(len(self.data))} for j, col in enumerate(self.columns)}
        elif orient == "list":
            return {col: [row[j] for row in self.data] for j, col in enumerate(self.columns)}
        elif orient == "records":
            return [{col: row[j] for j, col in enumerate(self.columns)} for row in self.data]
        else:
            raise ValueError(f"Invalid orient: {orient}")

    def to_csv(
        self,
        path: str | Path | None = None,
        *,
        delimiter: str = ",",
        include_header: bool = True,
        encoding: str = "utf-8",
    ) -> str | None:
        """Export DataFrame to CSV file or string."""
        # Write to string buffer or file
        if path is None:
            # Return as string
            output = io.StringIO()
            writer = csv.writer(output, delimiter=delimiter)

            if include_header:
                writer.writerow(self.columns)

            writer.writerows(self.data)

            return output.getvalue()
        else:
            # Write to file
            path_obj = Path(path)
            with path_obj.open("w", encoding=encoding, newline="") as f:
                writer = csv.writer(f, delimiter=delimiter)

                if include_header:
                    writer.writerow(self.columns)

                writer.writerows(self.data)

            return None

    # Convenience aliases
    from_dataframe = from_pandas
    to_dataframe = to_pandas

    @property
    def shape(self) -> tuple[int, int]:
        """Return tuple representing dimensionality of the DataFrame."""
        return (len(self.data), len(self.columns))

    @property
    def empty(self) -> bool:
        """Indicator whether DataFrame is empty."""
        return len(self.data) == 0 or len(self.columns) == 0

    @property
    def size(self) -> int:
        """Return int representing number of elements in this object."""
        return len(self.data) * len(self.columns)

    @property
    def ndim(self) -> int:
        """Return int representing number of axes/array dimensions."""
        return 2

    def head(self, n: int = 5) -> Self:
        """Return first n rows."""
        if n >= 0:
            selected_data = self.data[:n]
        else:
            selected_data = self.data[:n] if n != 0 else self.data
        return self.__class__(columns=self.columns, data=selected_data)

    def tail(self, n: int = 5) -> Self:
        """Return last n rows."""
        if n >= 0:
            selected_data = self.data[-n:] if n > 0 else []
        else:
            selected_data = self.data[abs(n) :]
        return self.__class__(columns=self.columns, data=selected_data)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        *,
        random_state: int | None = None,
    ) -> Self:
        """Return random sample of rows."""
        # Validate parameters
        if n is None and frac is None:
            raise ValueError("Either n or frac must be provided")
        if n is not None and frac is not None:
            raise ValueError("n and frac are mutually exclusive")

        # Set random seed if provided
        if random_state is not None:
            random.seed(random_state)

        # Calculate sample size
        total_rows = len(self.data)
        if frac is not None:
            if frac > 1.0:
                raise ValueError("frac must be <= 1.0")
            sample_size = int(total_rows * frac)
        else:
            sample_size = min(n, total_rows) if n is not None else 0

        # Sample indices
        if sample_size >= total_rows:
            sampled_indices = list(range(total_rows))
            random.shuffle(sampled_indices)
        else:
            sampled_indices = random.sample(range(total_rows), sample_size)

        # Extract sampled rows
        sampled_data = [self.data[i] for i in sampled_indices]

        return self.__class__(columns=self.columns, data=sampled_data)

    def select(self, columns: list[str]) -> Self:
        """Return DataFrame with only specified columns."""
        # Validate all columns exist
        for col in columns:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Get column indices
        indices = [self.columns.index(col) for col in columns]

        # Extract data for selected columns
        new_data = [[row[i] for i in indices] for row in self.data]

        return self.__class__(columns=columns, data=new_data)

    def drop(self, columns: list[str]) -> Self:
        """Return DataFrame without specified columns."""
        # Validate all columns exist
        for col in columns:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Get columns to keep
        keep_cols = [c for c in self.columns if c not in columns]

        # Get indices for columns to keep
        indices = [self.columns.index(col) for col in keep_cols]

        # Extract data for kept columns
        new_data = [[row[i] for i in indices] for row in self.data]

        return self.__class__(columns=keep_cols, data=new_data)

    def rename(self, mapper: dict[str, str]) -> Self:
        """Return DataFrame with renamed columns."""
        # Validate all old column names exist
        for old_name in mapper:
            if old_name not in self.columns:
                raise KeyError(f"Column '{old_name}' not found in DataFrame")

        # Create new column list
        new_cols = [mapper.get(col, col) for col in self.columns]

        # Check for duplicates
        if len(new_cols) != len(set(new_cols)):
            raise ValueError("Renaming would create duplicate column names")

        return self.__class__(columns=new_cols, data=self.data)

    def rename_columns(self, mapper: dict[str, str]) -> Self:
        """Return DataFrame with renamed columns (alias for rename)."""
        return self.rename(mapper)

    def validate_structure(self) -> None:
        """Validate DataFrame structure."""
        # Check for empty column names
        for i, col in enumerate(self.columns):
            if col == "":
                raise ValueError(f"Column at index {i} is empty")

        # Check for duplicate column names
        if len(self.columns) != len(set(self.columns)):
            duplicates = [col for col in self.columns if self.columns.count(col) > 1]
            raise ValueError(f"Duplicate column names found: {set(duplicates)}")

        # Check all rows have same length as columns
        num_cols = len(self.columns)
        for i, row in enumerate(self.data):
            if len(row) != num_cols:
                raise ValueError(f"Row {i} has {len(row)} values, expected {num_cols}")

    def infer_types(self) -> dict[str, str]:
        """Infer column data types."""
        result: dict[str, str] = {}

        for col_idx, col_name in enumerate(self.columns):
            # Extract all values for this column
            values = [row[col_idx] for row in self.data]

            # Filter out None values for type checking
            non_null_values = [v for v in values if v is not None]

            if not non_null_values:
                result[col_name] = "null"
                continue

            # Check types
            types_found = set()
            for val in non_null_values:
                if isinstance(val, bool):
                    types_found.add("bool")
                elif isinstance(val, int):
                    types_found.add("int")
                elif isinstance(val, float):
                    types_found.add("float")
                elif isinstance(val, str):
                    types_found.add("str")
                else:
                    types_found.add("other")

            # Determine final type
            if len(types_found) > 1:
                # Special case: int and float can be treated as float
                if types_found == {"int", "float"}:
                    result[col_name] = "float"
                else:
                    result[col_name] = "mixed"
            elif "bool" in types_found:
                result[col_name] = "bool"
            elif "int" in types_found:
                result[col_name] = "int"
            elif "float" in types_found:
                result[col_name] = "float"
            elif "str" in types_found:
                result[col_name] = "str"
            else:
                result[col_name] = "mixed"

        return result

    def has_nulls(self) -> dict[str, bool]:
        """Check for null values in each column."""
        result: dict[str, bool] = {}

        for col_idx, col_name in enumerate(self.columns):
            # Check if any value in this column is None
            has_null = any(row[col_idx] is None for row in self.data)
            result[col_name] = has_null

        return result

    # Iteration and length

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.data)

    def __iter__(self) -> Any:
        """Iterate over rows as dictionaries."""
        for row in self.data:
            yield dict(zip(self.columns, row))

    # JSON support

    @classmethod
    def from_json(cls, json_string: str) -> Self:
        """Create DataFrame from JSON string (array of objects)."""
        records = json.loads(json_string)
        if not isinstance(records, list):
            raise ValueError("JSON must be an array of objects")
        return cls.from_records(records)

    def to_json(self, orient: Literal["records", "columns"] = "records") -> str:
        """Export DataFrame as JSON string."""
        # Map "columns" to "list" for to_dict()
        dict_orient: Literal["dict", "list", "records"] = "list" if orient == "columns" else orient
        return json.dumps(self.to_dict(orient=dict_orient))

    # Column access

    def get_column(self, column: str) -> list[Any]:
        """Get all values for a column."""
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        idx = self.columns.index(column)
        return [row[idx] for row in self.data]

    def __getitem__(self, key: str | list[str]) -> list[Any] | Self:
        """Support df['col'] and df[['col1', 'col2']]."""
        if isinstance(key, str):
            return self.get_column(key)
        return self.select(key)

    # Analytics methods

    def unique(self, column: str) -> list[Any]:
        """Get unique values from a column (preserves order)."""
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.columns.index(column)
        seen = set()
        result = []
        for row in self.data:
            val = row[col_idx]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result

    def value_counts(self, column: str) -> dict[Any, int]:
        """Count occurrences of each unique value in column."""
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.columns.index(column)
        counts: dict[Any, int] = {}
        for row in self.data:
            val = row[col_idx]
            counts[val] = counts.get(val, 0) + 1
        return counts

    def sort(self, by: str, ascending: bool = True) -> Self:
        """Sort DataFrame by column."""
        if by not in self.columns:
            raise KeyError(f"Column '{by}' not found in DataFrame")

        col_idx = self.columns.index(by)

        # Sort with None values at the end
        def sort_key(row: list[Any]) -> tuple[int, Any]:
            val = row[col_idx]
            if val is None:
                # Use a tuple to ensure None sorts last
                return (1, None) if ascending else (0, None)
            return (0, val) if ascending else (1, val)

        sorted_data = sorted(self.data, key=sort_key, reverse=not ascending)
        return self.__class__(columns=self.columns, data=sorted_data)

    # Row filtering and transformation

    def filter(self, predicate: Any) -> Self:
        """Filter rows using a predicate function."""
        filtered_data = []
        for row in self.data:
            row_dict = dict(zip(self.columns, row))
            if predicate(row_dict):
                filtered_data.append(row)
        return self.__class__(columns=self.columns, data=filtered_data)

    def apply(self, func: Any, column: str) -> Self:
        """Apply function to column values."""
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.columns.index(column)
        new_data = []
        for row in self.data:
            new_row = row.copy()
            new_row[col_idx] = func(row[col_idx])
            new_data.append(new_row)

        return self.__class__(columns=self.columns, data=new_data)

    def add_column(self, name: str, values: list[Any]) -> Self:
        """Add new column to DataFrame."""
        if name in self.columns:
            raise ValueError(f"Column '{name}' already exists")

        if len(values) != len(self.data):
            raise ValueError(f"Values length ({len(values)}) must match row count ({len(self.data)})")

        new_columns = self.columns + [name]
        new_data = [row + [values[i]] for i, row in enumerate(self.data)]

        return self.__class__(columns=new_columns, data=new_data)

    def drop_rows(self, indices: list[int]) -> Self:
        """Drop rows by index."""
        indices_set = set(indices)
        new_data = [row for i, row in enumerate(self.data) if i not in indices_set]
        return self.__class__(columns=self.columns, data=new_data)

    def drop_duplicates(self, subset: list[str] | None = None) -> Self:
        """Remove duplicate rows."""
        # Validate subset columns
        if subset is not None:
            for col in subset:
                if col not in self.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
            col_indices = [self.columns.index(col) for col in subset]
        else:
            col_indices = list(range(len(self.columns)))

        # Track seen values
        seen = set()
        new_data = []

        for row in self.data:
            # Create tuple of relevant column values
            key = tuple(row[i] for i in col_indices)

            if key not in seen:
                seen.add(key)
                new_data.append(row)

        return self.__class__(columns=self.columns, data=new_data)

    def fillna(self, value: Any | dict[str, Any]) -> Self:
        """Replace None values."""
        if isinstance(value, dict):
            # Validate column names
            for col in value:
                if col not in self.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")

            # Create mapping of column index to fill value
            fill_map = {self.columns.index(col): val for col, val in value.items()}

            # Fill values
            new_data = []
            for row in self.data:
                new_row = [fill_map[i] if i in fill_map and val is None else val for i, val in enumerate(row)]
                new_data.append(new_row)
        else:
            # Single fill value for all None
            new_data = [[value if val is None else val for val in row] for row in self.data]

        return self.__class__(columns=self.columns, data=new_data)

    def concat(self, other: Self) -> Self:
        """Concatenate DataFrames vertically (stack rows)."""
        if self.columns != other.columns:
            raise ValueError(f"Column mismatch: {self.columns} != {other.columns}")

        combined_data = self.data + other.data
        return self.__class__(columns=self.columns, data=combined_data)

    def melt(
        self,
        id_vars: list[str] | None = None,
        value_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        """Unpivot DataFrame from wide to long format."""
        # Handle empty DataFrame
        if not self.columns or not self.data:
            return self.__class__(columns=[var_name, value_name], data=[])

        # Default id_vars to empty list if not specified
        if id_vars is None:
            id_vars = []

        # Validate id_vars exist
        for col in id_vars:
            if col not in self.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Default value_vars to all non-id columns
        if value_vars is None:
            value_vars = [col for col in self.columns if col not in id_vars]
        else:
            # Validate value_vars exist
            for col in value_vars:
                if col not in self.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")

        # If no value_vars to melt, return empty result
        if not value_vars:
            # Return just id columns if all columns are id_vars
            if id_vars:
                return self.select(id_vars)
            return self.__class__(columns=[var_name, value_name], data=[])

        # Check for column name conflicts
        new_columns = id_vars + [var_name, value_name]
        if len(new_columns) != len(set(new_columns)):
            raise ValueError(
                f"Duplicate column names in result: {new_columns}. "
                f"Choose different var_name or value_name to avoid conflicts."
            )

        # Get indices for id and value columns
        id_indices = [self.columns.index(col) for col in id_vars]
        value_indices = [(self.columns.index(col), col) for col in value_vars]

        # Build melted data
        melted_data: list[list[Any]] = []

        for row in self.data:
            # Extract id values for this row
            id_values = [row[idx] for idx in id_indices]

            # Create one new row for each value_var
            for val_idx, var_col_name in value_indices:
                new_row = id_values + [var_col_name, row[val_idx]]
                melted_data.append(new_row)

        return self.__class__(columns=new_columns, data=melted_data)

    def pivot(self, index: str, columns: str, values: str) -> Self:
        """Pivot DataFrame from long to wide format."""
        # Validate columns exist
        for col_name, param in [(index, "index"), (columns, "columns"), (values, "values")]:
            if col_name not in self.columns:
                raise KeyError(f"Column '{col_name}' not found in DataFrame (parameter: {param})")

        # Get column indices
        index_idx = self.columns.index(index)
        columns_idx = self.columns.index(columns)
        values_idx = self.columns.index(values)

        # Build pivot structure: dict[index_value, dict[column_value, value]]
        pivot_dict: dict[Any, dict[Any, Any]] = {}
        column_values_set: set[Any] = set()

        for row in self.data:
            idx_val = row[index_idx]
            col_val = row[columns_idx]
            val = row[values_idx]

            # Track column values for final column list
            column_values_set.add(col_val)

            # Initialize nested dict if needed
            if idx_val not in pivot_dict:
                pivot_dict[idx_val] = {}

            # Check for duplicates
            if col_val in pivot_dict[idx_val]:
                raise ValueError(
                    f"Duplicate entries found for index='{idx_val}' and columns='{col_val}'. "
                    f"Cannot reshape with duplicate index/column combinations. "
                    f"Consider using aggregation or removing duplicates first."
                )

            pivot_dict[idx_val] = {**pivot_dict[idx_val], col_val: val}

        # Sort column values for consistent ordering
        column_values = sorted(column_values_set, key=lambda x: (x is None, x))

        # Build result columns: [index_column, col1, col2, ...]
        result_columns = [index] + column_values

        # Build result data
        result_data: list[list[Any]] = []
        for idx_val in sorted(pivot_dict.keys(), key=lambda x: (x is None, x)):
            row_dict = pivot_dict[idx_val]
            # Build row: [index_value, value_for_col1, value_for_col2, ...]
            row = [idx_val] + [row_dict.get(col_val, None) for col_val in column_values]
            result_data.append(row)

        return self.__class__(columns=result_columns, data=result_data)

    def merge(
        self,
        other: Self,
        on: str | list[str] | None = None,
        how: Literal["inner", "left", "right", "outer"] = "inner",
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Self:
        """Merge DataFrames using database-style join."""
        # Determine join keys
        if on is not None:
            if left_on is not None or right_on is not None:
                raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")
            left_keys = [on] if isinstance(on, str) else on
            right_keys = left_keys
        elif left_on is not None and right_on is not None:
            left_keys = [left_on] if isinstance(left_on, str) else left_on
            right_keys = [right_on] if isinstance(right_on, str) else right_on
            if len(left_keys) != len(right_keys):
                raise ValueError("left_on and right_on must have same length")
        else:
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

        # Validate join keys exist
        for key in left_keys:
            if key not in self.columns:
                raise KeyError(f"Join key '{key}' not found in left DataFrame")
        for key in right_keys:
            if key not in other.columns:
                raise KeyError(f"Join key '{key}' not found in right DataFrame")

        # Get indices for join keys
        left_key_indices = [self.columns.index(k) for k in left_keys]
        right_key_indices = [other.columns.index(k) for k in right_keys]

        # Build lookup dict for right DataFrame: key_tuple -> list[row_indices]
        right_lookup: dict[tuple[Any, ...], list[int]] = {}
        for row_idx, row in enumerate(other.data):
            key_tuple = tuple(row[idx] for idx in right_key_indices)
            if key_tuple not in right_lookup:
                right_lookup[key_tuple] = []
            right_lookup[key_tuple].append(row_idx)

        # Determine result columns
        left_suffix, right_suffix = suffixes

        # Start with left DataFrame columns
        result_columns = self.columns.copy()

        # Add right DataFrame columns (excluding join keys if using 'on')
        for col in other.columns:
            if on is not None and col in left_keys:
                # Skip join key columns from right when using 'on'
                continue

            if col in result_columns:
                # Handle collision with suffix
                result_columns.append(f"{col}{right_suffix}")
                # Also need to rename left column
                left_col_idx = result_columns.index(col)
                result_columns[left_col_idx] = f"{col}{left_suffix}"
            else:
                result_columns.append(col)

        # Get indices of right columns to include
        right_col_indices = []
        for col in other.columns:
            if on is not None and col in right_keys:
                continue
            right_col_indices.append(other.columns.index(col))

        # Perform join
        result_data: list[list[Any]] = []
        matched_right_indices: set[int] = set()

        for left_row in self.data:
            # Extract key from left row
            left_key_tuple = tuple(left_row[idx] for idx in left_key_indices)

            # Find matching rows in right DataFrame
            right_matches = right_lookup.get(left_key_tuple, [])

            if right_matches:
                # Join matched rows
                for right_idx in right_matches:
                    matched_right_indices.add(right_idx)
                    right_row = other.data[right_idx]

                    # Build result row: left columns + right columns (excluding join keys)
                    result_row = left_row.copy()
                    for col_idx in right_col_indices:
                        result_row.append(right_row[col_idx])

                    result_data.append(result_row)
            else:
                # No match
                if how in ("left", "outer"):
                    # Include left row with None for right columns
                    result_row = left_row.copy()
                    result_row.extend([None] * len(right_col_indices))
                    result_data.append(result_row)

        # Handle right/outer joins - add unmatched right rows
        if how in ("right", "outer"):
            for right_idx, right_row in enumerate(other.data):
                if right_idx not in matched_right_indices:
                    # Build row with None for left columns
                    result_row = [None] * len(self.columns)

                    # Fill in join key values if using 'on'
                    if on is not None:
                        for left_idx, right_idx_key in zip(left_key_indices, right_key_indices):
                            result_row[left_idx] = right_row[right_idx_key]

                    # Add right columns
                    for col_idx in right_col_indices:
                        result_row.append(right_row[col_idx])

                    result_data.append(result_row)

        return self.__class__(columns=result_columns, data=result_data)

    def transpose(self) -> Self:
        """Transpose DataFrame by swapping rows and columns."""
        if not self.data:
            # Empty DataFrame - return with swapped structure
            return self.__class__(columns=[], data=[])

        # First column becomes the new column names
        # Remaining columns become data rows
        if not self.columns:
            return self.__class__(columns=[], data=[])

        # Extract first column values as new column names
        # Convert to strings to ensure valid column names
        new_columns = [str(row[0]) for row in self.data]

        # Transpose the remaining columns
        num_original_cols = len(self.columns)
        if num_original_cols == 1:
            # Only one column (the index) - result is just column names as rows
            single_col_data = [[col] for col in self.columns]
            return self.__class__(columns=new_columns if new_columns else ["0"], data=single_col_data)

        # Build transposed data
        # Each original column (except first) becomes a row
        # Each original row becomes a column
        result_data: list[list[Any]] = []

        for col_idx in range(1, num_original_cols):
            # Original column name becomes first value in new row
            row = [self.columns[col_idx]]
            # Add values from each original row for this column
            for orig_row in self.data:
                row.append(orig_row[col_idx])
            result_data.append(row)

        # New columns: first is placeholder for original column names, rest are from first column
        result_columns = ["index"] + new_columns

        return self.__class__(columns=result_columns, data=result_data)

    # Statistical methods

    def describe(self) -> Self:
        """Generate statistical summary for numeric columns."""
        import statistics

        stats_rows: list[list[Any]] = []
        stat_names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

        for col_idx in range(len(self.columns)):
            # Extract numeric values (filter None and non-numeric)
            values = []
            for row in self.data:
                val = row[col_idx]
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                    values.append(float(val))

            if not values:
                # Non-numeric column - fill with None
                stats_rows.append([None] * len(stat_names))
                continue

            # Calculate statistics
            count = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values) if count > 1 else 0.0
            min_val = min(values)
            max_val = max(values)

            # Quantiles
            sorted_vals = sorted(values)
            try:
                q25 = statistics.quantiles(sorted_vals, n=4)[0] if count > 1 else sorted_vals[0]
                q50 = statistics.median(sorted_vals)
                q75 = statistics.quantiles(sorted_vals, n=4)[2] if count > 1 else sorted_vals[0]
            except statistics.StatisticsError:
                q25 = q50 = q75 = sorted_vals[0] if sorted_vals else 0.0

            stats_rows.append([count, mean, std, min_val, q25, q50, q75, max_val])

        # Transpose to make stats the rows and columns the columns
        transposed_data = [
            [stats_rows[col_idx][stat_idx] for col_idx in range(len(self.columns))]
            for stat_idx in range(len(stat_names))
        ]

        return self.__class__(columns=self.columns, data=transposed_data).add_column("stat", stat_names)

    def groupby(self, by: str) -> "GroupBy":
        """Group DataFrame by column values."""
        if by not in self.columns:
            raise KeyError(f"Column '{by}' not found in DataFrame")

        return GroupBy(self, by)

    # Utility methods

    def equals(self, other: Any) -> bool:
        """Check if two DataFrames are identical."""
        if not isinstance(other, DataFrame):
            return False
        return self.columns == other.columns and self.data == other.data

    def deepcopy(self) -> Self:
        """Create a deep copy of the DataFrame."""
        import copy

        return self.__class__(columns=self.columns.copy(), data=copy.deepcopy(self.data))

    def isna(self) -> Self:
        """Return DataFrame of booleans showing None locations."""
        null_data = [[val is None for val in row] for row in self.data]
        return self.__class__(columns=self.columns, data=null_data)

    def notna(self) -> Self:
        """Return DataFrame of booleans showing non-None locations."""
        not_null_data = [[val is not None for val in row] for row in self.data]
        return self.__class__(columns=self.columns, data=not_null_data)

    def dropna(self, axis: Literal[0, 1] = 0, how: Literal["any", "all"] = "any") -> Self:
        """Drop rows or columns with None values."""
        if axis == 0:
            # Drop rows
            if how == "any":
                # Drop rows with any None
                new_data = [row for row in self.data if not any(val is None for val in row)]
            else:
                # Drop rows with all None
                new_data = [row for row in self.data if not all(val is None for val in row)]
            return self.__class__(columns=self.columns, data=new_data)
        else:
            # Drop columns (axis=1)
            cols_to_keep = []
            indices_to_keep = []

            for col_idx, col_name in enumerate(self.columns):
                col_values = [row[col_idx] for row in self.data]

                if how == "any":
                    # Keep column if no None values
                    if not any(val is None for val in col_values):
                        cols_to_keep.append(col_name)
                        indices_to_keep.append(col_idx)
                else:
                    # Keep column if not all None
                    if not all(val is None for val in col_values):
                        cols_to_keep.append(col_name)
                        indices_to_keep.append(col_idx)

            # Extract data for kept columns
            new_data = [[row[i] for i in indices_to_keep] for row in self.data]
            return self.__class__(columns=cols_to_keep, data=new_data)

    def nunique(self, column: str) -> int:
        """Count number of unique values in column."""
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.columns.index(column)
        unique_values = set()
        for row in self.data:
            val = row[col_idx]
            # Count None as a unique value
            unique_values.add(val)
        return len(unique_values)


class GroupBy:
    """GroupBy helper for aggregations."""

    def __init__(self, dataframe: DataFrame, by: str):
        """Initialize GroupBy helper."""
        self.dataframe = dataframe
        self.by = by
        self.by_idx = dataframe.columns.index(by)

        # Build groups
        self.groups: dict[Any, list[list[Any]]] = {}
        for row in dataframe.data:
            key = row[self.by_idx]
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(row)

    def count(self) -> DataFrame:
        """Count rows per group."""
        data = [[key, len(rows)] for key, rows in self.groups.items()]
        return DataFrame(columns=[self.by, "count"], data=data)

    def sum(self, column: str) -> DataFrame:
        """Sum numeric column per group."""
        if column not in self.dataframe.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.dataframe.columns.index(column)
        data = []

        for key, rows in self.groups.items():
            values = [
                row[col_idx] for row in rows if row[col_idx] is not None and isinstance(row[col_idx], (int, float))
            ]
            total = sum(values) if values else None
            data.append([key, total])

        return DataFrame(columns=[self.by, f"{column}_sum"], data=data)

    def mean(self, column: str) -> DataFrame:
        """Calculate mean of numeric column per group."""
        if column not in self.dataframe.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        import statistics

        col_idx = self.dataframe.columns.index(column)
        data = []

        for key, rows in self.groups.items():
            values = [
                row[col_idx] for row in rows if row[col_idx] is not None and isinstance(row[col_idx], (int, float))
            ]
            avg = statistics.mean(values) if values else None
            data.append([key, avg])

        return DataFrame(columns=[self.by, f"{column}_mean"], data=data)

    def min(self, column: str) -> DataFrame:
        """Find minimum of numeric column per group."""
        if column not in self.dataframe.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.dataframe.columns.index(column)
        data = []

        for key, rows in self.groups.items():
            values = [
                row[col_idx] for row in rows if row[col_idx] is not None and isinstance(row[col_idx], (int, float))
            ]
            minimum = min(values) if values else None
            data.append([key, minimum])

        return DataFrame(columns=[self.by, f"{column}_min"], data=data)

    def max(self, column: str) -> DataFrame:
        """Find maximum of numeric column per group."""
        if column not in self.dataframe.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        col_idx = self.dataframe.columns.index(column)
        data = []

        for key, rows in self.groups.items():
            values = [
                row[col_idx] for row in rows if row[col_idx] is not None and isinstance(row[col_idx], (int, float))
            ]
            maximum = max(values) if values else None
            data.append([key, maximum])

        return DataFrame(columns=[self.by, f"{column}_max"], data=data)


__all__ = ["DataFrame", "GroupBy"]
