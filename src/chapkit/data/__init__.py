"""Common data schemas for chapkit services.

This module provides a universal DataFrame schema that works with
pandas, polars, xarray, and other data libraries.

Example:
    >>> from chapkit.data import DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> schema = DataFrame.from_pandas(df)
    >>> schema.to_polars()  # Convert to Polars
    >>> schema.to_dict()  # Convert to dict
"""

# ruff: noqa: F401

from .dataframe import DataFrame, DataFrameField, DataFrameSchema, GroupBy
from .generator import TestDataGenerator
from .geo import bounding_box

__all__ = [
    "DataFrame",
    "DataFrameField",
    "DataFrameSchema",
    "GroupBy",
    "TestDataGenerator",
    "bounding_box",
]
