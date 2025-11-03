"""Tests for dataframe usage examples."""

import sys
from pathlib import Path

import pytest

# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "dataframe_usage"


class TestPandasExample:
    """Test pandas DataFrame conversion example."""

    def test_pandas_example(self) -> None:
        """Test pandas example script runs successfully."""
        pytest.importorskip("pandas", reason="pandas not installed")

        sys.path.insert(0, str(EXAMPLES_DIR))
        try:
            from pandas_example import df, df_back, sk_df  # type: ignore[import-not-found]

            # Verify conversion worked
            assert sk_df.columns == ["name", "age", "city"]
            assert len(sk_df.data) == 3
            assert sk_df.data[0] == ["Alice", 25, "New York"]

            # Verify round-trip conversion
            assert df_back.equals(df)
        finally:
            sys.path.pop(0)


class TestPolarsExample:
    """Test Polars DataFrame conversion example."""

    def test_polars_example(self) -> None:
        """Test polars example script runs successfully."""
        pytest.importorskip("polars", reason="polars not installed")

        sys.path.insert(0, str(EXAMPLES_DIR))
        try:
            from polars_example import df, df_back, sk_df  # type: ignore[import-not-found]

            # Verify conversion worked
            assert sk_df.columns == ["name", "age", "city"]
            assert len(sk_df.data) == 3
            assert sk_df.data[0] == ["Alice", 25, "New York"]

            # Verify round-trip conversion
            assert df_back.equals(df)
        finally:
            sys.path.pop(0)


class TestXarrayExample:
    """Test xarray DataArray conversion example."""

    def test_xarray_example(self) -> None:
        """Test xarray example script runs successfully."""
        pytest.importorskip("xarray", reason="xarray not installed")

        sys.path.insert(0, str(EXAMPLES_DIR))
        try:
            from xarray_example import df, sk_df  # type: ignore[import-not-found]

            # Verify conversion worked
            assert sk_df.columns == ["x", "y", "z"]
            assert len(sk_df.data) == 3
            assert sk_df.data[0] == [1, 2, 3]

            # Verify conversion to pandas worked
            assert list(df.columns) == ["x", "y", "z"]
            assert len(df) == 3
        finally:
            sys.path.pop(0)
