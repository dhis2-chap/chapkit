"""Tests for DataFrame data interchange format."""

from pathlib import Path
from typing import Any

import pytest

from chapkit.data import DataFrame


class TestDataFrameFromDict:
    """Test DataFrame.from_dict() method."""

    def test_from_dict_basic(self) -> None:
        """Create DataFrame from dictionary."""
        data: dict[str, list[Any]] = {"name": ["Alice", "Bob"], "age": [25, 30]}
        df = DataFrame.from_dict(data)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_dict_empty(self) -> None:
        """Create DataFrame from empty dictionary."""
        df = DataFrame.from_dict({})

        assert df.columns == []
        assert df.data == []

    def test_from_dict_mismatched_lengths(self) -> None:
        """from_dict raises ValueError for mismatched column lengths."""
        data: dict[str, list[Any]] = {"name": ["Alice", "Bob"], "age": [25]}

        with pytest.raises(ValueError, match="All columns must have the same length"):
            DataFrame.from_dict(data)


class TestDataFrameFromRecords:
    """Test DataFrame.from_records() method."""

    def test_from_records_basic(self) -> None:
        """Create DataFrame from list of records."""
        records = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        df = DataFrame.from_records(records)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_records_empty(self) -> None:
        """Create DataFrame from empty list."""
        df = DataFrame.from_records([])

        assert df.columns == []
        assert df.data == []


class TestDataFrameToDict:
    """Test DataFrame.to_dict() method."""

    def test_to_dict_orient_dict(self) -> None:
        """Convert DataFrame to dict orient."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        result = df.to_dict(orient="dict")

        assert result == {"name": {0: "Alice", 1: "Bob"}, "age": {0: 25, 1: 30}}

    def test_to_dict_orient_list(self) -> None:
        """Convert DataFrame to list orient."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        result = df.to_dict(orient="list")

        assert result == {"name": ["Alice", "Bob"], "age": [25, 30]}

    def test_to_dict_orient_records(self) -> None:
        """Convert DataFrame to records orient."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        result = df.to_dict(orient="records")

        assert result == [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]

    def test_to_dict_invalid_orient(self) -> None:
        """to_dict raises ValueError for invalid orient."""
        df = DataFrame(columns=["name"], data=[["Alice"]])

        with pytest.raises(ValueError, match="Invalid orient"):
            df.to_dict(orient="invalid")  # type: ignore[arg-type]


class TestDataFrameImportErrors:
    """Test ImportError handling for optional dependencies."""

    def test_from_pandas_no_pandas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_pandas raises ImportError when pandas not installed."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            raise ImportError()

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="pandas is required"):
            DataFrame.from_pandas(None)

    def test_from_polars_no_polars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_polars raises ImportError when polars not installed."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            raise ImportError()

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="polars is required"):
            DataFrame.from_polars(None)

    def test_from_xarray_no_xarray(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_xarray raises ImportError when xarray not installed."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            raise ImportError()

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="xarray is required"):
            DataFrame.from_xarray(None)

    def test_to_pandas_no_pandas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """to_pandas raises ImportError when pandas not installed."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            raise ImportError()

        monkeypatch.setattr("builtins.__import__", mock_import)

        df = DataFrame(columns=["x"], data=[[1]])
        with pytest.raises(ImportError, match="pandas is required"):
            df.to_pandas()

    def test_to_polars_no_polars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """to_polars raises ImportError when polars not installed."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            raise ImportError()

        monkeypatch.setattr("builtins.__import__", mock_import)

        df = DataFrame(columns=["x"], data=[[1]])
        with pytest.raises(ImportError, match="polars is required"):
            df.to_polars()


class TestDataFrameTypeErrors:
    """Test TypeError handling for wrong input types."""

    def test_from_pandas_wrong_type(self) -> None:
        """from_pandas raises TypeError for non-DataFrame input."""
        pytest.importorskip("pandas")

        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            DataFrame.from_pandas("not a dataframe")

    def test_from_polars_wrong_type(self) -> None:
        """from_polars raises TypeError for non-DataFrame input."""
        pytest.importorskip("polars")

        with pytest.raises(TypeError, match="Expected Polars DataFrame"):
            DataFrame.from_polars("not a dataframe")

    def test_from_xarray_wrong_type(self) -> None:
        """from_xarray raises TypeError for non-DataArray input."""
        pytest.importorskip("xarray")

        with pytest.raises(TypeError, match="Expected xarray DataArray"):
            DataFrame.from_xarray("not a dataarray")

    def test_from_xarray_wrong_dimensions(self) -> None:
        """from_xarray raises ValueError for non-2D DataArray."""
        xr = pytest.importorskip("xarray")
        import numpy as np

        # Create 1D DataArray
        da = xr.DataArray(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="Only 2D DataArrays supported"):
            DataFrame.from_xarray(da)


class TestDataFrameProperties:
    """Test DataFrame utility properties."""

    def test_shape_basic(self) -> None:
        """shape property returns correct dimensions."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        assert df.shape == (2, 2)

    def test_shape_empty(self) -> None:
        """shape property handles empty DataFrame."""
        df = DataFrame(columns=[], data=[])
        assert df.shape == (0, 0)

    def test_shape_no_rows(self) -> None:
        """shape property with columns but no rows."""
        df = DataFrame(columns=["a", "b"], data=[])
        assert df.shape == (0, 2)

    def test_shape_rectangular(self) -> None:
        """shape property with different row and column counts."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3], [4, 5, 6]])
        assert df.shape == (2, 3)

    def test_empty_true_no_rows(self) -> None:
        """empty returns True when no rows."""
        df = DataFrame(columns=["a"], data=[])
        assert df.empty is True

    def test_empty_true_no_columns(self) -> None:
        """empty returns True when no columns."""
        df = DataFrame(columns=[], data=[])
        assert df.empty is True

    def test_empty_false_with_data(self) -> None:
        """empty returns False when DataFrame has data."""
        df = DataFrame(columns=["a"], data=[[1], [2]])
        assert df.empty is False

    def test_size_basic(self) -> None:
        """size property returns total element count."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        assert df.size == 4

    def test_size_empty(self) -> None:
        """size property returns 0 for empty DataFrame."""
        df = DataFrame(columns=[], data=[])
        assert df.size == 0

    def test_size_single_column(self) -> None:
        """size property with single column."""
        df = DataFrame(columns=["a"], data=[[1], [2], [3]])
        assert df.size == 3

    def test_size_rectangular(self) -> None:
        """size property with non-square DataFrame."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3], [4, 5, 6]])
        assert df.size == 6

    def test_ndim(self) -> None:
        """ndim property always returns 2."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])
        assert df.ndim == 2

    def test_ndim_empty(self) -> None:
        """ndim returns 2 even for empty DataFrame."""
        df = DataFrame(columns=[], data=[])
        assert df.ndim == 2


class TestDataFrameCSV:
    """Test DataFrame CSV methods."""

    def test_from_csv_string_basic(self) -> None:
        """Create DataFrame from CSV string with type inference."""
        csv_string = "name,age\nAlice,25\nBob,30"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_csv_string_no_header(self) -> None:
        """Create DataFrame from CSV string without header."""
        csv_string = "Alice,25\nBob,30"
        df = DataFrame.from_csv(csv_string=csv_string, has_header=False)

        assert df.columns == ["col_0", "col_1"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_csv_string_custom_delimiter(self) -> None:
        """Create DataFrame from CSV string with custom delimiter."""
        csv_string = "name;age\nAlice;25\nBob;30"
        df = DataFrame.from_csv(csv_string=csv_string, delimiter=";")

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_csv_string_empty(self) -> None:
        """Create DataFrame from empty CSV string."""
        df = DataFrame.from_csv(csv_string="")

        assert df.columns == []
        assert df.data == []

    def test_from_csv_file_basic(self, tmp_path: Path) -> None:
        """Create DataFrame from CSV file with type inference."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,25\nBob,30")

        df = DataFrame.from_csv(path=csv_file)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_csv_file_not_found(self) -> None:
        """from_csv raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            DataFrame.from_csv(path="nonexistent.csv")

    def test_from_csv_neither_path_nor_string(self) -> None:
        """from_csv raises ValueError when neither path nor csv_string provided."""
        with pytest.raises(ValueError, match="Either path or csv_string must be provided"):
            DataFrame.from_csv()

    def test_from_csv_both_path_and_string(self, tmp_path: Path) -> None:
        """from_csv raises ValueError when both path and csv_string provided."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2")

        with pytest.raises(ValueError, match="mutually exclusive"):
            DataFrame.from_csv(path=csv_file, csv_string="a,b\n1,2")

    def test_from_csv_encoding(self, tmp_path: Path) -> None:
        """Create DataFrame from CSV file with specific encoding."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,city\nAlice,Zürich", encoding="utf-8")

        df = DataFrame.from_csv(path=csv_file, encoding="utf-8")

        assert df.columns == ["name", "city"]
        assert df.data == [["Alice", "Zürich"]]

    def test_to_csv_string_basic(self) -> None:
        """Export DataFrame to CSV string."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        csv_string = df.to_csv()

        assert csv_string is not None
        assert "name,age" in csv_string
        assert "Alice,25" in csv_string
        assert "Bob,30" in csv_string

    def test_to_csv_string_no_header(self) -> None:
        """Export DataFrame to CSV string without header."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        csv_string = df.to_csv(include_header=False)

        assert csv_string is not None
        assert "name,age" not in csv_string
        assert "Alice,25" in csv_string
        assert "Bob,30" in csv_string

    def test_to_csv_string_custom_delimiter(self) -> None:
        """Export DataFrame to CSV string with custom delimiter."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        csv_string = df.to_csv(delimiter=";")

        assert csv_string is not None
        assert "name;age" in csv_string
        assert "Alice;25" in csv_string

    def test_to_csv_string_empty(self) -> None:
        """Export empty DataFrame to CSV string."""
        df = DataFrame(columns=[], data=[])
        csv_string = df.to_csv()

        # Empty DataFrame produces only a newline character
        assert csv_string is not None
        assert csv_string.strip() == ""

    def test_to_csv_file_basic(self, tmp_path: Path) -> None:
        """Export DataFrame to CSV file."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        csv_file = tmp_path / "output.csv"

        result = df.to_csv(path=csv_file)

        assert result is None
        assert csv_file.exists()

        content = csv_file.read_text()
        assert "name,age" in content
        assert "Alice,25" in content
        assert "Bob,30" in content

    def test_to_csv_file_encoding(self, tmp_path: Path) -> None:
        """Export DataFrame to CSV file with specific encoding."""
        df = DataFrame(columns=["name", "city"], data=[["Alice", "Zürich"]])
        csv_file = tmp_path / "output.csv"

        df.to_csv(path=csv_file, encoding="utf-8")

        content = csv_file.read_text(encoding="utf-8")
        assert "Zürich" in content

    def test_csv_roundtrip_string(self) -> None:
        """Round-trip DataFrame through CSV string preserves types via inference."""
        original = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])

        csv_string = original.to_csv()
        restored = DataFrame.from_csv(csv_string=csv_string)

        assert restored.columns == original.columns
        assert restored.data == [["Alice", 25], ["Bob", 30]]

    def test_csv_roundtrip_file(self, tmp_path: Path) -> None:
        """Round-trip DataFrame through CSV file preserves types via inference."""
        original = DataFrame(columns=["x", "y"], data=[[1, 2], [3, 4]])
        csv_file = tmp_path / "roundtrip.csv"

        original.to_csv(path=csv_file)
        restored = DataFrame.from_csv(path=csv_file)

        assert restored.columns == original.columns
        assert restored.data == [[1, 2], [3, 4]]


class TestDataFrameCSVTypeInference:
    """Test DataFrame CSV type inference."""

    def test_infer_integers(self) -> None:
        """Type inference converts integer strings to int."""
        csv_string = "id,count\n1,100\n2,200\n3,300"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[1, 100], [2, 200], [3, 300]]
        assert df.infer_types() == {"id": "int", "count": "int"}

    def test_infer_floats(self) -> None:
        """Type inference converts float strings to float."""
        csv_string = "price,discount\n19.99,0.15\n29.99,0.20"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[19.99, 0.15], [29.99, 0.20]]
        assert df.infer_types() == {"price": "float", "discount": "float"}

    def test_infer_mixed_int_float(self) -> None:
        """Type inference promotes int+float column to float."""
        csv_string = "value\n1\n2.5\n3"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[1.0], [2.5], [3.0]]
        assert df.infer_types() == {"value": "float"}

    def test_infer_booleans_true_false(self) -> None:
        """Type inference converts true/false to bool."""
        csv_string = "active\ntrue\nfalse\nTrue\nFalse"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[True], [False], [True], [False]]
        assert df.infer_types() == {"active": "bool"}

    def test_infer_booleans_yes_no(self) -> None:
        """Type inference converts yes/no to bool."""
        csv_string = "enabled\nyes\nno\nYes\nNo"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[True], [False], [True], [False]]
        assert df.infer_types() == {"enabled": "bool"}

    def test_infer_empty_to_none(self) -> None:
        """Type inference converts empty strings to None."""
        csv_string = "name,age\nAlice,25\n,30\nBob,"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [["Alice", 25], [None, 30], ["Bob", None]]

    def test_infer_whitespace_to_none(self) -> None:
        """Type inference converts whitespace-only strings to None."""
        csv_string = "value\n1\n   \n3"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[1], [None], [3]]

    def test_infer_mixed_types_fallback_string(self) -> None:
        """Type inference falls back to string for mixed types."""
        csv_string = "data\n1\nhello\n3"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [["1"], ["hello"], ["3"]]
        assert df.infer_types() == {"data": "str"}

    def test_infer_types_disabled(self) -> None:
        """Type inference can be disabled."""
        csv_string = "id,active\n1,true"
        df = DataFrame.from_csv(csv_string=csv_string, infer_types=False)

        assert df.data == [["1", "true"]]
        assert df.infer_types() == {"id": "str", "active": "str"}

    def test_infer_all_none_column(self) -> None:
        """Type inference handles all-empty columns."""
        csv_string = "a,b\n1,\n2,"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[1, None], [2, None]]

    def test_infer_scientific_notation(self) -> None:
        """Type inference handles scientific notation as float."""
        csv_string = "value\n1e10\n2.5e-3"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[1e10], [2.5e-3]]
        assert df.infer_types() == {"value": "float"}

    def test_infer_negative_numbers(self) -> None:
        """Type inference handles negative numbers."""
        csv_string = "value\n-1\n-2.5\n3"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [[-1.0], [-2.5], [3.0]]

    def test_infer_preserves_strings(self) -> None:
        """Type inference preserves string columns."""
        csv_string = "name,city\nAlice,NYC\nBob,LA"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [["Alice", "NYC"], ["Bob", "LA"]]
        assert df.infer_types() == {"name": "str", "city": "str"}

    def test_infer_with_none_in_numeric(self) -> None:
        """Type inference handles None mixed with numbers."""
        csv_string = "a,value\nx,1\ny,\nz,3"
        df = DataFrame.from_csv(csv_string=csv_string)

        assert df.data == [["x", 1], ["y", None], ["z", 3]]
        assert df.infer_types() == {"a": "str", "value": "int"}

    def test_infer_empty_dataframe(self) -> None:
        """Type inference handles empty CSV."""
        df = DataFrame.from_csv(csv_string="")
        assert df.columns == []
        assert df.data == []

    def test_infer_header_only(self) -> None:
        """Type inference handles CSV with only headers."""
        csv_string = "a,b,c"
        df = DataFrame.from_csv(csv_string=csv_string)
        assert df.columns == ["a", "b", "c"]
        assert df.data == []


class TestDataFrameInspection:
    """Test DataFrame data inspection methods."""

    def test_head_default(self) -> None:
        """head() returns first 5 rows by default."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(10)])
        result = df.head()

        assert result.columns == ["x"]
        assert result.data == [[0], [1], [2], [3], [4]]

    def test_head_custom_n(self) -> None:
        """head() returns first n rows."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(10)])
        result = df.head(3)

        assert result.data == [[0], [1], [2]]

    def test_head_negative(self) -> None:
        """head() with negative n returns all except last |n| rows."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(10)])
        result = df.head(-3)

        assert result.data == [[0], [1], [2], [3], [4], [5], [6]]

    def test_head_exceeds_length(self) -> None:
        """head() with n > row count returns all rows."""
        df = DataFrame(columns=["x"], data=[[1], [2]])
        result = df.head(100)

        assert result.data == [[1], [2]]

    def test_tail_default(self) -> None:
        """tail() returns last 5 rows by default."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(10)])
        result = df.tail()

        assert result.columns == ["x"]
        assert result.data == [[5], [6], [7], [8], [9]]

    def test_tail_custom_n(self) -> None:
        """tail() returns last n rows."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(10)])
        result = df.tail(3)

        assert result.data == [[7], [8], [9]]

    def test_tail_negative(self) -> None:
        """tail() with negative n returns all except first |n| rows."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(10)])
        result = df.tail(-3)

        assert result.data == [[3], [4], [5], [6], [7], [8], [9]]

    def test_tail_exceeds_length(self) -> None:
        """tail() with n > row count returns all rows."""
        df = DataFrame(columns=["x"], data=[[1], [2]])
        result = df.tail(100)

        assert result.data == [[1], [2]]

    def test_sample_n(self) -> None:
        """sample() with n returns specified number of rows."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(100)])
        result = df.sample(n=10, random_state=42)

        assert len(result.data) == 10
        assert result.columns == ["x"]

    def test_sample_frac(self) -> None:
        """sample() with frac returns fractional sample."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(100)])
        result = df.sample(frac=0.1, random_state=42)

        assert len(result.data) == 10

    def test_sample_random_state(self) -> None:
        """sample() with random_state gives reproducible results."""
        df = DataFrame(columns=["x"], data=[[i] for i in range(100)])

        result1 = df.sample(n=10, random_state=42)
        result2 = df.sample(n=10, random_state=42)

        assert result1.data == result2.data

    def test_sample_both_n_and_frac_error(self) -> None:
        """sample() raises ValueError when both n and frac provided."""
        df = DataFrame(columns=["x"], data=[[1], [2], [3]])

        with pytest.raises(ValueError, match="mutually exclusive"):
            df.sample(n=2, frac=0.5)

    def test_sample_neither_error(self) -> None:
        """sample() raises ValueError when neither n nor frac provided."""
        df = DataFrame(columns=["x"], data=[[1], [2], [3]])

        with pytest.raises(ValueError, match="Either n or frac must be provided"):
            df.sample()

    def test_sample_frac_too_large(self) -> None:
        """sample() raises ValueError when frac > 1.0."""
        df = DataFrame(columns=["x"], data=[[1], [2], [3]])

        with pytest.raises(ValueError, match="frac must be <= 1.0"):
            df.sample(frac=1.5)


class TestDataFrameColumnOps:
    """Test DataFrame column operation methods."""

    def test_select_basic(self) -> None:
        """select() returns DataFrame with only specified columns."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3], [4, 5, 6]])
        result = df.select(["a", "c"])

        assert result.columns == ["a", "c"]
        assert result.data == [[1, 3], [4, 6]]

    def test_select_single_column(self) -> None:
        """select() works with single column."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])
        result = df.select(["b"])

        assert result.columns == ["b"]
        assert result.data == [[2]]

    def test_select_column_not_found(self) -> None:
        """select() raises KeyError for non-existent column."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])

        with pytest.raises(KeyError, match="Column 'z' not found"):
            df.select(["a", "z"])

    def test_drop_basic(self) -> None:
        """drop() returns DataFrame without specified columns."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3], [4, 5, 6]])
        result = df.drop(["b"])

        assert result.columns == ["a", "c"]
        assert result.data == [[1, 3], [4, 6]]

    def test_drop_multiple_columns(self) -> None:
        """drop() can remove multiple columns."""
        df = DataFrame(columns=["a", "b", "c", "d"], data=[[1, 2, 3, 4]])
        result = df.drop(["b", "d"])

        assert result.columns == ["a", "c"]
        assert result.data == [[1, 3]]

    def test_drop_column_not_found(self) -> None:
        """drop() raises KeyError for non-existent column."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])

        with pytest.raises(KeyError, match="Column 'z' not found"):
            df.drop(["z"])

    def test_rename_basic(self) -> None:
        """rename() returns DataFrame with renamed columns."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])
        result = df.rename({"a": "x", "c": "z"})

        assert result.columns == ["x", "b", "z"]
        assert result.data == [[1, 2, 3]]

    def test_rename_partial(self) -> None:
        """rename() only renames specified columns."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])
        result = df.rename({"a": "x"})

        assert result.columns == ["x", "b", "c"]

    def test_rename_column_not_found(self) -> None:
        """rename() raises KeyError for non-existent column."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])

        with pytest.raises(KeyError, match="Column 'z' not found"):
            df.rename({"z": "new_name"})

    def test_rename_duplicate_names(self) -> None:
        """rename() raises ValueError if renaming creates duplicates."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])

        with pytest.raises(ValueError, match="duplicate column names"):
            df.rename({"a": "b"})

    def test_rename_columns_basic(self) -> None:
        """rename_columns() returns DataFrame with renamed columns."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])
        result = df.rename_columns({"a": "x", "c": "z"})

        assert result.columns == ["x", "b", "z"]
        assert result.data == [[1, 2, 3]]

    def test_rename_columns_partial(self) -> None:
        """rename_columns() only renames specified columns."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])
        result = df.rename_columns({"a": "x"})

        assert result.columns == ["x", "b", "c"]

    def test_rename_columns_column_not_found(self) -> None:
        """rename_columns() raises KeyError for non-existent column."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])

        with pytest.raises(KeyError, match="Column 'z' not found"):
            df.rename_columns({"z": "new_name"})

    def test_rename_columns_duplicate_names(self) -> None:
        """rename_columns() raises ValueError if renaming creates duplicates."""
        df = DataFrame(columns=["a", "b", "c"], data=[[1, 2, 3]])

        with pytest.raises(ValueError, match="duplicate column names"):
            df.rename_columns({"a": "b"})

    def test_rename_columns_immutability(self) -> None:
        """rename_columns() returns new instance without modifying original."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])
        result = df.rename_columns({"a": "x"})

        assert df.columns == ["a", "b"]
        assert result.columns == ["x", "b"]

    def test_rename_columns_empty_mapper(self) -> None:
        """rename_columns() with empty mapper returns DataFrame unchanged."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2]])
        result = df.rename_columns({})

        assert result.columns == ["a", "b"]
        assert result.data == [[1, 2]]


class TestDataFrameValidation:
    """Test DataFrame validation methods."""

    def test_validate_success(self) -> None:
        """validate_structure() succeeds for valid DataFrame."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        df.validate_structure()  # Should not raise

    def test_validate_unequal_row_lengths(self) -> None:
        """validate_structure() raises ValueError for rows with wrong length."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4, 5]])

        with pytest.raises(ValueError, match="Row 1 has 3 values, expected 2"):
            df.validate_structure()

    def test_validate_duplicate_columns(self) -> None:
        """validate_structure() raises ValueError for duplicate column names."""
        df = DataFrame(columns=["a", "a", "b"], data=[[1, 2, 3]])

        with pytest.raises(ValueError, match="Duplicate column names"):
            df.validate_structure()

    def test_validate_empty_column_name(self) -> None:
        """validate_structure() raises ValueError for empty column name."""
        df = DataFrame(columns=["a", "", "b"], data=[[1, 2, 3]])

        with pytest.raises(ValueError, match="Column at index 1 is empty"):
            df.validate_structure()

    def test_infer_types_basic(self) -> None:
        """infer_types() correctly identifies column types."""
        df = DataFrame(columns=["int_col", "float_col", "str_col", "bool_col"], data=[[1, 2.5, "hello", True]])

        result = df.infer_types()

        assert result["int_col"] == "int"
        assert result["float_col"] == "float"
        assert result["str_col"] == "str"
        assert result["bool_col"] == "bool"

    def test_infer_types_mixed(self) -> None:
        """infer_types() detects mixed types."""
        df = DataFrame(columns=["mixed_col"], data=[[1], ["hello"], [3]])

        result = df.infer_types()

        assert result["mixed_col"] == "mixed"

    def test_infer_types_int_and_float(self) -> None:
        """infer_types() treats int+float columns as float."""
        df = DataFrame(columns=["num_col"], data=[[1], [2.5], [3]])

        result = df.infer_types()

        assert result["num_col"] == "float"

    def test_infer_types_null(self) -> None:
        """infer_types() detects null columns."""
        df = DataFrame(columns=["null_col"], data=[[None], [None]])

        result = df.infer_types()

        assert result["null_col"] == "null"

    def test_has_nulls_true(self) -> None:
        """has_nulls() detects columns with None values."""
        df = DataFrame(columns=["a", "b"], data=[[1, None], [2, 3]])

        result = df.has_nulls()

        assert result["a"] is False
        assert result["b"] is True

    def test_has_nulls_false(self) -> None:
        """has_nulls() returns False for columns without None values."""
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])

        result = df.has_nulls()

        assert result["a"] is False
        assert result["b"] is False


class TestDataFrameIteration:
    """Test DataFrame iteration and length."""

    def test_len(self) -> None:
        """len() returns number of rows."""
        df = DataFrame(columns=["a"], data=[[1], [2], [3]])
        assert len(df) == 3

    def test_len_empty(self) -> None:
        """len() returns 0 for empty DataFrame."""
        df = DataFrame(columns=["a"], data=[])
        assert len(df) == 0

    def test_iter(self) -> None:
        """Iterate over rows as dictionaries."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])

        rows = list(df)

        assert rows == [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]

    def test_iter_empty(self) -> None:
        """Iterate over empty DataFrame."""
        df = DataFrame(columns=["a"], data=[])
        assert list(df) == []


class TestDataFrameJSON:
    """Test DataFrame JSON support."""

    def test_from_json_records(self) -> None:
        """Create DataFrame from JSON array."""
        json_string = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
        df = DataFrame.from_json(json_string)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_from_json_empty(self) -> None:
        """Create DataFrame from empty JSON array."""
        df = DataFrame.from_json("[]")
        assert df.columns == []
        assert df.data == []

    def test_from_json_not_array(self) -> None:
        """from_json raises ValueError for non-array JSON."""
        with pytest.raises(ValueError, match="JSON must be an array of objects"):
            DataFrame.from_json('{"a": 1}')

    def test_to_json_records(self) -> None:
        """Export DataFrame as JSON records."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])

        result = df.to_json(orient="records")

        assert result == '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'

    def test_to_json_columns(self) -> None:
        """Export DataFrame as JSON columns."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])

        result = df.to_json(orient="columns")

        assert result == '{"name": ["Alice", "Bob"], "age": [25, 30]}'

    def test_to_json_empty(self) -> None:
        """Export empty DataFrame as JSON."""
        df = DataFrame(columns=[], data=[])
        assert df.to_json() == "[]"

    def test_json_roundtrip(self) -> None:
        """Round-trip JSON conversion preserves data."""
        original = DataFrame(columns=["x", "y"], data=[[1, 2], [3, 4]])

        json_string = original.to_json()
        restored = DataFrame.from_json(json_string)

        assert restored.columns == original.columns
        assert restored.data == original.data


class TestDataFrameColumnAccess:
    """Test DataFrame column access methods."""

    def test_get_column(self) -> None:
        """get_column() returns column values."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])

        result = df.get_column("age")

        assert result == [25, 30]

    def test_get_column_not_found(self) -> None:
        """get_column() raises KeyError for missing column."""
        df = DataFrame(columns=["a"], data=[[1]])

        with pytest.raises(KeyError, match="Column 'b' not found"):
            df.get_column("b")

    def test_getitem_single_column(self) -> None:
        """df['col'] returns column values."""
        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])

        result = df["age"]

        assert result == [25, 30]

    def test_getitem_multiple_columns(self) -> None:
        """df[['col1', 'col2']] returns DataFrame."""
        df = DataFrame(columns=["name", "age", "city"], data=[["Alice", 25, "NYC"], ["Bob", 30, "LA"]])

        result = df[["name", "city"]]

        assert isinstance(result, DataFrame)
        assert result.columns == ["name", "city"]
        assert result.data == [["Alice", "NYC"], ["Bob", "LA"]]

    def test_getitem_not_found(self) -> None:
        """df['col'] raises KeyError for missing column."""
        df = DataFrame(columns=["a"], data=[[1]])

        with pytest.raises(KeyError, match="Column 'b' not found"):
            df["b"]


class TestDataFrameUnique:
    """Test DataFrame.unique() method."""

    def test_unique_basic(self) -> None:
        """unique() returns unique values."""
        df = DataFrame(columns=["category"], data=[["A"], ["B"], ["A"], ["C"], ["B"]])

        result = df.unique("category")

        assert result == ["A", "B", "C"]

    def test_unique_preserves_order(self) -> None:
        """unique() preserves order of first appearance."""
        df = DataFrame(columns=["x"], data=[[3], [1], [2], [1], [3]])

        result = df.unique("x")

        assert result == [3, 1, 2]

    def test_unique_with_none(self) -> None:
        """unique() includes None values."""
        df = DataFrame(columns=["x"], data=[[1], [None], [2], [None], [1]])

        result = df.unique("x")

        assert result == [1, None, 2]

    def test_unique_all_same(self) -> None:
        """unique() with all same values."""
        df = DataFrame(columns=["x"], data=[["A"], ["A"], ["A"]])
        assert df.unique("x") == ["A"]

    def test_unique_not_found(self) -> None:
        """unique() raises KeyError for missing column."""
        df = DataFrame(columns=["a"], data=[[1]])

        with pytest.raises(KeyError, match="Column 'b' not found"):
            df.unique("b")


class TestDataFrameValueCounts:
    """Test DataFrame.value_counts() method."""

    def test_value_counts_basic(self) -> None:
        """value_counts() returns counts."""
        df = DataFrame(columns=["category"], data=[["A"], ["B"], ["A"], ["C"], ["B"], ["A"]])

        result = df.value_counts("category")

        assert result == {"A": 3, "B": 2, "C": 1}

    def test_value_counts_with_none(self) -> None:
        """value_counts() counts None values."""
        df = DataFrame(columns=["x"], data=[[1], [None], [2], [None], [1]])

        result = df.value_counts("x")

        assert result == {1: 2, None: 2, 2: 1}

    def test_value_counts_single_value(self) -> None:
        """value_counts() with all same values."""
        df = DataFrame(columns=["x"], data=[["A"], ["A"], ["A"]])
        assert df.value_counts("x") == {"A": 3}

    def test_value_counts_empty(self) -> None:
        """value_counts() on empty DataFrame."""
        df = DataFrame(columns=["x"], data=[])
        assert df.value_counts("x") == {}

    def test_value_counts_not_found(self) -> None:
        """value_counts() raises KeyError for missing column."""
        df = DataFrame(columns=["a"], data=[[1]])

        with pytest.raises(KeyError, match="Column 'b' not found"):
            df.value_counts("b")


class TestDataFrameSort:
    """Test DataFrame.sort() method."""

    def test_sort_ascending(self) -> None:
        """sort() in ascending order."""
        df = DataFrame(columns=["name", "age"], data=[["Bob", 30], ["Alice", 25], ["Charlie", 35]])

        result = df.sort("age")

        assert result.data == [["Alice", 25], ["Bob", 30], ["Charlie", 35]]

    def test_sort_descending(self) -> None:
        """sort() in descending order."""
        df = DataFrame(columns=["name", "score"], data=[["Alice", 85], ["Bob", 95], ["Charlie", 90]])

        result = df.sort("score", ascending=False)

        assert result.data == [["Bob", 95], ["Charlie", 90], ["Alice", 85]]

    def test_sort_with_none_ascending(self) -> None:
        """sort() with None values (ascending)."""
        df = DataFrame(columns=["x"], data=[[3], [None], [1], [None], [2]])

        result = df.sort("x")

        # None values should be at the end
        assert result.data == [[1], [2], [3], [None], [None]]

    def test_sort_with_none_descending(self) -> None:
        """sort() with None values (descending)."""
        df = DataFrame(columns=["x"], data=[[3], [None], [1], [None], [2]])

        result = df.sort("x", ascending=False)

        # None values should still be at the end
        assert result.data == [[3], [2], [1], [None], [None]]

    def test_sort_strings(self) -> None:
        """sort() string values."""
        df = DataFrame(columns=["name"], data=[["Charlie"], ["Alice"], ["Bob"]])

        result = df.sort("name")

        assert result.data == [["Alice"], ["Bob"], ["Charlie"]]

    def test_sort_immutable(self) -> None:
        """sort() returns new DataFrame, doesn't modify original."""
        df = DataFrame(columns=["x"], data=[[3], [1], [2]])

        result = df.sort("x")

        assert df.data == [[3], [1], [2]]  # Original unchanged
        assert result.data == [[1], [2], [3]]

    def test_sort_not_found(self) -> None:
        """sort() raises KeyError for missing column."""
        df = DataFrame(columns=["a"], data=[[1]])

        with pytest.raises(KeyError, match="Column 'b' not found"):
            df.sort("b")


class TestDataFramePandasIntegration:
    """Test DataFrame integration with pandas."""

    def test_from_pandas_basic(self) -> None:
        """Convert pandas DataFrame to servicekit DataFrame."""
        pd = pytest.importorskip("pandas")

        pdf = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df = DataFrame.from_pandas(pdf)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_to_pandas_basic(self) -> None:
        """Convert servicekit DataFrame to pandas DataFrame."""
        pytest.importorskip("pandas")

        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        pdf = df.to_pandas()

        assert list(pdf.columns) == ["name", "age"]
        assert pdf.values.tolist() == [["Alice", 25], ["Bob", 30]]

    def test_pandas_roundtrip(self) -> None:
        """Round-trip conversion with pandas preserves data."""
        pd = pytest.importorskip("pandas")

        original_pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "z": ["a", "b", "c"]})

        df = DataFrame.from_pandas(original_pdf)
        restored_pdf = df.to_pandas()

        pd.testing.assert_frame_equal(original_pdf, restored_pdf)

    def test_pandas_with_none(self) -> None:
        """Pandas conversion handles None/NaN values."""
        pd = pytest.importorskip("pandas")
        import numpy as np

        pdf = pd.DataFrame({"a": [1, np.nan, 3], "b": ["x", None, "z"]})
        df = DataFrame.from_pandas(pdf)

        # np.nan stays as np.nan in the data
        assert isinstance(df.data[1][0], float) and np.isnan(df.data[1][0])
        # None in string columns becomes nan with pandas 3.0's default str dtype
        assert pd.isna(df.data[1][1])


class TestDataFramePolarsIntegration:
    """Test DataFrame integration with Polars."""

    def test_from_polars_basic(self) -> None:
        """Convert Polars DataFrame to servicekit DataFrame."""
        pl = pytest.importorskip("polars")

        pldf = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df = DataFrame.from_polars(pldf)

        assert df.columns == ["name", "age"]
        assert df.data == [["Alice", 25], ["Bob", 30]]

    def test_to_polars_basic(self) -> None:
        """Convert servicekit DataFrame to Polars DataFrame."""
        pytest.importorskip("polars")

        df = DataFrame(columns=["name", "age"], data=[["Alice", 25], ["Bob", 30]])
        pldf = df.to_polars()

        assert pldf.columns == ["name", "age"]
        assert pldf.rows() == [("Alice", 25), ("Bob", 30)]

    def test_polars_roundtrip(self) -> None:
        """Round-trip conversion with Polars preserves data."""
        pl = pytest.importorskip("polars")

        original_pldf = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "z": ["a", "b", "c"]})

        df = DataFrame.from_polars(original_pldf)
        restored_pldf = df.to_polars()

        assert original_pldf.equals(restored_pldf)

    def test_polars_with_none(self) -> None:
        """Polars conversion handles None values."""
        pl = pytest.importorskip("polars")

        pldf = pl.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        df = DataFrame.from_polars(pldf)

        assert df.data[1][0] is None
        assert df.data[1][1] is None


class TestDataFrameXarrayIntegration:
    """Test DataFrame integration with xarray."""

    def test_from_xarray_basic(self) -> None:
        """Convert xarray DataArray to servicekit DataFrame."""
        xr = pytest.importorskip("xarray")
        pytest.importorskip("pandas")
        import numpy as np

        # xarray with named coordinates produces string column names
        da = xr.DataArray(np.array([[1, 2], [3, 4], [5, 6]]), dims=["x", "y"], coords={"y": ["col1", "col2"]})
        df = DataFrame.from_xarray(da)

        assert df.shape == (3, 2)
        assert df.columns == ["col1", "col2"]
        assert len(df.data) == 3

    def test_xarray_via_pandas(self) -> None:
        """Xarray conversion goes through pandas."""
        xr = pytest.importorskip("xarray")
        pytest.importorskip("pandas")
        import numpy as np

        # Create 2D DataArray with named dimensions
        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]), dims=["row", "col"], coords={"row": [0, 1], "col": ["a", "b"]}
        )

        df = DataFrame.from_xarray(da)

        # Should have 2 rows (from row dimension) and 2 columns (from col dimension)
        assert df.shape == (2, 2)
        assert df.data == [[1.0, 2.0], [3.0, 4.0]]


class TestDataFrameFilter:
    """Test DataFrame filter() method."""

    def test_filter_basic(self) -> None:
        """Filter rows with simple predicate."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

        filtered = df.filter(lambda row: row["age"] > 25)

        assert filtered.shape == (2, 2)
        assert filtered.data == [["Bob", 30], ["Charlie", 35]]

    def test_filter_multiple_conditions(self) -> None:
        """Filter with multiple conditions."""
        df = DataFrame.from_dict(
            {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "active": [True, False, True]}
        )

        filtered = df.filter(lambda row: row["age"] >= 30 and row["active"])

        assert filtered.shape == (1, 3)
        assert filtered.data == [["Charlie", 35, True]]

    def test_filter_no_matches(self) -> None:
        """Filter returns empty DataFrame when no matches."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        filtered = df.filter(lambda row: row["age"] > 50)

        assert filtered.shape == (0, 2)
        assert filtered.data == []

    def test_filter_all_match(self) -> None:
        """Filter returns all rows when all match."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        filtered = df.filter(lambda row: row["age"] > 0)

        assert filtered.shape == df.shape
        assert filtered.data == df.data


class TestDataFrameApply:
    """Test DataFrame apply() method."""

    def test_apply_string_function(self) -> None:
        """Apply string transformation."""
        df = DataFrame.from_dict({"name": ["alice", "bob"], "age": [25, 30]})

        result = df.apply(str.upper, "name")

        assert result.columns == df.columns
        assert result.data == [["ALICE", 25], ["BOB", 30]]

    def test_apply_lambda(self) -> None:
        """Apply lambda function."""
        df = DataFrame.from_dict({"price": [10, 20, 30]})

        result = df.apply(lambda x: x * 2, "price")

        assert result.data == [[20], [40], [60]]

    def test_apply_nonexistent_column(self) -> None:
        """Apply to nonexistent column raises KeyError."""
        df = DataFrame.from_dict({"name": ["Alice"]})

        with pytest.raises(KeyError, match="Column 'age' not found"):
            df.apply(str.upper, "age")

    def test_apply_preserves_other_columns(self) -> None:
        """Apply only modifies target column."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        result = df.apply(str.upper, "name")

        # age column should be unchanged
        assert result.get_column("age") == [25, 30]


class TestDataFrameAddColumn:
    """Test DataFrame add_column() method."""

    def test_add_column_basic(self) -> None:
        """Add column to DataFrame."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        result = df.add_column("city", ["NYC", "LA"])

        assert result.columns == ["name", "age", "city"]
        assert result.data == [["Alice", 25, "NYC"], ["Bob", 30, "LA"]]

    def test_add_column_mismatched_length(self) -> None:
        """Add column with wrong length raises ValueError."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"]})

        with pytest.raises(ValueError, match="Values length .* must match row count"):
            df.add_column("city", ["NYC"])

    def test_add_column_existing_name(self) -> None:
        """Add column with existing name raises ValueError."""
        df = DataFrame.from_dict({"name": ["Alice"]})

        with pytest.raises(ValueError, match="Column 'name' already exists"):
            df.add_column("name", ["Bob"])

    def test_add_column_empty_dataframe(self) -> None:
        """Add column to empty DataFrame."""
        df = DataFrame(columns=[], data=[])

        result = df.add_column("name", [])

        assert result.columns == ["name"]
        assert result.data == []


class TestDataFrameDropRows:
    """Test DataFrame drop_rows() method."""

    def test_drop_rows_basic(self) -> None:
        """Drop rows by indices."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob", "Charlie", "Dave"], "age": [25, 30, 35, 40]})

        result = df.drop_rows([0, 2])

        assert result.shape == (2, 2)
        assert result.data == [["Bob", 30], ["Dave", 40]]

    def test_drop_rows_single(self) -> None:
        """Drop single row."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"]})

        result = df.drop_rows([0])

        assert result.shape == (1, 1)
        assert result.data == [["Bob"]]

    def test_drop_rows_none(self) -> None:
        """Drop no rows returns same data."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"]})

        result = df.drop_rows([])

        assert result.shape == df.shape
        assert result.data == df.data

    def test_drop_rows_out_of_range(self) -> None:
        """Drop rows with out of range indices (ignores invalid indices)."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"]})

        result = df.drop_rows([0, 5, 10])

        assert result.shape == (1, 1)
        assert result.data == [["Bob"]]


class TestDataFrameDropDuplicates:
    """Test DataFrame drop_duplicates() method."""

    def test_drop_duplicates_all_columns(self) -> None:
        """Drop duplicates considering all columns."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob", "Alice", "Charlie"], "age": [25, 30, 25, 35]})

        result = df.drop_duplicates()

        assert result.shape == (3, 2)
        assert result.data == [["Alice", 25], ["Bob", 30], ["Charlie", 35]]

    def test_drop_duplicates_subset(self) -> None:
        """Drop duplicates by specific columns."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob", "Alice"], "age": [25, 30, 35], "city": ["NYC", "LA", "SF"]})

        result = df.drop_duplicates(subset=["name"])

        # Should keep first Alice (age 25) and Bob
        assert result.shape == (2, 3)
        assert result.data == [["Alice", 25, "NYC"], ["Bob", 30, "LA"]]

    def test_drop_duplicates_no_duplicates(self) -> None:
        """Drop duplicates with no duplicates returns same data."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        result = df.drop_duplicates()

        assert result.shape == df.shape
        assert result.data == df.data

    def test_drop_duplicates_nonexistent_column(self) -> None:
        """Drop duplicates with nonexistent column raises KeyError."""
        df = DataFrame.from_dict({"name": ["Alice"]})

        with pytest.raises(KeyError, match="Column 'age' not found"):
            df.drop_duplicates(subset=["age"])


class TestDataFrameFillna:
    """Test DataFrame fillna() method."""

    def test_fillna_single_value(self) -> None:
        """Fill all None values with single value."""
        df = DataFrame.from_dict({"name": ["Alice", None, "Charlie"], "age": [25, None, 35]})

        result = df.fillna("Unknown")

        assert result.data == [["Alice", 25], ["Unknown", "Unknown"], ["Charlie", 35]]

    def test_fillna_dict(self) -> None:
        """Fill None values with column-specific values."""
        df = DataFrame.from_dict({"name": ["Alice", None], "age": [25, None]})

        result = df.fillna({"name": "Unknown", "age": 0})

        assert result.data == [["Alice", 25], ["Unknown", 0]]

    def test_fillna_partial_dict(self) -> None:
        """Fill only specified columns."""
        df = DataFrame.from_dict({"name": ["Alice", None], "age": [25, None]})

        result = df.fillna({"name": "Unknown"})

        assert result.data == [["Alice", 25], ["Unknown", None]]

    def test_fillna_no_nulls(self) -> None:
        """Fill on DataFrame with no nulls."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"]})

        result = df.fillna("Unknown")

        assert result.data == df.data

    def test_fillna_nonexistent_column(self) -> None:
        """Fill with nonexistent column raises KeyError."""
        df = DataFrame.from_dict({"name": ["Alice"]})

        with pytest.raises(KeyError, match="Column 'age' not found"):
            df.fillna({"age": 0})


class TestDataFrameConcat:
    """Test DataFrame concat() method."""

    def test_concat_basic(self) -> None:
        """Concatenate two DataFrames."""
        df1 = DataFrame.from_dict({"name": ["Alice"], "age": [25]})
        df2 = DataFrame.from_dict({"name": ["Bob"], "age": [30]})

        result = df1.concat(df2)

        assert result.shape == (2, 2)
        assert result.data == [["Alice", 25], ["Bob", 30]]

    def test_concat_multiple_rows(self) -> None:
        """Concatenate DataFrames with multiple rows."""
        df1 = DataFrame.from_dict({"x": [1, 2]})
        df2 = DataFrame.from_dict({"x": [3, 4]})

        result = df1.concat(df2)

        assert result.shape == (4, 1)
        assert result.data == [[1], [2], [3], [4]]

    def test_concat_mismatched_columns(self) -> None:
        """Concat with mismatched columns raises ValueError."""
        df1 = DataFrame.from_dict({"name": ["Alice"]})
        df2 = DataFrame.from_dict({"age": [30]})

        with pytest.raises(ValueError, match="Column mismatch"):
            df1.concat(df2)

    def test_concat_empty(self) -> None:
        """Concatenate with empty DataFrame."""
        df1 = DataFrame.from_dict({"name": ["Alice"]})
        df2 = DataFrame(columns=["name"], data=[])

        result = df1.concat(df2)

        assert result.shape == (1, 1)
        assert result.data == [["Alice"]]


class TestDataFrameDescribe:
    """Test DataFrame describe() method."""

    def test_describe_numeric(self) -> None:
        """Describe numeric columns."""
        df = DataFrame.from_dict({"age": [25, 30, 35, 40, 45]})

        result = df.describe()

        # Should have columns: age, stat
        assert "age" in result.columns
        assert "stat" in result.columns

        # Check stats are present
        stats = result.get_column("stat")
        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_describe_mixed_columns(self) -> None:
        """Describe with mixed numeric and non-numeric columns."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        result = df.describe()

        # Should have both columns
        assert "name" in result.columns
        assert "age" in result.columns

        # name column should have None stats
        name_col_idx = result.columns.index("name")
        for row in result.data:
            assert row[name_col_idx] is None

    def test_describe_empty(self) -> None:
        """Describe empty DataFrame."""
        df = DataFrame(columns=["age"], data=[])

        result = df.describe()

        assert "age" in result.columns
        assert "stat" in result.columns


class TestDataFrameGroupBy:
    """Test DataFrame groupby() method and GroupBy class."""

    def test_groupby_count(self) -> None:
        """Group by and count."""
        df = DataFrame.from_dict({"category": ["A", "B", "A", "C", "B"], "value": [10, 20, 30, 40, 50]})

        result = df.groupby("category").count()

        assert result.columns == ["category", "count"]
        assert result.shape[0] == 3  # Three unique categories

        # Check counts (order may vary)
        counts_dict = {row[0]: row[1] for row in result.data}
        assert counts_dict["A"] == 2
        assert counts_dict["B"] == 2
        assert counts_dict["C"] == 1

    def test_groupby_sum(self) -> None:
        """Group by and sum."""
        df = DataFrame.from_dict({"category": ["A", "B", "A"], "value": [10, 20, 30]})

        result = df.groupby("category").sum("value")

        assert result.columns == ["category", "value_sum"]

        # Check sums
        sums_dict = {row[0]: row[1] for row in result.data}
        assert sums_dict["A"] == 40
        assert sums_dict["B"] == 20

    def test_groupby_mean(self) -> None:
        """Group by and mean."""
        df = DataFrame.from_dict({"category": ["A", "B", "A"], "value": [10, 20, 30]})

        result = df.groupby("category").mean("value")

        assert result.columns == ["category", "value_mean"]

        # Check means
        means_dict = {row[0]: row[1] for row in result.data}
        assert means_dict["A"] == 20.0
        assert means_dict["B"] == 20.0

    def test_groupby_min_max(self) -> None:
        """Group by and find min/max."""
        df = DataFrame.from_dict({"category": ["A", "A", "B"], "value": [10, 30, 20]})

        min_result = df.groupby("category").min("value")
        max_result = df.groupby("category").max("value")

        min_dict = {row[0]: row[1] for row in min_result.data}
        max_dict = {row[0]: row[1] for row in max_result.data}

        assert min_dict["A"] == 10
        assert max_dict["A"] == 30
        assert min_dict["B"] == 20
        assert max_dict["B"] == 20

    def test_groupby_with_none(self) -> None:
        """Group by handles None values in aggregation column."""
        df = DataFrame.from_dict({"category": ["A", "A", "B"], "value": [10, None, 20]})

        result = df.groupby("category").sum("value")

        sums_dict = {row[0]: row[1] for row in result.data}
        assert sums_dict["A"] == 10  # Ignores None
        assert sums_dict["B"] == 20

    def test_groupby_nonexistent_column(self) -> None:
        """Group by nonexistent column raises KeyError."""
        df = DataFrame.from_dict({"name": ["Alice"]})

        with pytest.raises(KeyError, match="Column 'category' not found"):
            df.groupby("category")

    def test_groupby_agg_nonexistent_column(self) -> None:
        """GroupBy aggregation on nonexistent column raises KeyError."""
        df = DataFrame.from_dict({"category": ["A"], "value": [10]})

        with pytest.raises(KeyError, match="Column 'price' not found"):
            df.groupby("category").sum("price")


class TestDataFrameEquals:
    """Test DataFrame.equals() method."""

    def test_equals_identical(self) -> None:
        """Two identical DataFrames are equal."""
        df1 = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        df2 = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

        assert df1.equals(df2)

    def test_equals_same_instance(self) -> None:
        """DataFrame equals itself."""
        df = DataFrame.from_dict({"name": ["Alice"], "age": [25]})

        assert df.equals(df)

    def test_equals_different_data(self) -> None:
        """DataFrames with different data are not equal."""
        df1 = DataFrame.from_dict({"name": ["Alice"], "age": [25]})
        df2 = DataFrame.from_dict({"name": ["Bob"], "age": [30]})

        assert not df1.equals(df2)

    def test_equals_different_columns(self) -> None:
        """DataFrames with different columns are not equal."""
        df1 = DataFrame.from_dict({"name": ["Alice"], "age": [25]})
        df2 = DataFrame.from_dict({"name": ["Alice"], "score": [95]})

        assert not df1.equals(df2)

    def test_equals_different_order(self) -> None:
        """DataFrames with different row order are not equal."""
        df1 = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        df2 = DataFrame.from_dict({"name": ["Bob", "Alice"], "age": [30, 25]})

        assert not df1.equals(df2)

    def test_equals_empty_dataframes(self) -> None:
        """Two empty DataFrames are equal."""
        df1 = DataFrame.from_dict({})
        df2 = DataFrame.from_dict({})

        assert df1.equals(df2)

    def test_equals_non_dataframe(self) -> None:
        """DataFrame is not equal to non-DataFrame objects."""
        df = DataFrame.from_dict({"name": ["Alice"], "age": [25]})

        assert not df.equals({"name": ["Alice"], "age": [25]})
        assert not df.equals([["Alice", 25]])
        assert not df.equals(None)


class TestDataFrameDeepCopy:
    """Test DataFrame.deepcopy() method."""

    def test_deepcopy_basic(self) -> None:
        """Deepcopy creates identical but separate DataFrame."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        df_copy = df.deepcopy()

        assert df.equals(df_copy)
        assert df is not df_copy
        assert df.data is not df_copy.data
        assert df.columns is not df_copy.columns

    def test_deepcopy_mutation_independence(self) -> None:
        """Modifying copy doesn't affect original."""
        df = DataFrame.from_dict({"name": ["Alice"], "age": [25]})
        df_copy = df.deepcopy()

        # Modify copy's data
        df_copy.data[0][0] = "Bob"

        assert df.data[0][0] == "Alice"
        assert df_copy.data[0][0] == "Bob"

    def test_deepcopy_empty(self) -> None:
        """Deepcopy works with empty DataFrame."""
        df = DataFrame.from_dict({})
        df_copy = df.deepcopy()

        assert df.equals(df_copy)


class TestDataFrameIsna:
    """Test DataFrame.isna() method."""

    def test_isna_basic(self) -> None:
        """isna identifies None values."""
        df = DataFrame.from_dict({"a": [1, None, 3], "b": [None, 2, 3]})
        result = df.isna()

        expected = DataFrame.from_dict({"a": [False, True, False], "b": [True, False, False]})
        assert result.equals(expected)

    def test_isna_no_nulls(self) -> None:
        """isna returns all False when no None values."""
        df = DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.isna()

        expected = DataFrame.from_dict({"a": [False, False, False], "b": [False, False, False]})
        assert result.equals(expected)

    def test_isna_all_nulls(self) -> None:
        """isna returns all True when all None values."""
        df = DataFrame.from_dict({"a": [None, None], "b": [None, None]})
        result = df.isna()

        expected = DataFrame.from_dict({"a": [True, True], "b": [True, True]})
        assert result.equals(expected)

    def test_isna_empty(self) -> None:
        """isna works with empty DataFrame."""
        df = DataFrame.from_dict({})
        result = df.isna()

        assert result.equals(df)


class TestDataFrameNotna:
    """Test DataFrame.notna() method."""

    def test_notna_basic(self) -> None:
        """notna identifies non-None values."""
        df = DataFrame.from_dict({"a": [1, None, 3], "b": [None, 2, 3]})
        result = df.notna()

        expected = DataFrame.from_dict({"a": [True, False, True], "b": [False, True, True]})
        assert result.equals(expected)

    def test_notna_no_nulls(self) -> None:
        """notna returns all True when no None values."""
        df = DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.notna()

        expected = DataFrame.from_dict({"a": [True, True, True], "b": [True, True, True]})
        assert result.equals(expected)

    def test_notna_all_nulls(self) -> None:
        """notna returns all False when all None values."""
        df = DataFrame.from_dict({"a": [None, None], "b": [None, None]})
        result = df.notna()

        expected = DataFrame.from_dict({"a": [False, False], "b": [False, False]})
        assert result.equals(expected)


class TestDataFrameDropna:
    """Test DataFrame.dropna() method."""

    def test_dropna_rows_any(self) -> None:
        """dropna removes rows with any None values."""
        df = DataFrame.from_dict({"a": [1, None, 3, 4], "b": [5, 6, None, 8]})
        result = df.dropna(axis=0, how="any")

        expected = DataFrame.from_dict({"a": [1, 4], "b": [5, 8]})
        assert result.equals(expected)

    def test_dropna_rows_all(self) -> None:
        """dropna with how='all' removes only rows with all None."""
        df = DataFrame.from_dict({"a": [1, None, None, 4], "b": [5, 6, None, 8]})
        result = df.dropna(axis=0, how="all")

        expected = DataFrame.from_dict({"a": [1, None, 4], "b": [5, 6, 8]})
        assert result.equals(expected)

    def test_dropna_columns_any(self) -> None:
        """dropna removes columns with any None values."""
        df = DataFrame.from_dict({"a": [1, 2, 3], "b": [4, None, 6], "c": [7, 8, 9]})
        result = df.dropna(axis=1, how="any")

        expected = DataFrame.from_dict({"a": [1, 2, 3], "c": [7, 8, 9]})
        assert result.equals(expected)

    def test_dropna_columns_all(self) -> None:
        """dropna with how='all' removes only columns with all None."""
        df = DataFrame.from_dict({"a": [1, 2, 3], "b": [None, None, None], "c": [7, None, 9]})
        result = df.dropna(axis=1, how="all")

        expected = DataFrame.from_dict({"a": [1, 2, 3], "c": [7, None, 9]})
        assert result.equals(expected)

    def test_dropna_no_nulls(self) -> None:
        """dropna returns same DataFrame when no None values."""
        df = DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.dropna()

        assert result.equals(df)

    def test_dropna_all_rows_removed(self) -> None:
        """dropna can remove all rows."""
        df = DataFrame.from_dict({"a": [None, None], "b": [None, None]})
        result = df.dropna(axis=0, how="any")

        expected = DataFrame.from_dict({"a": [], "b": []})
        assert result.equals(expected)

    def test_dropna_all_columns_removed(self) -> None:
        """dropna can remove all columns."""
        df = DataFrame.from_dict({"a": [None, None], "b": [None, None]})
        result = df.dropna(axis=1, how="any")

        assert result.columns == []
        assert result.data == [[], []]


class TestDataFrameNunique:
    """Test DataFrame.nunique() method."""

    def test_nunique_basic(self) -> None:
        """nunique counts unique values."""
        df = DataFrame.from_dict({"category": ["A", "B", "A", "C", "B"]})

        assert df.nunique("category") == 3

    def test_nunique_all_unique(self) -> None:
        """nunique with all unique values."""
        df = DataFrame.from_dict({"id": [1, 2, 3, 4, 5]})

        assert df.nunique("id") == 5

    def test_nunique_all_same(self) -> None:
        """nunique with all same values."""
        df = DataFrame.from_dict({"status": ["active", "active", "active"]})

        assert df.nunique("status") == 1

    def test_nunique_with_none(self) -> None:
        """nunique counts None as a unique value."""
        df = DataFrame.from_dict({"values": [1, None, 1, None, 2]})

        assert df.nunique("values") == 3

    def test_nunique_single_value(self) -> None:
        """nunique with single row."""
        df = DataFrame.from_dict({"value": [42]})

        assert df.nunique("value") == 1

    def test_nunique_empty(self) -> None:
        """nunique with empty column."""
        df = DataFrame.from_dict({"value": []})

        assert df.nunique("value") == 0

    def test_nunique_nonexistent_column(self) -> None:
        """nunique raises KeyError for nonexistent column."""
        df = DataFrame.from_dict({"a": [1, 2, 3]})

        with pytest.raises(KeyError, match="Column 'b' not found"):
            df.nunique("b")


class TestDataFrameMelt:
    """Test DataFrame.melt() method."""

    def test_melt_basic(self) -> None:
        """Melt DataFrame with id_vars and value_vars."""
        df = DataFrame.from_dict(
            {
                "name": ["Alice", "Bob"],
                "math": [90, 78],
                "science": [85, 92],
            }
        )

        melted = df.melt(id_vars=["name"], value_vars=["math", "science"])

        assert melted.columns == ["name", "variable", "value"]
        assert melted.data == [
            ["Alice", "math", 90],
            ["Alice", "science", 85],
            ["Bob", "math", 78],
            ["Bob", "science", 92],
        ]

    def test_melt_custom_names(self) -> None:
        """Melt with custom var_name and value_name."""
        df = DataFrame.from_dict(
            {
                "product": ["Widget", "Gadget"],
                "q1": [1000, 800],
                "q2": [1100, 850],
            }
        )

        melted = df.melt(
            id_vars=["product"],
            value_vars=["q1", "q2"],
            var_name="quarter",
            value_name="sales",
        )

        assert melted.columns == ["product", "quarter", "sales"]
        assert melted.data == [
            ["Widget", "q1", 1000],
            ["Widget", "q2", 1100],
            ["Gadget", "q1", 800],
            ["Gadget", "q2", 850],
        ]

    def test_melt_no_id_vars(self) -> None:
        """Melt without id_vars."""
        df = DataFrame.from_dict(
            {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            }
        )

        melted = df.melt(value_vars=["a", "b"])

        assert melted.columns == ["variable", "value"]
        # Melt processes row-by-row: for each row, create entries for each value_var
        assert melted.data == [
            ["a", 1],  # row 0, column a
            ["b", 3],  # row 0, column b
            ["a", 2],  # row 1, column a
            ["b", 4],  # row 1, column b
        ]

    def test_melt_no_value_vars(self) -> None:
        """Melt with no value_vars defaults to all non-id columns."""
        df = DataFrame.from_dict(
            {
                "id": [1, 2],
                "x": [10, 20],
                "y": [30, 40],
            }
        )

        melted = df.melt(id_vars=["id"])

        assert melted.columns == ["id", "variable", "value"]
        assert melted.data == [
            [1, "x", 10],
            [1, "y", 30],
            [2, "x", 20],
            [2, "y", 40],
        ]

    def test_melt_multiple_id_vars(self) -> None:
        """Melt with multiple id_vars."""
        df = DataFrame.from_dict(
            {
                "region": ["North", "South"],
                "product": ["Widget", "Gadget"],
                "jan": [1000, 800],
                "feb": [1100, 850],
            }
        )

        melted = df.melt(
            id_vars=["region", "product"],
            value_vars=["jan", "feb"],
        )

        assert melted.columns == ["region", "product", "variable", "value"]
        assert melted.data == [
            ["North", "Widget", "jan", 1000],
            ["North", "Widget", "feb", 1100],
            ["South", "Gadget", "jan", 800],
            ["South", "Gadget", "feb", 850],
        ]

    def test_melt_with_none_values(self) -> None:
        """Melt preserves None values."""
        df = DataFrame.from_dict(
            {
                "id": [1, 2],
                "a": [10, None],
                "b": [None, 40],
            }
        )

        melted = df.melt(id_vars=["id"], value_vars=["a", "b"])

        assert melted.data == [
            [1, "a", 10],
            [1, "b", None],
            [2, "a", None],
            [2, "b", 40],
        ]

    def test_melt_empty_dataframe(self) -> None:
        """Melt empty DataFrame."""
        df = DataFrame.from_dict({})

        melted = df.melt()

        assert melted.columns == ["variable", "value"]
        assert melted.data == []

    def test_melt_all_columns_as_id_vars(self) -> None:
        """Melt with all columns as id_vars."""
        df = DataFrame.from_dict(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )

        melted = df.melt(id_vars=["a", "b"])

        assert melted.columns == ["a", "b"]
        assert melted.data == df.data

    def test_melt_no_value_vars_no_id_vars(self) -> None:
        """Melt with no value_vars and no id_vars returns empty with default columns."""
        df = DataFrame.from_dict({"a": [1, 2]})

        # All columns as id_vars, no value_vars specified
        melted = df.melt(id_vars=["a"], value_vars=[])

        # Should return just the id column since no value_vars to melt
        assert melted.columns == ["a"]
        assert melted.data == [[1], [2]]

    def test_melt_single_value_var(self) -> None:
        """Melt with single value_var."""
        df = DataFrame.from_dict(
            {
                "name": ["Alice", "Bob"],
                "score": [90, 85],
            }
        )

        melted = df.melt(id_vars=["name"], value_vars=["score"])

        assert melted.columns == ["name", "variable", "value"]
        assert melted.data == [
            ["Alice", "score", 90],
            ["Bob", "score", 85],
        ]

    def test_melt_nonexistent_id_var(self) -> None:
        """Melt raises KeyError for nonexistent id_var."""
        df = DataFrame.from_dict({"a": [1, 2], "b": [3, 4]})

        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            df.melt(id_vars=["nonexistent"])

    def test_melt_nonexistent_value_var(self) -> None:
        """Melt raises KeyError for nonexistent value_var."""
        df = DataFrame.from_dict({"a": [1, 2], "b": [3, 4]})

        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            df.melt(value_vars=["nonexistent"])

    def test_melt_column_name_conflict(self) -> None:
        """Melt raises ValueError when var_name conflicts with id_vars."""
        df = DataFrame.from_dict(
            {
                "id": [1, 2],
                "variable": [10, 20],
                "x": [30, 40],
            }
        )

        with pytest.raises(ValueError, match="Duplicate column names"):
            df.melt(id_vars=["id", "variable"], value_vars=["x"])

    def test_melt_single_row(self) -> None:
        """Melt DataFrame with single row."""
        df = DataFrame.from_dict(
            {
                "id": [1],
                "a": [10],
                "b": [20],
            }
        )

        melted = df.melt(id_vars=["id"], value_vars=["a", "b"])

        assert melted.data == [
            [1, "a", 10],
            [1, "b", 20],
        ]

    def test_melt_preserves_order(self) -> None:
        """Melt preserves row order and value_var order."""
        df = DataFrame.from_dict(
            {
                "id": [3, 1, 2],
                "z": [30, 10, 20],
                "y": [33, 11, 22],
                "x": [36, 12, 24],
            }
        )

        melted = df.melt(id_vars=["id"], value_vars=["z", "y", "x"])

        # Should process rows in order: 3, 1, 2
        # Should process columns in order: z, y, x
        assert melted.data[0] == [3, "z", 30]
        assert melted.data[1] == [3, "y", 33]
        assert melted.data[2] == [3, "x", 36]
        assert melted.data[3] == [1, "z", 10]

    def test_melt_mixed_types(self) -> None:
        """Melt with mixed data types."""
        df = DataFrame.from_dict(
            {
                "name": ["Alice", "Bob"],
                "score": [90, 85],
                "passed": [True, False],
                "grade": ["A", "B"],
            }
        )

        melted = df.melt(id_vars=["name"], value_vars=["score", "passed", "grade"])

        assert melted.columns == ["name", "variable", "value"]
        assert len(melted.data) == 6  # 2 rows * 3 value_vars
        assert melted.data[0] == ["Alice", "score", 90]
        assert melted.data[1] == ["Alice", "passed", True]
        assert melted.data[2] == ["Alice", "grade", "A"]


class TestDataFramePivot:
    """Test DataFrame.pivot() method."""

    def test_pivot_basic(self) -> None:
        """Pivot DataFrame from long to wide format."""
        df = DataFrame.from_dict(
            {
                "date": ["2024-01", "2024-01", "2024-02", "2024-02"],
                "metric": ["sales", "profit", "sales", "profit"],
                "value": [1000, 200, 1100, 220],
            }
        )

        pivoted = df.pivot(index="date", columns="metric", values="value")

        assert pivoted.columns == ["date", "profit", "sales"]
        assert pivoted.data == [
            ["2024-01", 200, 1000],
            ["2024-02", 220, 1100],
        ]

    def test_pivot_student_grades(self) -> None:
        """Pivot student grades from long to wide format."""
        df_long = DataFrame.from_dict(
            {
                "student": ["Alice", "Alice", "Alice", "Bob", "Bob", "Bob"],
                "subject": ["math", "science", "history", "math", "science", "history"],
                "score": [90, 85, 88, 78, 92, 81],
            }
        )

        df_wide = df_long.pivot(index="student", columns="subject", values="score")

        assert df_wide.columns == ["student", "history", "math", "science"]
        assert df_wide.data == [
            ["Alice", 88, 90, 85],
            ["Bob", 81, 78, 92],
        ]

    def test_pivot_with_none_values(self) -> None:
        """Pivot handles None values in data."""
        df = DataFrame.from_dict(
            {
                "id": [1, 1, 2, 2],
                "category": ["a", "b", "a", "b"],
                "value": [10, None, 20, 30],
            }
        )

        pivoted = df.pivot(index="id", columns="category", values="value")

        assert pivoted.data == [
            [1, 10, None],
            [2, 20, 30],
        ]

    def test_pivot_sparse_data(self) -> None:
        """Pivot with missing combinations fills None."""
        df = DataFrame.from_dict(
            {
                "id": [1, 1, 2],
                "category": ["a", "b", "a"],
                "value": [10, 20, 30],
            }
        )

        pivoted = df.pivot(index="id", columns="category", values="value")

        assert pivoted.columns == ["id", "a", "b"]
        assert pivoted.data == [
            [1, 10, 20],
            [2, 30, None],  # Missing 'b' for id=2
        ]

    def test_pivot_numeric_index(self) -> None:
        """Pivot with numeric index values."""
        df = DataFrame.from_dict(
            {
                "week": [1, 1, 2, 2],
                "day": ["mon", "tue", "mon", "tue"],
                "hours": [8, 7, 9, 8],
            }
        )

        pivoted = df.pivot(index="week", columns="day", values="hours")

        assert pivoted.columns == ["week", "mon", "tue"]
        assert pivoted.data == [
            [1, 8, 7],
            [2, 9, 8],
        ]

    def test_pivot_duplicate_error(self) -> None:
        """Pivot raises ValueError for duplicate index/column combinations."""
        df = DataFrame.from_dict(
            {
                "id": [1, 1, 2],
                "category": ["a", "a", "b"],  # Duplicate: id=1, category=a
                "value": [10, 20, 30],
            }
        )

        with pytest.raises(ValueError, match="Duplicate entries found"):
            df.pivot(index="id", columns="category", values="value")

    def test_pivot_nonexistent_index(self) -> None:
        """Pivot raises KeyError for nonexistent index column."""
        df = DataFrame.from_dict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        with pytest.raises(KeyError, match="Column 'nonexistent' not found.*parameter: index"):
            df.pivot(index="nonexistent", columns="b", values="c")

    def test_pivot_nonexistent_columns(self) -> None:
        """Pivot raises KeyError for nonexistent columns parameter."""
        df = DataFrame.from_dict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        with pytest.raises(KeyError, match="Column 'nonexistent' not found.*parameter: columns"):
            df.pivot(index="a", columns="nonexistent", values="c")

    def test_pivot_nonexistent_values(self) -> None:
        """Pivot raises KeyError for nonexistent values parameter."""
        df = DataFrame.from_dict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        with pytest.raises(KeyError, match="Column 'nonexistent' not found.*parameter: values"):
            df.pivot(index="a", columns="b", values="nonexistent")

    def test_pivot_single_row(self) -> None:
        """Pivot with single row."""
        df = DataFrame.from_dict(
            {
                "id": [1, 1, 1],
                "metric": ["a", "b", "c"],
                "value": [10, 20, 30],
            }
        )

        pivoted = df.pivot(index="id", columns="metric", values="value")

        assert pivoted.data == [[1, 10, 20, 30]]

    def test_pivot_mixed_types(self) -> None:
        """Pivot with mixed data types."""
        df = DataFrame.from_dict(
            {
                "id": ["x", "x", "y", "y"],
                "field": ["count", "active", "count", "active"],
                "value": [5, True, 10, False],
            }
        )

        pivoted = df.pivot(index="id", columns="field", values="value")

        assert pivoted.columns == ["id", "active", "count"]
        assert pivoted.data == [
            ["x", True, 5],
            ["y", False, 10],
        ]

    def test_pivot_column_ordering(self) -> None:
        """Pivot sorts columns consistently."""
        df = DataFrame.from_dict(
            {
                "id": [1, 1, 1, 1],
                "category": ["d", "b", "c", "a"],
                "value": [4, 2, 3, 1],
            }
        )

        pivoted = df.pivot(index="id", columns="category", values="value")

        # Columns should be sorted
        assert pivoted.columns == ["id", "a", "b", "c", "d"]
        assert pivoted.data == [[1, 1, 2, 3, 4]]


class TestDataFrameMerge:
    """Test DataFrame.merge() method."""

    def test_merge_inner_basic(self) -> None:
        """Inner join keeps only matching rows."""
        left = DataFrame.from_dict({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"key": [1, 2, 4], "right_val": ["x", "y", "z"]})

        result = left.merge(right, on="key", how="inner")

        assert result.columns == ["key", "left_val", "right_val"]
        assert result.data == [
            [1, "a", "x"],
            [2, "b", "y"],
        ]

    def test_merge_left(self) -> None:
        """Left join keeps all left rows."""
        left = DataFrame.from_dict({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"key": [1, 2, 4], "right_val": ["x", "y", "z"]})

        result = left.merge(right, on="key", how="left")

        assert result.columns == ["key", "left_val", "right_val"]
        assert result.data == [
            [1, "a", "x"],
            [2, "b", "y"],
            [3, "c", None],  # No match in right
        ]

    def test_merge_right(self) -> None:
        """Right join keeps all right rows."""
        left = DataFrame.from_dict({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"key": [1, 2, 4], "right_val": ["x", "y", "z"]})

        result = left.merge(right, on="key", how="right")

        assert result.columns == ["key", "left_val", "right_val"]
        assert result.data == [
            [1, "a", "x"],
            [2, "b", "y"],
            [4, None, "z"],  # No match in left
        ]

    def test_merge_outer(self) -> None:
        """Outer join keeps all rows from both DataFrames."""
        left = DataFrame.from_dict({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"key": [1, 2, 4], "right_val": ["x", "y", "z"]})

        result = left.merge(right, on="key", how="outer")

        assert result.columns == ["key", "left_val", "right_val"]
        assert sorted(result.data, key=lambda x: x[0]) == [
            [1, "a", "x"],
            [2, "b", "y"],
            [3, "c", None],
            [4, None, "z"],
        ]

    def test_merge_multiple_matches(self) -> None:
        """Merge handles one-to-many relationships."""
        users = DataFrame.from_dict({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        orders = DataFrame.from_dict({"user_id": [1, 1, 2, 3], "amount": [100, 150, 200, 75]})

        result = orders.merge(users, on="user_id", how="left")

        assert result.columns == ["user_id", "amount", "name"]
        assert result.data == [
            [1, 100, "Alice"],
            [1, 150, "Alice"],
            [2, 200, "Bob"],
            [3, 75, "Charlie"],
        ]

    def test_merge_left_on_right_on(self) -> None:
        """Merge with different column names."""
        left = DataFrame.from_dict({"left_key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"right_key": [1, 2, 4], "right_val": ["x", "y", "z"]})

        result = left.merge(right, left_on="left_key", right_on="right_key", how="inner")

        assert result.columns == ["left_key", "left_val", "right_key", "right_val"]
        assert result.data == [
            [1, "a", 1, "x"],
            [2, "b", 2, "y"],
        ]

    def test_merge_multiple_keys(self) -> None:
        """Merge on multiple columns."""
        left = DataFrame.from_dict({"key1": [1, 1, 2], "key2": ["a", "b", "a"], "left_val": [10, 20, 30]})
        right = DataFrame.from_dict({"key1": [1, 1, 2], "key2": ["a", "b", "b"], "right_val": [100, 200, 300]})

        result = left.merge(right, on=["key1", "key2"], how="inner")

        assert result.columns == ["key1", "key2", "left_val", "right_val"]
        assert result.data == [
            [1, "a", 10, 100],
            [1, "b", 20, 200],
        ]

    def test_merge_column_name_collision(self) -> None:
        """Merge handles column name collisions with suffixes."""
        left = DataFrame.from_dict({"key": [1, 2], "value": [10, 20]})
        right = DataFrame.from_dict({"key": [1, 2], "value": [100, 200]})

        result = left.merge(right, on="key", how="inner")

        assert result.columns == ["key", "value_x", "value_y"]
        assert result.data == [
            [1, 10, 100],
            [2, 20, 200],
        ]

    def test_merge_custom_suffixes(self) -> None:
        """Merge with custom suffixes."""
        left = DataFrame.from_dict({"key": [1, 2], "value": [10, 20]})
        right = DataFrame.from_dict({"key": [1, 2], "value": [100, 200]})

        result = left.merge(right, on="key", how="inner", suffixes=("_left", "_right"))

        assert result.columns == ["key", "value_left", "value_right"]
        assert result.data == [
            [1, 10, 100],
            [2, 20, 200],
        ]

    def test_merge_empty_left(self) -> None:
        """Merge with empty left DataFrame."""
        left = DataFrame.from_dict({"key": [], "left_val": []})
        right = DataFrame.from_dict({"key": [1, 2], "right_val": ["x", "y"]})

        result = left.merge(right, on="key", how="inner")

        assert result.columns == ["key", "left_val", "right_val"]
        assert result.data == []

    def test_merge_empty_right(self) -> None:
        """Merge with empty right DataFrame."""
        left = DataFrame.from_dict({"key": [1, 2], "left_val": ["a", "b"]})
        right = DataFrame.from_dict({"key": [], "right_val": []})

        result = left.merge(right, on="key", how="left")

        assert result.columns == ["key", "left_val", "right_val"]
        assert result.data == [
            [1, "a", None],
            [2, "b", None],
        ]

    def test_merge_with_none_keys(self) -> None:
        """Merge handles None in join keys."""
        left = DataFrame.from_dict({"key": [1, None, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"key": [1, None, 4], "right_val": ["x", "y", "z"]})

        result = left.merge(right, on="key", how="inner")

        # None matches None
        assert [1, "a", "x"] in result.data
        assert [None, "b", "y"] in result.data

    def test_merge_no_matches(self) -> None:
        """Merge with no matching keys."""
        left = DataFrame.from_dict({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = DataFrame.from_dict({"key": [4, 5, 6], "right_val": ["x", "y", "z"]})

        result = left.merge(right, on="key", how="inner")

        assert result.data == []

    def test_merge_error_no_keys(self) -> None:
        """Merge raises ValueError when no keys specified."""
        left = DataFrame.from_dict({"a": [1, 2]})
        right = DataFrame.from_dict({"b": [3, 4]})

        with pytest.raises(ValueError, match="Must specify either"):
            left.merge(right)

    def test_merge_error_both_on_and_left_on(self) -> None:
        """Merge raises ValueError when both 'on' and 'left_on' specified."""
        left = DataFrame.from_dict({"a": [1, 2]})
        right = DataFrame.from_dict({"a": [3, 4]})

        with pytest.raises(ValueError, match="Cannot specify both"):
            left.merge(right, on="a", left_on="a")

    def test_merge_error_left_key_not_found(self) -> None:
        """Merge raises KeyError for nonexistent left key."""
        left = DataFrame.from_dict({"a": [1, 2]})
        right = DataFrame.from_dict({"b": [3, 4]})

        with pytest.raises(KeyError, match="Join key 'nonexistent' not found in left"):
            left.merge(right, left_on="nonexistent", right_on="b")

    def test_merge_error_right_key_not_found(self) -> None:
        """Merge raises KeyError for nonexistent right key."""
        left = DataFrame.from_dict({"a": [1, 2]})
        right = DataFrame.from_dict({"b": [3, 4]})

        with pytest.raises(KeyError, match="Join key 'nonexistent' not found in right"):
            left.merge(right, left_on="a", right_on="nonexistent")

    def test_merge_error_key_length_mismatch(self) -> None:
        """Merge raises ValueError for mismatched key lengths."""
        left = DataFrame.from_dict({"a": [1, 2], "b": [3, 4]})
        right = DataFrame.from_dict({"c": [5, 6], "d": [7, 8]})

        with pytest.raises(ValueError, match="left_on and right_on must have same length"):
            left.merge(right, left_on=["a", "b"], right_on=["c"])


class TestDataFrameTranspose:
    """Test DataFrame.transpose() method."""

    def test_transpose_basic(self) -> None:
        """Transpose basic DataFrame."""
        df = DataFrame.from_dict(
            {"metric": ["revenue", "profit", "growth"], "2023": [1000, 200, 0.10], "2024": [1200, 250, 0.20]}
        )

        transposed = df.transpose()

        assert transposed.columns == ["index", "revenue", "profit", "growth"]
        assert transposed.data == [
            ["2023", 1000, 200, 0.10],
            ["2024", 1200, 250, 0.20],
        ]

    def test_transpose_numeric_data(self) -> None:
        """Transpose DataFrame with numeric values."""
        df = DataFrame.from_dict({"id": [1, 2, 3], "a": [10, 20, 30], "b": [40, 50, 60]})

        transposed = df.transpose()

        assert transposed.columns == ["index", "1", "2", "3"]
        assert transposed.data == [
            ["a", 10, 20, 30],
            ["b", 40, 50, 60],
        ]

    def test_transpose_string_index(self) -> None:
        """Transpose with string index column."""
        df = DataFrame.from_dict(
            {"product": ["Widget", "Gadget"], "jan": [100, 200], "feb": [110, 210], "mar": [120, 220]}
        )

        transposed = df.transpose()

        assert transposed.columns == ["index", "Widget", "Gadget"]
        assert transposed.data == [
            ["jan", 100, 200],
            ["feb", 110, 210],
            ["mar", 120, 220],
        ]

    def test_transpose_with_none_values(self) -> None:
        """Transpose preserves None values."""
        df = DataFrame.from_dict({"id": ["a", "b"], "x": [1, None], "y": [None, 4]})

        transposed = df.transpose()

        assert transposed.data == [
            ["x", 1, None],
            ["y", None, 4],
        ]

    def test_transpose_single_row(self) -> None:
        """Transpose DataFrame with single row."""
        df = DataFrame.from_dict({"id": [1], "a": [10], "b": [20], "c": [30]})

        transposed = df.transpose()

        assert transposed.columns == ["index", "1"]
        assert transposed.data == [
            ["a", 10],
            ["b", 20],
            ["c", 30],
        ]

    def test_transpose_single_column(self) -> None:
        """Transpose DataFrame with single column."""
        df = DataFrame.from_dict({"values": [1, 2, 3]})

        transposed = df.transpose()

        # Single column transposes to row of column names
        assert transposed.columns == ["1", "2", "3"]
        assert transposed.data == [["values"]]

    def test_transpose_twice(self) -> None:
        """Transposing twice restores original structure."""
        original = DataFrame.from_dict({"id": ["a", "b"], "x": [1, 2], "y": [3, 4]})

        transposed_once = original.transpose()
        transposed_twice = transposed_once.transpose()

        # Column names might differ but structure should match
        assert len(transposed_twice.columns) == len(original.columns)
        assert len(transposed_twice.data) == len(original.data)

    def test_transpose_empty(self) -> None:
        """Transpose empty DataFrame."""
        df = DataFrame.from_dict({})

        transposed = df.transpose()

        assert transposed.columns == []
        assert transposed.data == []

    def test_transpose_mixed_types(self) -> None:
        """Transpose with mixed data types."""
        df = DataFrame.from_dict({"name": ["Alice", "Bob"], "age": [25, 30], "active": [True, False]})

        transposed = df.transpose()

        assert transposed.columns == ["index", "Alice", "Bob"]
        assert transposed.data == [
            ["age", 25, 30],
            ["active", True, False],
        ]

    def test_transpose_large_dataset(self) -> None:
        """Transpose larger DataFrame."""
        df = DataFrame.from_dict(
            {
                "quarter": ["Q1", "Q2", "Q3", "Q4"],
                "north": [100, 110, 120, 130],
                "south": [200, 210, 220, 230],
                "east": [150, 160, 170, 180],
                "west": [180, 190, 200, 210],
            }
        )

        transposed = df.transpose()

        assert transposed.columns == ["index", "Q1", "Q2", "Q3", "Q4"]
        assert len(transposed.data) == 4  # 4 regions
        assert transposed.data[0] == ["north", 100, 110, 120, 130]
        assert transposed.data[3] == ["west", 180, 190, 200, 210]

    def test_transpose_maintains_order(self) -> None:
        """Transpose maintains row/column order."""
        df = DataFrame.from_dict({"key": ["x", "y", "z"], "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

        transposed = df.transpose()

        # Check order is preserved
        assert transposed.data[0][0] == "col1"
        assert transposed.data[1][0] == "col2"
        assert transposed.data[2][0] == "col3"
        assert transposed.data[0] == ["col1", 1, 2, 3]
        assert transposed.data[1] == ["col2", 4, 5, 6]
        assert transposed.data[2] == ["col3", 7, 8, 9]

    def test_transpose_no_columns_edge_case(self) -> None:
        """Transpose handles edge case of DataFrame with data but no columns."""
        # This is an edge case that shouldn't normally occur
        df = DataFrame(columns=[], data=[])

        transposed = df.transpose()

        assert transposed.columns == []
        assert transposed.data == []
