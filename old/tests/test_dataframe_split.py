import pandas as pd

from chapkit.types import DataFrameSplit


def test_round_trip():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[10, 11])
    m = DataFrameSplit.from_pandas(df)
    df2 = m.to_pandas()
    assert df.equals(df2)
