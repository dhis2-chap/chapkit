from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class DataFrameSplit(BaseModel):
    columns: list[str] = Field(...)
    index: list[int] = Field(...)
    data: list[list[Any]] = Field(...)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.data, index=self.index, columns=self.columns)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "DataFrameSplit":
        payload = df.to_dict(orient="split")
        return cls.model_validate(payload)


def test_round_trip():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[10, 11])
    m = DataFrameSplit.from_pandas(df)
    df2 = m.to_pandas()
    assert df.equals(df2)
