from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar
from uuid import UUID, uuid4

import pandas as pd
from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, ConfigDict, Field


class ChapConfig(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str


TChapConfig = TypeVar("T", bound=ChapConfig)


class HealthStatus(StrEnum):
    up = "up"
    down = "down"


class HealthResponse(BaseModel):
    status: HealthStatus
    # Allow extra fields so runners can attach arbitrary metadata
    model_config = ConfigDict(extra="allow")


class ChapServiceInfo(BaseModel):
    display_name: str

    model_config = ConfigDict(extra="forbid")


class JobStatus(StrEnum):
    # Aligned with your current naming (and added 'canceled')
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


class JobType(StrEnum):
    train = "train"
    predict = "predict"


class JobRequest[T: ChapConfig](BaseModel):
    """What you submit to the scheduler/runner."""

    id: UUID
    type: JobType
    config: T

    # Allow arbitrary (non-JSON) Python objects, e.g., pandas.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)


class JobResponse(BaseModel):
    """What you return immediately after enqueueing a job."""

    id: UUID
    type: JobType
    status: JobStatus = Field(description="Current status of the job")


class JobRecord(BaseModel):
    id: UUID
    type: JobType | None = None
    status: JobStatus = JobStatus.pending
    submitted_at: datetime | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    error: str | None = None


class DataFrameSplit(BaseModel):
    """
    Pydantic model for pandas.DataFrame serialized with orient="split".
    """

    columns: list[str] = Field(..., description="List of column names")
    index: list[int] = Field(..., description="List of row indices")
    data: list[list[Any]] = Field(..., description="2D row-major data (each row is a list of values)")

    def to_pandas(self) -> pd.DataFrame:
        """Convert this validated structure into a pandas DataFrame."""
        return pd.DataFrame(data=self.data, index=self.index, columns=self.columns)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "DataFrameSplit":
        """Create a DataFrameSplit model from a pandas DataFrame."""
        payload = df.to_dict(orient="split")
        return cls.model_validate(payload)


class TrainBody(BaseModel):
    df: DataFrameSplit
    geo: FeatureCollection | None = None

    model_config = ConfigDict(extra="allow")


class PredictBody(BaseModel):
    df: DataFrameSplit
    geo: FeatureCollection | None = None

    model_config = ConfigDict(extra="allow")


class TrainData(BaseModel):
    df: pd.DataFrame
    geo: FeatureCollection | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class TrainParams(BaseModel):
    config: TChapConfig
    data: TrainData

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PredictData(BaseModel):
    df: pd.DataFrame
    geo: FeatureCollection | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PredictParams(BaseModel):
    config: TChapConfig
    data: TrainData

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
