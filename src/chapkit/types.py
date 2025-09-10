from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar
from uuid import UUID, uuid4

import pandas as pd
from geojson_pydantic import FeatureCollection
from pydantic import BaseModel, Field, ConfigDict, EmailStr, HttpUrl


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


class AssessedStatus(StrEnum):
    gray = "gray"  # Gray: Not intended for use, or deprecated/meant for legacy use only.
    red = "red"  # Red: Highly experimental prototype - not at all validated and only meant for early experimentation
    orange = "orange"  # Orange: Has seen promise on limited data, needs manual configuration and careful evaluation
    yellow = "yellow"  # Yellow: Ready for more rigorous testing
    green = "green"  # Green: Validated, ready for use


class ChapServiceInfo(BaseModel):
    display_name: str
    author: str | None = None
    author_note: str | None = None
    author_assessed_status: AssessedStatus | None = None
    contact_email: EmailStr | None = None
    description: str | None = None
    organization: str | None = None
    organization_logo_url: HttpUrl | None = None
    citation_info: str | None = None

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
    type: JobType | None = None
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


class TrainData(BaseModel):
    data: pd.DataFrame
    geo: FeatureCollection | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class TrainParams(BaseModel):
    config: TChapConfig
    body: TrainData

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class TrainBody(BaseModel):
    data: DataFrameSplit
    geo: FeatureCollection | None = None

    model_config = ConfigDict(extra="allow")


class PredictData(BaseModel):
    historic: pd.DataFrame
    future: pd.DataFrame
    geo: FeatureCollection | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PredictParams(BaseModel):
    config: TChapConfig
    artifact: Any | None = None
    body: PredictData

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PredictBody(BaseModel):
    historic: DataFrameSplit
    future: DataFrameSplit
    geo: FeatureCollection | None = None

    model_config = ConfigDict(extra="allow")


class ArtifactInfo(BaseModel):
    id: UUID
    config_id: UUID
    config_name: str
    data: Any | None = None
