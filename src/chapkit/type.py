from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TypeVar
from uuid import UUID, uuid4

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
    # Strict: no extras
    model_config = ConfigDict(extra="forbid")


class JobStatus(StrEnum):
    # Aligned with your current naming (and added 'canceled')
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"  # <-- new; useful for DELETE /jobs/{id}


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
    type: JobType
    status: JobStatus = JobStatus.pending
    submitted_at: datetime | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    error: str | None = None
