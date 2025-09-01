from enum import Enum
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChapConfig(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str


TChapConfig = TypeVar("T", bound=ChapConfig)


class HealthStatus(str, Enum):
    up = "up"
    down = "down"


class HealthResponse(BaseModel):
    status: HealthStatus
    model_config = ConfigDict(extra="allow")


class ChapServiceInfo(BaseModel):
    display_name: str

    # required-only, no extras sneaking in:
    model_config = ConfigDict(extra="forbid")


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobType(str, Enum):
    train = "train"
    predict = "predict"


class JobResponse(BaseModel):
    status: JobStatus = JobStatus.pending
    type: JobType
