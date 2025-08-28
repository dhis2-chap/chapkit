from enum import Enum
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

TChapModelConfig = TypeVar("T", bound="ChapConfig")


class HealthStatus(str, Enum):
    up = "up"
    down = "down"


class HealthResponse(BaseModel):
    status: HealthStatus
    model_config = ConfigDict(extra="allow")


class ChapConfig(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str


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
