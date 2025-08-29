from enum import Enum
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl

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


class AssessedStatus(str, Enum):
    gray = "gray"  # Gray: Not intended for use, or deprecated/meant for legacy use only.
    red = "red"  # Red: Highly experimental prototype - not at all validated and only meant for early experimentation
    orange = "orange"  # Orange: Has seen promise on limited data, needs manual configuration and careful evaluation
    yellow = "yellow"  # Yellow: Ready for more rigorous testing
    green = "green"  # Green: Validated, ready for use


class ChapServiceInfo(BaseModel):
    author: str
    author_note: str
    author_assessed_status: AssessedStatus
    contact_email: EmailStr
    description: str
    display_name: str
    organization: str
    organization_logo_url: HttpUrl
    citation_info: str

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
