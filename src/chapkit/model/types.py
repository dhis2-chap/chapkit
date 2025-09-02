from enum import Enum
from typing import TypeVar

from pydantic import ConfigDict, EmailStr, HttpUrl

from chapkit.types import ChapConfig, ChapServiceInfo


class AssessedStatus(str, Enum):
    gray = "gray"  # Gray: Not intended for use, or deprecated/meant for legacy use only.
    red = "red"  # Red: Highly experimental prototype - not at all validated and only meant for early experimentation
    orange = "orange"  # Orange: Has seen promise on limited data, needs manual configuration and careful evaluation
    yellow = "yellow"  # Yellow: Ready for more rigorous testing
    green = "green"  # Green: Validated, ready for use


class ChapModelConfig(ChapConfig):
    pass


TChapModelConfig = TypeVar("TChapModelConfig", bound=ChapModelConfig)


class ChapModelServiceInfo(ChapServiceInfo):
    author: str
    author_note: str
    author_assessed_status: AssessedStatus
    contact_email: EmailStr
    description: str
    organization: str
    organization_logo_url: HttpUrl
    citation_info: str

    # required-only, no extras sneaking in:
    model_config = ConfigDict(extra="forbid")
