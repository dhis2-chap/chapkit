from abc import ABC, abstractmethod
from typing import Generic

from fastapi import APIRouter

from chapkit.types import TChapConfig


class ChapApi(Generic[TChapConfig], ABC):
    @abstractmethod
    def create_router(self) -> APIRouter: ...
