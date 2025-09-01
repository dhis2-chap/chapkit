from abc import ABC, abstractmethod

from chapkit.type import JobResponse


class Scheduler(ABC):
    @abstractmethod
    def schedule(self) -> JobResponse:
        pass
