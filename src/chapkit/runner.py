from typing import Protocol, runtime_checkable

from chapkit.types import ChapConfig, HealthResponse, HealthStatus


@runtime_checkable
class ChapRunner[T: ChapConfig](Protocol):
    def on_health(self) -> HealthResponse: ...


class ChapRunnerBase[T: ChapConfig](ChapRunner[T]):
    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)
