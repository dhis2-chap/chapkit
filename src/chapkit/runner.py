from typing import Protocol, runtime_checkable

from chapkit.types import ChapConfig, HealthResponse, HealthStatus


@runtime_checkable
class ChapRunner[T: ChapConfig](Protocol):
    def on_health(self) -> HealthResponse: ...

    # def on_info(self, info: ChapServiceInfo) -> ChapServiceInfo: ...
    # def on_train(self, cfg: T) -> JobResponse: ...
    # def on_predict(self, cfg: T) -> JobResponse: ...


class ChapRunnerBase[T: ChapConfig](ChapRunner[T]):
    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)
