from abc import ABC
from typing import Generic

from chapkit.types import ChapServiceInfo, HealthResponse, HealthStatus, TChapConfig


class ChapRunner(Generic[TChapConfig], ABC):
    def __init__(self, info: ChapServiceInfo, *, config_type: type[TChapConfig]) -> None:
        self._info = info
        self._config_type: type[TChapConfig] = config_type

    async def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)

    async def on_info(self) -> ChapServiceInfo:
        return self._info

    @property
    def config_type(self) -> type[TChapConfig]:
        return self._config_type
