# chapkit/api/info.py
from fastapi import APIRouter

from chapkit.api.type import ChapApi
from chapkit.runner import ChapRunner
from chapkit.type import ChapServiceInfo, TChapConfig


class InfoApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
    ) -> None:
        self._runner = runner

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["information"])

        router.add_api_route(
            path="/info",
            endpoint=self._runner.on_info,
            methods=["GET"],
            name="info",
            response_model=ChapServiceInfo,
        )

        return router
