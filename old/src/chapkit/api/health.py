from fastapi import APIRouter

from chapkit.api.types import ChapApi
from chapkit.runner import ChapRunner
from chapkit.types import HealthResponse, TChapConfig


class HealthApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
    ) -> None:
        self._runner = runner

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["information"])

        router.add_api_route(
            path="/health",
            endpoint=self._runner.on_health,
            methods=["GET"],
            name="health",
            response_model=HealthResponse,
            responses={
                200: {
                    "description": "Service health status",
                    "content": {"application/json": {"example": {"status": "up"}}},
                }
            },
        )

        return router
