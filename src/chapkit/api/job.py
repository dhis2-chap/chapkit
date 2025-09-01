from fastapi import APIRouter

from chapkit.api.type import ChapApi
from chapkit.runner import ChapRunner
from chapkit.storage import ChapStorage
from chapkit.type import TChapConfig


class JobApi(ChapApi[TChapConfig]):
    def __init__(
        self,
        runner: ChapRunner[TChapConfig],
        storage: ChapStorage[TChapConfig],
    ) -> None:
        self._runner = runner
        self._storage = storage

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["jobs"])

        return router

    def _setup_routes(self, router: APIRouter) -> None:
        pass
