# chapkit/service.py (excerpt)
from typing import Any, Generic, TypeVar
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query

from chapkit.model.runner import ChapModelRunner
from chapkit.model.type import ChapModelConfig, ChapModelServiceInfo
from chapkit.service import ChapService
from chapkit.storage import ChapStorage
from chapkit.type import JobResponse

TModelConfig = TypeVar("TModelConfig", bound=ChapModelConfig)


class ChapModelService(ChapService[TModelConfig], Generic[TModelConfig]):
    def __init__(
        self,
        runner: ChapModelRunner[TModelConfig],
        storage: ChapStorage[TModelConfig],
    ) -> None:
        super().__init__(runner, storage)
        self._runner = runner  # narrowed type

    def _setup_routes(self, router: APIRouter) -> None:
        super()._setup_routes(router)
        self._setup_train(router)
        self._setup_predict(router)

    def _setup_info(self, router: APIRouter) -> None:
        router.add_api_route(
            path="/info",
            endpoint=self._runner.on_info,
            methods=["GET"],
            tags=["information"],
            name="info",
            response_model=ChapModelServiceInfo,
        )

    def _resolve_cfg(self, id: UUID) -> TModelConfig:
        cfg = self._storage.get_config(id)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {id} not found")
        return cfg

    @staticmethod
    def _df_from_json(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(rows)

    def _setup_train(self, router: APIRouter) -> None:
        def endpoint(
            config: UUID = Query(..., description="Config ID"),
            rows: list[dict[str, Any]] = Body(
                ...,
                description="Data as JSON records: [{...}, {...}]",
                example=[
                    {"k1": "v1", "k2": 1.0},
                    {"k1": "v2", "k2": 2.0},
                ],
            ),
        ) -> JobResponse:
            cfg = self._resolve_cfg(config)
            df = self._df_from_json(rows)
            return self._runner.on_train(cfg, df)

        router.add_api_route(
            path="/train",
            endpoint=endpoint,
            methods=["POST"],
            response_model=JobResponse,
            status_code=202,
            tags=["chap"],
            name="train",
            responses={
                202: {"description": "Job accepted"},
                404: {"description": "Config not found"},
                422: {"description": "Validation error"},
            },
        )

    def _setup_predict(self, router: APIRouter) -> None:
        def endpoint(
            config: UUID = Query(..., description="Config ID"),
            rows: list[dict[str, Any]] = Body(
                ...,
                description="Data as JSON records: [{...}, {...}]",
                example=[
                    {"k1": "v1", "k2": 1.0},
                    {"k1": "v2", "k2": 2.0},
                ],
            ),
        ) -> JobResponse:
            cfg = self._resolve_cfg(config)
            df = self._df_from_json(rows)
            return self._runner.on_predict(cfg, df)

        router.add_api_route(
            path="/predict",
            endpoint=endpoint,
            methods=["POST"],
            response_model=JobResponse,
            status_code=202,
            tags=["chap"],
            name="predict",
            responses={
                202: {"description": "Job accepted"},
                404: {"description": "Config not found"},
                422: {"description": "Validation error"},
            },
        )
