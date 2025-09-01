# chapkit/model/api/train.py
from typing import Any, Generic
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Body, Query

from chapkit.api.type import ChapApi
from chapkit.model.runner import ChapModelRunner
from chapkit.model.type import TChapModelConfig
from chapkit.service import ChapService
from chapkit.type import JobResponse


class TrainApi(ChapApi[TChapModelConfig], Generic[TChapModelConfig]):
    def __init__(
        self,
        runner: ChapModelRunner[TChapModelConfig],
        service: ChapService[TChapModelConfig],
    ) -> None:
        self._runner = runner
        self._service = service

    def create_router(self) -> APIRouter:
        router = APIRouter(tags=["chap"])

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
            cfg = self._service._resolve_cfg(config)
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

        return router

    @staticmethod
    def _df_from_json(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(rows)
