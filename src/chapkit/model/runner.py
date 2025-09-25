from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
import shlex
from typing import Generic

import pandas as pd
import structlog


from chapkit.logging import log_time
from chapkit.model.types import OnPredictCallable, OnTrainCallable, TChapModelConfig
from chapkit.types import ChapServiceInfo, DataFrameSplit, ULID
from chapkit.runner import ChapRunner
from chapkit.database import ChapDatabase
from chapkit.types import HealthResponse, HealthStatus, PredictParams, TrainParams
from chapkit.utils import make_awaitable

log = structlog.get_logger()


class ChapModelRunner(ChapRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def __init__(
        self,
        info: ChapServiceInfo,
        database: ChapDatabase[TChapModelConfig],
        *,
        config_type: type[TChapModelConfig],
    ) -> None:
        super().__init__(info, config_type=config_type)
        self._database = database

    @abstractmethod
    def on_health(self) -> HealthResponse: ...

    @abstractmethod
    async def on_train(self, params: TrainParams) -> ULID: ...

    @abstractmethod
    async def on_predict(self, params: PredictParams) -> ULID: ...


class ChapModelRunnerBase(ChapModelRunner[TChapModelConfig], Generic[TChapModelConfig], ABC):
    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)

    def on_info(self) -> ChapServiceInfo:
        return self._info


class FunctionalChapModelRunner(ChapModelRunnerBase[TChapModelConfig], Generic[TChapModelConfig]):
    def __init__(
        self,
        info: ChapServiceInfo,
        database: ChapDatabase[TChapModelConfig],
        *,
        config_type: type[TChapModelConfig],
        on_train: OnTrainCallable,
        on_predict: OnPredictCallable,
    ):
        self._on_train_func = make_awaitable(on_train)
        self._on_predict_func = make_awaitable(on_predict)
        super().__init__(info, database, config_type=config_type)

    async def on_train(self, params: TrainParams) -> ULID:
        with log_time("on_train", step="on_train_func"):
            model = await self._on_train_func(
                config=params.config,
                data=params.body.data,
                geo=params.body.geo,
            )

        with log_time("on_train", step="save_artifact"):
            artifact_id = ULID()
            self._database.add_artifact(artifact_id, params.config, model)

        return artifact_id

    async def on_predict(self, params: PredictParams) -> ULID:
        with log_time("on_predict", step="on_predict_func"):
            result = await self._on_predict_func(
                config=params.config,
                model=params.artifact,
                historic=params.body.historic,
                future=params.body.future,
                geo=params.body.geo,
            )

        with log_time("on_predict", step="save_artifact"):
            artifact_id = ULID()
            self._database.add_artifact(
                artifact_id,
                params.config,
                DataFrameSplit.from_pandas(result),
                parent_id=params.artifact_id,
            )

        return artifact_id


class ShellCommandChapModelRunner(ChapModelRunnerBase[TChapModelConfig], Generic[TChapModelConfig]):
    def __init__(
        self,
        info: ChapServiceInfo,
        database: ChapDatabase[TChapModelConfig],
        *,
        config_type: type[TChapModelConfig],
        on_train: str,
        on_predict: str,
    ):
        # Split strings into argv lists (safe for subprocess_exec)
        self.on_train_argv = shlex.split(on_train)
        self.on_predict_argv = shlex.split(on_predict)
        super().__init__(info, database, config_type=config_type)

    async def _run(self, argv: list[str]) -> None:
        log.info("exec.start", argv=argv)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if stdout:
            log.info("exec.stdout", text=stdout.decode(errors="replace"))

        if stderr:
            log.warning("exec.stderr", text=stderr.decode(errors="replace"))

        if proc.returncode != 0:
            raise RuntimeError(f"Process failed with rc={proc.returncode}")

        log.info("exec.done", rc=proc.returncode)

    async def on_train(self, params: TrainParams) -> ULID:
        Path("input").mkdir(parents=True, exist_ok=True)
        Path("output").mkdir(parents=True, exist_ok=True)

        params.body.data.to_csv("input/train.csv", index=False)

        file_args = [
            "--input-data=input/train.csv",
            "--output-model=output/model.bin",
        ]

        if params.body.geo is not None:
            with open("input/geometry.geojson", "w") as f:
                f.write(params.body.geo.model_dump_json())

            file_args.append("--input-geojson=input/geometry.geojson")

        config_args = [f"--{k}={v}" for k, v in params.config.model_dump().items()]

        await self._run([*self.on_train_argv, *config_args, *file_args])

        if Path("output/model.bin").exists():
            model = Path("output/model.bin").read_bytes()
            artifact_id = ULID()

            self._database.add_artifact(
                artifact_id,
                params.config,
                model,
            )

            log.info("artifact_id", artifact_id=str(artifact_id))

            return artifact_id

    async def on_predict(self, params: PredictParams) -> ULID:
        Path("input").mkdir(parents=True, exist_ok=True)
        Path("output").mkdir(parents=True, exist_ok=True)

        params.body.historic.to_csv("input/historic.csv", index=False)
        params.body.future.to_csv("input/future.csv", index=False)

        file_args = [
            "--input-historic-data=input/historic.csv",
            "--input-future-data=input/future.csv",
            "--output-data=output/predictions.csv",
        ]

        if params.body.geo is not None:
            with open("input/geometry.geojson", "w") as f:
                f.write(params.body.geo.model_dump_json())

            file_args.append("--input-geojson=input/geometry.geojson")

        if params.artifact_id is not None:
            model = self._database.get_artifact(params.artifact_id)
            Path("output/model.bin").write_bytes(model)

            file_args.append("--input-model=output/model.bin")

        config_args = [f"--{k}={v}" for k, v in params.config.model_dump().items()]

        await self._run([*self.on_predict_argv, *config_args, *file_args])

        if Path("output/predictions.csv").exists():
            artifact_id = ULID()
            self._database.add_artifact(
                artifact_id,
                params.config,
                DataFrameSplit.from_pandas(pd.read_csv("output/predictions.csv")),
                parent_id=params.artifact_id,
            )

            return artifact_id
