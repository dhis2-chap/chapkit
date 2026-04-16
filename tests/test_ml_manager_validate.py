"""Tests for MLManager.validate and _check_prediction_periods."""

from __future__ import annotations

import io
import pickle
import time
import zipfile
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import Artifact, ArtifactRepository
from chapkit.config import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner, MLManager, ShellModelRunner
from chapkit.ml.schemas import ValidatePredictRequest


class SampleConfig(BaseConfig):
    """Test configuration schema."""

    prediction_periods: int = 3


class TestCheckPredictionPeriods:
    """Unit tests for _check_prediction_periods (returns diagnostics, does not raise)."""

    def _bare_manager(self, *, minimum: int, maximum: int) -> MLManager[BaseConfig]:
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = minimum
        manager.max_prediction_periods = maximum
        return manager

    def test_within_bounds_returns_empty(self) -> None:
        manager = self._bare_manager(minimum=1, maximum=10)
        assert manager._check_prediction_periods(SampleConfig(prediction_periods=5)) == []

    def test_below_minimum_returns_error_diagnostic(self) -> None:
        manager = self._bare_manager(minimum=5, maximum=10)
        diagnostics = manager._check_prediction_periods(SampleConfig(prediction_periods=3))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].code == "prediction_periods_out_of_bounds"
        assert diagnostics[0].field == "config.prediction_periods"
        assert "below the minimum" in diagnostics[0].message

    def test_above_maximum_returns_error_diagnostic(self) -> None:
        manager = self._bare_manager(minimum=1, maximum=5)
        diagnostics = manager._check_prediction_periods(SampleConfig(prediction_periods=10))
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].code == "prediction_periods_out_of_bounds"
        assert "exceeds the maximum" in diagnostics[0].message

    def test_at_boundaries_returns_empty(self) -> None:
        manager = self._bare_manager(minimum=3, maximum=7)
        assert manager._check_prediction_periods(SampleConfig(prediction_periods=3)) == []
        assert manager._check_prediction_periods(SampleConfig(prediction_periods=7)) == []


async def _noop_train(config: Any, data: Any, geo: Any = None) -> Any:
    """Placeholder train callback so FunctionalModelRunner can be constructed."""
    return {}


async def _noop_predict(config: Any, model: Any, historic: Any, future: Any, geo: Any = None) -> Any:
    """Placeholder predict callback so FunctionalModelRunner can be constructed."""
    return future


def _zip_with_pickle(obj: Any) -> bytes:
    """Build a ZIP archive containing a single model.pickle entry with the given Python object."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.pickle", pickle.dumps(obj))
    return buffer.getvalue()


def _zip_without_pickle() -> bytes:
    """Build a valid ZIP that omits model.pickle (simulating a malformed training workspace)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.yml", "prediction_periods: 3\n")
    return buffer.getvalue()


class TestCheckTrainingWorkspace:
    """Regression tests for _check_training_workspace (CLIM-581 review)."""

    def _manager_with_runner(self, runner: Any) -> MLManager[BaseConfig]:
        manager = MLManager.__new__(MLManager)
        manager.runner = runner
        return manager

    def test_non_zip_content_is_skipped(self) -> None:
        """Non-ZIP training artifacts (pickled model as object) emit no diagnostics here."""
        manager = self._manager_with_runner(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict))
        artifact_id = ULID()
        training_data = {"content_type": "application/x-pickle", "content": object()}

        assert manager._check_training_workspace(training_data, artifact_id) == []

    def test_empty_workspace_content_emits_corrupted(self) -> None:
        manager = self._manager_with_runner(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict))
        training_data = {"content_type": "application/zip", "content": b""}

        diagnostics = manager._check_training_workspace(training_data, ULID())

        assert len(diagnostics) == 1
        assert diagnostics[0].code == "training_workspace_corrupted"

    def test_bad_zip_bytes_emit_corrupted(self) -> None:
        """Non-ZIP bytes stored as application/zip must surface as a structured diagnostic."""
        manager = self._manager_with_runner(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict))
        training_data = {"content_type": "application/zip", "content": b"not-a-zip"}

        diagnostics = manager._check_training_workspace(training_data, ULID())

        assert len(diagnostics) == 1
        assert diagnostics[0].code == "training_workspace_corrupted"

    def test_missing_model_pickle_emits_missing(self) -> None:
        """Functional/base runner without model.pickle in workspace is a deterministic predict failure."""
        manager = self._manager_with_runner(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict))
        training_data = {"content_type": "application/zip", "content": _zip_without_pickle()}

        diagnostics = manager._check_training_workspace(training_data, ULID())

        assert len(diagnostics) == 1
        assert diagnostics[0].code == "model_pickle_missing"

    def test_corrupt_model_pickle_emits_corrupted(self) -> None:
        """Corrupt model.pickle bytes must surface as model_pickle_corrupted, not valid=True."""
        manager = self._manager_with_runner(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict))

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pickle", b"not-a-valid-pickle")
        training_data = {"content_type": "application/zip", "content": buffer.getvalue()}

        diagnostics = manager._check_training_workspace(training_data, ULID())

        assert len(diagnostics) == 1
        assert diagnostics[0].code == "model_pickle_corrupted"

    def test_valid_workspace_passes(self) -> None:
        """A well-formed workspace with a loadable pickle emits no diagnostics."""
        manager = self._manager_with_runner(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict))
        training_data = {"content_type": "application/zip", "content": _zip_with_pickle({"weights": [1.0, 2.0]})}

        assert manager._check_training_workspace(training_data, ULID()) == []

    def test_shell_runner_skips_pickle_check(self) -> None:
        """ShellModelRunner treats the workspace itself as the model, so missing model.pickle is OK."""
        manager = self._manager_with_runner(
            ShellModelRunner(train_command="true", predict_command="true"),
        )
        training_data = {"content_type": "application/zip", "content": _zip_without_pickle()}

        assert manager._check_training_workspace(training_data, ULID()) == []


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Spin up the ml_class example app for validate integration tests."""
    pytest.importorskip("pandas")
    from examples.ml_class.main import app

    with TestClient(app) as test_client:
        yield test_client


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[Any, Any]:
    start = time.time()
    while time.time() - start < timeout:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = cast(dict[Any, Any], response.json())
        if job["status"] in ["completed", "failed", "canceled"]:
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _create_config(client: TestClient, *, min_samples: int = 3, prediction_periods: int = 3) -> str:
    response = client.post(
        "/api/v1/configs",
        json={
            "name": f"validate_config_{ULID()}",
            "data": {
                "min_samples": min_samples,
                "normalize_features": True,
                "prediction_periods": prediction_periods,
            },
        },
    )
    assert response.status_code == 201, response.text
    return cast(str, response.json()["id"])


def test_validate_train_happy_path(client: TestClient) -> None:
    """A well-formed train payload produces valid=True."""
    config_id = _create_config(client)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [8.0, 22.0, 55.0, 3.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["diagnostics"] == []


def test_validate_train_config_not_found(client: TestClient) -> None:
    """Non-existent config produces a config_not_found diagnostic, not a 404."""
    body = {
        "type": "train",
        "config_id": str(ULID()),
        "data": {"columns": ["rainfall"], "data": [[1.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "config_not_found" in codes


def test_validate_train_data_empty(client: TestClient) -> None:
    """Empty training data produces a data_empty diagnostic."""
    config_id = _create_config(client)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {"columns": ["rainfall"], "data": []},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "data_empty" in codes


def test_validate_predict_artifact_not_found(client: TestClient) -> None:
    """Non-existent artifact produces a training_artifact_not_found diagnostic."""
    body = {
        "type": "predict",
        "artifact_id": str(ULID()),
        "historic": {"columns": ["rainfall"], "data": [[1.0]]},
        "future": {"columns": ["rainfall"], "data": [[2.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "training_artifact_not_found" in codes


def test_validate_predict_happy_path(client: TestClient) -> None:
    """After a real training run the predict validate call returns valid=True."""
    config_id = _create_config(client)
    train_body = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [8.0, 22.0, 55.0, 3.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }

    train_response = client.post("/api/v1/ml/$train", json=train_body)
    assert train_response.status_code == 202
    train_data = train_response.json()
    job = _wait_for_job(client, train_data["job_id"])
    assert job["status"] == "completed", job

    validate_body = {
        "type": "predict",
        "artifact_id": train_data["artifact_id"],
        "historic": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [[10.0, 25.0, 60.0, 5.0]],
        },
        "future": {
            "columns": ["rainfall", "mean_temperature", "humidity"],
            "data": [[11.0, 26.0, 62.0]],
        },
    }

    response = client.post("/api/v1/ml/$validate", json=validate_body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True, payload


def test_validate_predict_empty_historic_and_future(client: TestClient) -> None:
    """Empty historic and future both produce diagnostics after a valid artifact."""
    config_id = _create_config(client)
    train_body = {
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
                [8.0, 22.0, 55.0, 3.0],
                [20.0, 30.0, 80.0, 12.0],
            ],
        },
    }
    train_response = client.post("/api/v1/ml/$train", json=train_body)
    train_data = train_response.json()
    job = _wait_for_job(client, train_data["job_id"])
    assert job["status"] == "completed", job

    validate_body = {
        "type": "predict",
        "artifact_id": train_data["artifact_id"],
        "historic": {"columns": ["rainfall"], "data": []},
        "future": {"columns": ["rainfall"], "data": []},
    }

    response = client.post("/api/v1/ml/$validate", json=validate_body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "historic_empty" in codes
    assert "future_empty" in codes


def test_validate_train_runner_diagnostic_flows_through(client: TestClient) -> None:
    """A domain diagnostic from on_validate_train reaches the response."""
    # WeatherConfig.min_samples defaults to 5 in the example; WeatherModelRunner's
    # on_validate_train override (see examples/ml_class/main.py) emits an error
    # diagnostic when data has fewer rows than min_samples.
    config_id = _create_config(client, min_samples=10)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [
                [10.0, 25.0, 60.0, 5.0],
                [15.0, 28.0, 70.0, 8.0],
            ],
        },
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    codes = [d["code"] for d in payload["diagnostics"]]
    assert "insufficient_training_samples" in codes
    assert payload["valid"] is False


async def _seed_training_artifact(db: Any, data: dict[str, Any]) -> ULID:
    """Insert a hand-crafted training artifact into the DB and return its ID."""
    async with db.session() as session:
        repo = ArtifactRepository(session)
        artifact = Artifact(data=data, level=0)
        await repo.save(artifact)
        await repo.commit()
        await repo.refresh_many([artifact])
        return artifact.id


async def _build_manager(runner: Any, db: Any, *, schema: type[BaseConfig] = SampleConfig) -> MLManager[BaseConfig]:
    """Construct a minimal MLManager bound to a real DB — no scheduler."""
    manager: MLManager[BaseConfig] = MLManager.__new__(MLManager)
    manager.runner = runner
    manager.database = db
    manager.config_schema = schema
    manager.min_prediction_periods = 0
    manager.max_prediction_periods = 100
    return manager


async def test_validate_predict_handles_malformed_config_id_in_metadata() -> None:
    """Regression: metadata.config_id='not-a-ulid' must yield invalid_training_artifact, not 500."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        artifact_id = await _seed_training_artifact(
            db,
            {
                "type": "ml_training_workspace",
                "metadata": {"status": "success", "config_id": "not-a-ulid"},
                "content": b"",
                "content_type": "application/x-pickle",
            },
        )
        manager = await _build_manager(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict), db)

        response = await manager.validate(
            ValidatePredictRequest(
                artifact_id=artifact_id,
                historic=DataFrame(columns=["x"], data=[[1.0]]),
                future=DataFrame(columns=["x"], data=[[2.0]]),
            )
        )

        codes = [d.code for d in response.diagnostics]
        assert "invalid_training_artifact" in codes
        assert response.valid is False
    finally:
        await db.dispose()


async def test_validate_train_skips_runner_hook_when_framework_errored() -> None:
    """Regression: framework errors must short-circuit runner hooks.

    The reviewer filed: a hook that assumes non-empty data crashes
    $validate when data is empty, even though the framework already
    caught data_empty. After the fix, the hook must not be called.
    """
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        from chapkit.config import ConfigManager, ConfigRepository
        from chapkit.config.schemas import ConfigIn
        from chapkit.ml.schemas import ValidateTrainRequest

        async with db.session() as session:
            config_repo = ConfigRepository(session)
            config_manager: ConfigManager[BaseConfig] = ConfigManager(config_repo, SampleConfig)
            created = await config_manager.save(ConfigIn(name="seed", data=SampleConfig(prediction_periods=3)))
            config_id = created.id

        hook_called = {"count": 0}

        async def exploding_hook(config: Any, data: Any, geo: Any = None) -> Any:
            hook_called["count"] += 1
            raise RuntimeError("hook assumed non-empty data — must never be called")

        runner = FunctionalModelRunner(
            on_train=_noop_train,
            on_predict=_noop_predict,
            on_validate_train=exploding_hook,
        )
        manager = await _build_manager(runner, db)

        response = await manager.validate(
            ValidateTrainRequest(
                config_id=config_id,
                data=DataFrame(columns=["rainfall"], data=[]),
            )
        )

        assert response.valid is False
        assert hook_called["count"] == 0
        codes = [d.code for d in response.diagnostics]
        assert codes == ["data_empty"]
    finally:
        await db.dispose()


async def test_validate_predict_surfaces_corrupt_model_pickle() -> None:
    """Regression: $validate must not report valid=True when $predict would fail on pickle load."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        # Seed a config so the prediction_periods check is satisfied.
        from chapkit.config import ConfigManager, ConfigRepository
        from chapkit.config.schemas import ConfigIn

        async with db.session() as session:
            config_repo = ConfigRepository(session)
            config_manager: ConfigManager[BaseConfig] = ConfigManager(config_repo, SampleConfig)
            created = await config_manager.save(ConfigIn(name="seed", data=SampleConfig(prediction_periods=3)))
            config_id = created.id

        # Build a ZIP whose model.pickle bytes are not valid pickle data.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.pickle", b"not-a-valid-pickle")
        workspace_bytes = buffer.getvalue()

        artifact_id = await _seed_training_artifact(
            db,
            {
                "type": "ml_training_workspace",
                "metadata": {"status": "success", "config_id": str(config_id)},
                "content": workspace_bytes,
                "content_type": "application/zip",
            },
        )

        manager = await _build_manager(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict), db)

        response = await manager.validate(
            ValidatePredictRequest(
                artifact_id=artifact_id,
                historic=DataFrame(columns=["x"], data=[[1.0]]),
                future=DataFrame(columns=["x"], data=[[2.0]]),
            )
        )

        codes = [d.code for d in response.diagnostics]
        assert "model_pickle_corrupted" in codes
        assert response.valid is False
    finally:
        await db.dispose()
