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
from chapkit.ml.manager import resolve_prediction_periods
from chapkit.ml.schemas import (
    PredictRequest,
    RunInfo,
    TrainRequest,
    ValidatePredictRequest,
    ValidationDiagnostic,
)


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
        assert manager._check_prediction_periods(5, "config") == []

    def test_below_minimum_returns_error_diagnostic(self) -> None:
        manager = self._bare_manager(minimum=5, maximum=10)
        diagnostics = manager._check_prediction_periods(3, "config")
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].code == "prediction_periods_out_of_bounds"
        assert diagnostics[0].field == "config.prediction_periods"
        assert "below the minimum" in diagnostics[0].message

    def test_above_maximum_returns_error_diagnostic(self) -> None:
        manager = self._bare_manager(minimum=1, maximum=5)
        diagnostics = manager._check_prediction_periods(10, "config")
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].code == "prediction_periods_out_of_bounds"
        assert "exceeds the maximum" in diagnostics[0].message

    def test_at_boundaries_returns_empty(self) -> None:
        manager = self._bare_manager(minimum=3, maximum=7)
        assert manager._check_prediction_periods(3, "config") == []
        assert manager._check_prediction_periods(7, "config") == []

    def test_run_info_source_reports_run_info_field(self) -> None:
        """A horizon that came from run info blames run_info.prediction_periods."""
        manager = self._bare_manager(minimum=1, maximum=10)
        diagnostics = manager._check_prediction_periods(12, "run_info")
        assert len(diagnostics) == 1
        assert diagnostics[0].field == "run_info.prediction_periods"
        assert diagnostics[0].message == (
            "prediction_periods (12, from run_info) exceeds the maximum allowed value (10)"
        )

    def test_future_source_reports_future_field(self) -> None:
        """A horizon derived from the future frame blames the future frame."""
        manager = self._bare_manager(minimum=4, maximum=10)
        diagnostics = manager._check_prediction_periods(2, "future")
        assert len(diagnostics) == 1
        assert diagnostics[0].field == "future"
        assert "from future" in diagnostics[0].message


class TestRunInfoSchema:
    """Unit tests for the RunInfo request schema."""

    def test_accepts_chap_core_prediction_length_alias(self) -> None:
        """chap-core sends prediction_length; it populates prediction_periods."""
        run_info = RunInfo.model_validate({"prediction_length": 7})
        assert run_info.prediction_periods == 7

    def test_accepts_canonical_prediction_periods(self) -> None:
        """The canonical spelling validates just as well."""
        run_info = RunInfo.model_validate({"prediction_periods": 7})
        assert run_info.prediction_periods == 7

    def test_dumps_under_canonical_name(self) -> None:
        """Serialization always emits prediction_periods, never the legacy alias."""
        dumped = RunInfo.model_validate({"prediction_length": 7}).model_dump(by_alias=True)
        assert dumped["prediction_periods"] == 7
        assert "prediction_length" not in dumped

    def test_ignores_unknown_keys(self) -> None:
        """Unknown keys chap-core may add later are dropped, not rejected."""
        run_info = RunInfo.model_validate({"prediction_length": 3, "some_future_key": "value"})
        assert run_info.prediction_periods == 3
        assert not hasattr(run_info, "some_future_key")

    def test_defaults_are_empty(self) -> None:
        """An empty run info carries no horizon and no covariates."""
        run_info = RunInfo()
        assert run_info.prediction_periods is None
        assert run_info.additional_continuous_covariates == []
        assert run_info.future_covariate_origin is None

    def test_carries_covariates_and_origin(self) -> None:
        """The remaining chap-core fields round-trip unchanged."""
        run_info = RunInfo.model_validate(
            {
                "prediction_length": 2,
                "additional_continuous_covariates": ["rainfall"],
                "future_covariate_origin": "climate_model",
            }
        )
        assert run_info.additional_continuous_covariates == ["rainfall"]
        assert run_info.future_covariate_origin == "climate_model"


class TestResolvePredictionPeriods:
    """Unit tests for the pure prediction-horizon resolution helper."""

    def _future(self, periods_by_location: dict[str, int], *, with_location: bool = True) -> DataFrame:
        """Build a future frame with the given number of periods per location."""
        columns = ["time_period", "location"] if with_location else ["time_period"]
        rows: list[list[Any]] = []
        for location, period_count in periods_by_location.items():
            for period_index in range(period_count):
                period = f"2020-{period_index + 1:02d}"
                rows.append([period, location] if with_location else [period])
        return DataFrame(columns=columns, data=rows)

    def test_run_info_wins_over_future_and_config(self) -> None:
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            RunInfo(prediction_periods=9),
            self._future({"a": 4}),
        )
        assert (resolved, source) == (9, "run_info")

    def test_future_wins_over_config_when_run_info_absent(self) -> None:
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            None,
            self._future({"a": 4}),
        )
        assert (resolved, source) == (4, "future")

    def test_future_used_when_run_info_omits_the_horizon(self) -> None:
        """run_info without prediction_periods must not shadow the future frame."""
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            RunInfo(additional_continuous_covariates=["rainfall"]),
            self._future({"a": 4}),
        )
        assert (resolved, source) == (4, "future")

    def test_future_takes_the_max_across_locations(self) -> None:
        """A ragged panel resolves to the longest per-location horizon, not the row count."""
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            None,
            self._future({"a": 6, "b": 4, "c": 6}),
        )
        assert (resolved, source) == (6, "future")

    def test_future_without_location_column_counts_distinct_periods(self) -> None:
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            None,
            self._future({"a": 5}, with_location=False),
        )
        assert (resolved, source) == (5, "future")

    def test_future_without_time_period_column_falls_back_to_config(self) -> None:
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            None,
            DataFrame(columns=["location", "rainfall"], data=[["a", 1.0]]),
        )
        assert (resolved, source) == (3, "config")

    def test_empty_future_falls_back_to_config(self) -> None:
        resolved, source = resolve_prediction_periods(
            SampleConfig(prediction_periods=3),
            None,
            DataFrame(columns=["time_period", "location"], data=[]),
        )
        assert (resolved, source) == (3, "config")

    def test_no_future_and_no_run_info_falls_back_to_config(self) -> None:
        """Train has no future frame, so the config value is all that is left."""
        resolved, source = resolve_prediction_periods(SampleConfig(prediction_periods=3), None, None)
        assert (resolved, source) == (3, "config")


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
    """Spin up the class-based runner fixture app for validate integration tests."""
    from tests.fixtures.class_runner_app import build_class_runner_app

    with TestClient(build_class_runner_app()) as test_client:
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


def test_validate_predict_warns_when_run_info_and_future_disagree(client: TestClient) -> None:
    """A run_info horizon that contradicts the future frame is a warning, not an error."""
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
        "historic": {
            "columns": ["time_period", "location", "rainfall", "disease_cases"],
            "data": [["2019-12", "location_0", 10.0, 5.0]],
        },
        # Two future periods, but chap-core claims five.
        "future": {
            "columns": ["time_period", "location", "rainfall"],
            "data": [
                ["2020-01", "location_0", 11.0],
                ["2020-02", "location_0", 12.0],
            ],
        },
        "run_info": {"prediction_length": 5},
    }

    response = client.post("/api/v1/ml/$validate", json=validate_body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True, payload
    mismatches = [d for d in payload["diagnostics"] if d["code"] == "prediction_periods_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0]["severity"] == "warning"
    assert mismatches[0]["field"] == "future"
    assert "5" in mismatches[0]["message"]
    assert "2" in mismatches[0]["message"]


def test_validate_predict_no_warning_when_run_info_matches_future(client: TestClient) -> None:
    """Agreement between run_info and the future frame produces no diagnostic."""
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
        "historic": {
            "columns": ["time_period", "location", "rainfall", "disease_cases"],
            "data": [["2019-12", "location_0", 10.0, 5.0]],
        },
        "future": {
            "columns": ["time_period", "location", "rainfall"],
            "data": [
                ["2020-01", "location_0", 11.0],
                ["2020-02", "location_0", 12.0],
            ],
        },
        "run_info": {"prediction_periods": 2},
    }

    response = client.post("/api/v1/ml/$validate", json=validate_body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True, payload
    assert [d for d in payload["diagnostics"] if d["code"] == "prediction_periods_mismatch"] == []


def test_validate_train_rejects_a_run_info_horizon_above_the_maximum(client: TestClient) -> None:
    """The train bounds check runs on the resolved horizon and names run_info as its source."""
    config_id = _create_config(client)
    body = {
        "type": "train",
        "config_id": config_id,
        "data": {
            "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
            "data": [[10.0, 25.0, 60.0, 5.0]],
        },
        "run_info": {"prediction_length": 200},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    out_of_bounds = [d for d in payload["diagnostics"] if d["code"] == "prediction_periods_out_of_bounds"]
    assert len(out_of_bounds) == 1
    assert out_of_bounds[0]["field"] == "run_info.prediction_periods"
    assert "from run_info" in out_of_bounds[0]["message"]


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
    # ClassRunnerConfig.min_samples defaults to 5; FixtureModelRunner's
    # on_validate_train override (see tests/fixtures/class_runner_app.py) emits an
    # error diagnostic when data has fewer rows than min_samples.
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


async def test_validate_predict_invalid_artifact_type() -> None:
    """Artifact with wrong type (not ml_training_workspace) yields invalid_training_artifact."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        artifact_id = await _seed_training_artifact(db, {"type": "generic", "metadata": {}})
        manager = await _build_manager(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict), db)
        response = await manager.validate(
            ValidatePredictRequest(
                artifact_id=artifact_id,
                historic=DataFrame(columns=["x"], data=[[1.0]]),
                future=DataFrame(columns=["x"], data=[[2.0]]),
            )
        )
        assert response.valid is False
        assert response.diagnostics[0].code == "invalid_training_artifact"
    finally:
        await db.dispose()


async def test_validate_predict_failed_training_artifact() -> None:
    """Failed training artifact yields training_artifact_failed."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        artifact_id = await _seed_training_artifact(
            db,
            {
                "type": "ml_training_workspace",
                "metadata": {"status": "failed", "exit_code": 1},
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
        assert response.valid is False
        assert response.diagnostics[0].code == "training_artifact_failed"
    finally:
        await db.dispose()


async def test_validate_predict_missing_config_id_in_metadata() -> None:
    """Training artifact without config_id in metadata yields invalid_training_artifact."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        artifact_id = await _seed_training_artifact(
            db, {"type": "ml_training_workspace", "metadata": {"status": "success"}}
        )
        manager = await _build_manager(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict), db)
        response = await manager.validate(
            ValidatePredictRequest(
                artifact_id=artifact_id,
                historic=DataFrame(columns=["x"], data=[[1.0]]),
                future=DataFrame(columns=["x"], data=[[2.0]]),
            )
        )
        assert response.valid is False
        assert response.diagnostics[0].code == "invalid_training_artifact"
    finally:
        await db.dispose()


async def test_validate_predict_config_deleted_after_training() -> None:
    """Config referenced by training artifact no longer exists yields config_not_found."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        nonexistent_config_id = ULID()
        artifact_id = await _seed_training_artifact(
            db,
            {
                "type": "ml_training_workspace",
                "metadata": {"status": "success", "config_id": str(nonexistent_config_id)},
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
        assert response.valid is False
        assert response.diagnostics[0].code == "config_not_found"
    finally:
        await db.dispose()


def test_validation_diagnostic_info_classmethod() -> None:
    """Codecov: exercise the .info() classmethod."""
    diag = ValidationDiagnostic.info(code="using_defaults", message="No custom config; using defaults")
    assert diag.severity == "info"
    assert diag.code == "using_defaults"
    assert diag.field is None


class _LegacyRunner:
    """Simulates a pre-PR runner that implements on_train/on_predict but not on_validate_*."""

    async def on_train(self, config: Any, data: Any, geo: Any = None) -> Any:
        return {}

    async def on_predict(self, config: Any, model: Any, historic: Any, future: Any, geo: Any = None) -> Any:
        return future


async def test_validate_train_with_legacy_runner_missing_hook() -> None:
    """A runner that predates the validate hooks must not cause AttributeError."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        from chapkit.config import ConfigManager, ConfigRepository
        from chapkit.config.schemas import ConfigIn
        from chapkit.ml.schemas import ValidateTrainRequest

        async with db.session() as session:
            config_manager = ConfigManager(ConfigRepository(session), SampleConfig)
            created = await config_manager.save(ConfigIn(name="legacy", data=SampleConfig(prediction_periods=3)))
            config_id = created.id

        manager = await _build_manager(_LegacyRunner(), db)

        response = await manager.validate(
            ValidateTrainRequest(
                config_id=config_id,
                data=DataFrame(columns=["rainfall"], data=[[1.0]]),
            )
        )
        assert response.valid is True
        assert response.diagnostics == []
    finally:
        await db.dispose()


async def _seed_config(db: Any, config_data: BaseConfig) -> ULID:
    """Store a config and return its ID."""
    from chapkit.config import ConfigManager, ConfigRepository
    from chapkit.config.schemas import ConfigIn

    async with db.session() as session:
        config_repository = ConfigRepository(session)
        config_manager: ConfigManager[BaseConfig] = ConfigManager(config_repository, type(config_data))
        created = await config_manager.save(ConfigIn(name=f"horizon_config_{ULID()}", data=config_data))
        return created.id


def _future_frame(num_periods: int, locations: tuple[str, ...] = ("location_0", "location_1")) -> DataFrame:
    """Build a future frame spanning num_periods periods for each location."""
    rows: list[list[Any]] = []
    for period_index in range(num_periods):
        for location in locations:
            rows.append([f"2020-{period_index + 1:02d}", location, 1.0])
    return DataFrame(columns=["time_period", "location", "rainfall"], data=rows)


async def test_train_task_rejects_a_run_info_horizon_above_the_maximum() -> None:
    """The bounds check runs on the resolved horizon, so run info can fail a config that passes."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        config_id = await _seed_config(db, SampleConfig(prediction_periods=3))
        manager = await _build_manager(FunctionalModelRunner(on_train=_noop_train, on_predict=_noop_predict), db)

        with pytest.raises(ValueError) as exc_info:
            await manager._train_task(
                TrainRequest(
                    config_id=config_id,
                    data=DataFrame(columns=["rainfall"], data=[[1.0]]),
                    run_info=RunInfo(prediction_periods=200),
                ),
                ULID(),
            )

        assert "prediction_periods (200, from run_info)" in str(exc_info.value)
        assert "exceeds the maximum allowed value (100)" in str(exc_info.value)
    finally:
        await db.dispose()


async def test_tasks_hand_the_runner_the_resolved_horizon(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_info wins at train and predict, the future frame is the predict fallback, and the config is untouched."""
    monkeypatch.chdir(tmp_path)
    seen: dict[str, int] = {}

    async def record_train(config: Any, data: Any, geo: Any = None) -> Any:
        """Record the horizon the runner was given and return a trivial model."""
        seen["train"] = config.prediction_periods
        return {"weights": [1.0]}

    async def record_predict(config: Any, model: Any, historic: Any, future: Any, geo: Any = None) -> Any:
        """Record the horizon the runner was given and echo the future frame."""
        seen["predict"] = config.prediction_periods
        return future

    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        # The stored horizon is out of bounds on purpose: run info must be what gets checked.
        config_id = await _seed_config(db, SampleConfig(prediction_periods=200))
        manager = await _build_manager(FunctionalModelRunner(on_train=record_train, on_predict=record_predict), db)

        training_artifact_id = ULID()
        await manager._train_task(
            TrainRequest(
                config_id=config_id,
                data=DataFrame(columns=["rainfall", "disease_cases"], data=[[1.0, 2.0]]),
                run_info=RunInfo(prediction_periods=5),
            ),
            training_artifact_id,
        )
        assert seen["train"] == 5

        historic = DataFrame(columns=["time_period", "location", "rainfall"], data=[["2019-12", "location_0", 1.0]])

        await manager._predict_task(
            PredictRequest(
                artifact_id=training_artifact_id,
                historic=historic,
                future=_future_frame(4),
                run_info=RunInfo(prediction_periods=6),
            ),
            ULID(),
        )
        assert seen["predict"] == 6

        # Without run info the future frame decides: four periods per location.
        await manager._predict_task(
            PredictRequest(
                artifact_id=training_artifact_id,
                historic=historic,
                future=_future_frame(4),
            ),
            ULID(),
        )
        assert seen["predict"] == 4

        # The stored config is never rewritten by a request.
        from chapkit.config import ConfigManager, ConfigRepository

        async with db.session() as session:
            config_manager: ConfigManager[BaseConfig] = ConfigManager(ConfigRepository(session), SampleConfig)
            stored = await config_manager.find_by_id(config_id)
        assert stored is not None
        assert stored.data.prediction_periods == 200
    finally:
        await db.dispose()


async def test_train_task_without_run_info_uses_the_config_horizon(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a request without run info behaves exactly as before."""
    monkeypatch.chdir(tmp_path)
    seen: dict[str, int] = {}

    async def record_train(config: Any, data: Any, geo: Any = None) -> Any:
        """Record the horizon the runner was given and return a trivial model."""
        seen["train"] = config.prediction_periods
        return {"weights": [1.0]}

    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        config_id = await _seed_config(db, SampleConfig(prediction_periods=7))
        manager = await _build_manager(FunctionalModelRunner(on_train=record_train, on_predict=_noop_predict), db)

        await manager._train_task(
            TrainRequest(
                config_id=config_id,
                data=DataFrame(columns=["rainfall", "disease_cases"], data=[[1.0, 2.0]]),
            ),
            ULID(),
        )

        assert seen["train"] == 7
    finally:
        await db.dispose()


async def test_validate_hooks_receive_the_resolved_horizon() -> None:
    """on_validate_train and on_validate_predict see run_info's horizon, not the stored config value."""
    from chapkit.ml.schemas import ValidateTrainRequest

    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()
    try:
        config_id = await _seed_config(db, SampleConfig(prediction_periods=3))
        seen: dict[str, int] = {}

        async def train_hook(config: Any, data: Any, geo: Any = None) -> list[ValidationDiagnostic]:
            seen["train"] = config.prediction_periods
            return []

        async def predict_hook(config: Any, historic: Any, future: Any, geo: Any = None) -> list[ValidationDiagnostic]:
            seen["predict"] = config.prediction_periods
            return []

        runner = FunctionalModelRunner(
            on_train=_noop_train,
            on_predict=_noop_predict,
            on_validate_train=train_hook,
            on_validate_predict=predict_hook,
        )
        manager = await _build_manager(runner, db)

        await manager.validate(
            ValidateTrainRequest(
                config_id=config_id,
                data=DataFrame(columns=["rainfall"], data=[[1.0]]),
                run_info=RunInfo(prediction_periods=7),
            )
        )

        artifact_id = await _seed_training_artifact(
            db,
            {
                "type": "ml_training_workspace",
                "metadata": {"status": "success", "config_id": str(config_id)},
                "content": _zip_with_pickle({"trained": True}),
                "content_type": "application/zip",
            },
        )
        future = DataFrame(
            columns=["time_period", "location"],
            data=[["2020-01", "a"], ["2020-02", "a"], ["2020-03", "a"], ["2020-04", "a"]],
        )
        await manager.validate(
            ValidatePredictRequest(
                artifact_id=artifact_id,
                historic=DataFrame(columns=["time_period", "location"], data=[["2019-12", "a"]]),
                future=future,
            )
        )

        assert seen == {"train": 7, "predict": 4}

        stored = await manager.validate(
            ValidateTrainRequest(config_id=config_id, data=DataFrame(columns=["rainfall"], data=[[1.0]]))
        )
        assert stored.valid is True
        assert seen["train"] == 3
    finally:
        await db.dispose()
