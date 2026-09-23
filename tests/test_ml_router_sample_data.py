"""Tests for the MLRouter $generate-sample-data endpoint."""

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chapkit.ml import MLManager, MLRouter


def _client(sample_metadata: dict | None = None) -> TestClient:
    """Build a TestClient for an MLRouter with optional sample metadata."""

    def manager_factory() -> MLManager:
        return Mock(spec=MLManager)

    app = FastAPI()
    router = MLRouter.create(
        prefix="/api/v1/ml",
        tags=["ML"],
        manager_factory=manager_factory,
        sample_metadata=sample_metadata,
    )
    app.include_router(router)
    return TestClient(app)


def test_sample_data_train_returns_dataframe() -> None:
    """A train sample payload contains a DataFrame and echoes the config id."""
    client = _client({"required_covariates": ["population"], "period_type": "monthly"})

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "train", "config_id": "cfg-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["config_id"] == "cfg-1"
    assert set(payload["data"].keys()) == {"columns", "data", "schema"}
    assert "population" in payload["data"]["columns"]
    assert len(payload["data"]["data"]) > 0
    # The self-describing schema covers every column with contract-derived types.
    fields = payload["data"]["schema"]["fields"]
    by_name = {field["name"]: field["type"] for field in fields}
    assert [field["name"] for field in fields] == payload["data"]["columns"]
    assert by_name["time_period"] == "string"
    assert by_name["population"] == "integer"
    assert by_name["rainfall"] == "number"


def test_sample_data_predict_returns_historic_and_future() -> None:
    """A predict sample payload contains historic and future DataFrames."""
    client = _client()

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "predict"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["historic"].keys()) == {"columns", "data", "schema"}
    assert set(payload["future"].keys()) == {"columns", "data", "schema"}


def test_sample_data_honors_tunable_params() -> None:
    """num_locations x num_periods and weekly period_type shape the generated rows."""
    client = _client()

    response = client.get(
        "/api/v1/ml/$generate-sample-data",
        params={"kind": "train", "num_locations": 2, "num_periods": 3, "period_type": "weekly"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["data"]) == 6  # 2 locations x 3 periods
    period_index = data["columns"].index("time_period")
    assert data["data"][0][period_index].startswith("2020-W")


def test_sample_data_any_period_type_defaults_to_monthly() -> None:
    """A service declaring period_type any gets monthly sample rows unless overridden."""
    client = _client({"period_type": "any"})

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "train", "num_locations": 1})

    assert response.status_code == 200
    data = response.json()["data"]
    period_index = data["columns"].index("time_period")
    assert data["data"][0][period_index] == "2020-01"

    response = client.get(
        "/api/v1/ml/$generate-sample-data",
        params={"kind": "train", "num_locations": 1, "period_type": "weekly"},
    )
    data = response.json()["data"]
    assert data["data"][0][period_index].startswith("2020-W")


def test_sample_data_includes_geo_when_requested() -> None:
    """include_geo forces a GeoJSON FeatureCollection into the payload."""
    client = _client()

    response = client.get(
        "/api/v1/ml/$generate-sample-data",
        params={"kind": "train", "include_geo": True, "num_locations": 2},
    )

    assert response.status_code == 200
    geo = response.json()["geo"]
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 2


def test_sample_data_omits_geo_by_default() -> None:
    """Without requires_geo or include_geo, no geo is attached."""
    client = _client({"requires_geo": False})

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "train"})

    assert response.status_code == 200
    assert "geo" not in response.json()


def _distinct_periods(frame: dict) -> int:
    """Count distinct time_period values in a generated frame."""
    period_index = frame["columns"].index("time_period")
    return len({row[period_index] for row in frame["data"]})


def test_sample_data_predict_clamps_future_to_max_prediction_periods() -> None:
    """The default 50-period future is shortened to the service's declared maximum; historic keeps 50."""
    client = _client({"min_prediction_periods": 0, "max_prediction_periods": 12})

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "predict"})

    assert response.status_code == 200
    payload = response.json()
    assert _distinct_periods(payload["future"]) == 12
    assert _distinct_periods(payload["historic"]) == 50


def test_sample_data_predict_raises_future_to_min_prediction_periods() -> None:
    """A requested horizon below the service minimum is raised to the minimum."""
    client = _client({"min_prediction_periods": 4, "max_prediction_periods": 100})

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "predict", "num_periods": 2})

    assert response.status_code == 200
    assert _distinct_periods(response.json()["future"]) == 4


def test_sample_data_train_is_not_clamped_by_prediction_bounds() -> None:
    """Training data length is unrelated to the forecast horizon and keeps the requested size."""
    client = _client({"min_prediction_periods": 0, "max_prediction_periods": 12})

    response = client.get("/api/v1/ml/$generate-sample-data", params={"kind": "train", "num_periods": 30})

    assert response.status_code == 200
    assert _distinct_periods(response.json()["data"]) == 30
