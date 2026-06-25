"""Tests for the MLRouter $sample-data endpoint."""

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

    response = client.get("/api/v1/ml/$sample-data", params={"kind": "train", "config_id": "cfg-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["config_id"] == "cfg-1"
    assert set(payload["data"].keys()) == {"columns", "data"}
    assert "population" in payload["data"]["columns"]
    assert len(payload["data"]["data"]) > 0


def test_sample_data_predict_returns_historic_and_future() -> None:
    """A predict sample payload contains historic and future DataFrames."""
    client = _client()

    response = client.get("/api/v1/ml/$sample-data", params={"kind": "predict"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["historic"].keys()) == {"columns", "data"}
    assert set(payload["future"].keys()) == {"columns", "data"}


def test_sample_data_honors_tunable_params() -> None:
    """num_locations x num_periods and weekly period_type shape the generated rows."""
    client = _client()

    response = client.get(
        "/api/v1/ml/$sample-data",
        params={"kind": "train", "num_locations": 2, "num_periods": 3, "period_type": "weekly"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["data"]) == 6  # 2 locations x 3 periods
    period_index = data["columns"].index("time_period")
    assert data["data"][0][period_index].startswith("2020-W")


def test_sample_data_includes_geo_when_requested() -> None:
    """include_geo forces a GeoJSON FeatureCollection into the payload."""
    client = _client()

    response = client.get(
        "/api/v1/ml/$sample-data",
        params={"kind": "train", "include_geo": True, "num_locations": 2},
    )

    assert response.status_code == 200
    geo = response.json()["geo"]
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 2


def test_sample_data_omits_geo_by_default() -> None:
    """Without requires_geo or include_geo, no geo is attached."""
    client = _client({"requires_geo": False})

    response = client.get("/api/v1/ml/$sample-data", params={"kind": "train"})

    assert response.status_code == 200
    assert "geo" not in response.json()
