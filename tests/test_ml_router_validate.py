"""Tests for the MLRouter $validate endpoint."""

from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chapkit.ml import MLManager, MLRouter
from chapkit.ml.schemas import ValidationDiagnostic, ValidationResponse


def _client_with_mock(mock_manager: Mock) -> TestClient:
    """Build a TestClient wired to a mock MLManager."""

    def manager_factory() -> MLManager:
        return mock_manager

    app = FastAPI()
    router = MLRouter.create(
        prefix="/api/v1/ml",
        tags=["ML"],
        manager_factory=manager_factory,
    )
    app.include_router(router)
    return TestClient(app)


def test_validate_train_happy_path_returns_200() -> None:
    """Validate train payload with no diagnostics returns 200 and valid=True."""
    mock_manager = Mock(spec=MLManager)
    mock_manager.validate = AsyncMock(return_value=ValidationResponse(valid=True, diagnostics=[]))

    client = _client_with_mock(mock_manager)
    body = {
        "type": "train",
        "config_id": "01K72P5N5KCRM6MD3BRE4P0001",
        "data": {"columns": ["rainfall"], "data": [[1.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"valid": True, "diagnostics": []}


def test_validate_predict_happy_path_returns_200() -> None:
    """Validate predict payload with no diagnostics returns 200 and valid=True."""
    mock_manager = Mock(spec=MLManager)
    mock_manager.validate = AsyncMock(return_value=ValidationResponse(valid=True, diagnostics=[]))

    client = _client_with_mock(mock_manager)
    body = {
        "type": "predict",
        "artifact_id": "01K72P5N5KCRM6MD3BRE4P0001",
        "historic": {"columns": ["rainfall"], "data": [[1.0]]},
        "future": {"columns": ["rainfall"], "data": [[2.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_validate_surfaces_error_diagnostic_with_valid_false() -> None:
    """An error diagnostic from the manager is returned with valid=False."""
    mock_manager = Mock(spec=MLManager)
    mock_manager.validate = AsyncMock(
        return_value=ValidationResponse(
            valid=False,
            diagnostics=[
                ValidationDiagnostic(
                    severity="error",
                    code="data_empty",
                    message="Training data is empty",
                    field="data",
                )
            ],
        )
    )

    client = _client_with_mock(mock_manager)
    body = {
        "type": "train",
        "config_id": "01K72P5N5KCRM6MD3BRE4P0001",
        "data": {"columns": ["rainfall"], "data": []},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert payload["diagnostics"][0]["code"] == "data_empty"
    assert payload["diagnostics"][0]["severity"] == "error"
    assert payload["diagnostics"][0]["field"] == "data"


def test_validate_missing_type_returns_422() -> None:
    """Missing discriminator field is a Pydantic schema error (422)."""
    mock_manager = Mock(spec=MLManager)

    client = _client_with_mock(mock_manager)
    body = {
        "config_id": "01K72P5N5KCRM6MD3BRE4P0001",
        "data": {"columns": ["rainfall"], "data": [[1.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 422


def test_validate_unknown_type_returns_422() -> None:
    """Unknown discriminator value is a Pydantic schema error (422)."""
    mock_manager = Mock(spec=MLManager)

    client = _client_with_mock(mock_manager)
    body = {
        "type": "something-else",
        "config_id": "01K72P5N5KCRM6MD3BRE4P0001",
        "data": {"columns": ["rainfall"], "data": [[1.0]]},
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 422


def test_validate_train_wrong_shape_for_predict_returns_422() -> None:
    """A predict-typed payload missing historic/future is rejected by the discriminator."""
    mock_manager = Mock(spec=MLManager)

    client = _client_with_mock(mock_manager)
    body = {
        "type": "predict",
        "artifact_id": "01K72P5N5KCRM6MD3BRE4P0001",
    }

    response = client.post("/api/v1/ml/$validate", json=body)

    assert response.status_code == 422
