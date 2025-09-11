from uuid import uuid4
import pandas as pd
from fastapi.testclient import TestClient

from chapkit.model.runner import ChapModelRunner
from chapkit.model.service import ChapModelService
from chapkit.database import ChapDatabase
from chapkit.types import ChapConfig, PredictParams, TrainParams, HealthResponse, HealthStatus


class MockModelRunner(ChapModelRunner):
    def on_predict(self, params: PredictParams):
        pass

    def on_train(self, params: TrainParams):
        pass

    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)


class MockDatabase(ChapDatabase):
    def get_configs(self):
        return []

    def get_config(self, id):
        return None

    def add_config(self, cfg):
        pass

    def del_config(self, id):
        return True

    def update_config(self, cfg):
        return True

    def add_artifact(self, id, cfg, obj, parent_id=None):
        pass

    def get_artifact(self, artifact_id):
        return "mock_artifact"

    def get_artifacts_for_config(self, config_id):
        return []

    def del_artifact(self, artifact_id):
        return True

    def get_config_for_artifact(self, artifact_id):
        return ChapConfig(id=uuid4(), name="test_config")

    def get_artifact_row(self, id):
        return None

    def get_artifact_rows_for_config(self, config_id):
        return []

    def get_artifact_roots_for_config(self, config_id):
        return []


def test_predict_endpoint():
    database = MockDatabase()
    runner = MockModelRunner(None, database, config_type=ChapConfig)
    service = ChapModelService(runner, database)
    app = service.create_fastapi()
    client = TestClient(app)

    historic_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    future_df = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})

    payload = {
        "historic": historic_df.to_dict(orient="split"),
        "future": future_df.to_dict(orient="split"),
    }

    response = client.post(f"/api/v1/predict?artifact={uuid4()}", json=payload)

    assert response.status_code == 202
    response_json = response.json()
    assert "id" in response_json
    assert response_json["type"] == "predict"
    assert response_json["status"] == "pending"
