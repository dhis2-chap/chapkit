from fastapi.testclient import TestClient

from chapkit.runner import ChapRunner
from chapkit.service import ChapService
from chapkit.database import ChapDatabase
from chapkit.types import ChapConfig, ChapServiceInfo


class MockRunner(ChapRunner):
    pass


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


    def add_artifact(self, config_id, artifact):
        pass

    def get_artifact(self, artifact_id):
        return None

    def get_artifacts_for_config(self, config_id):
        return []

    def del_artifact(self, artifact_id):
        return True

    def get_config_for_artifact(self, artifact_id):
        return None


def test_info_endpoint():
    info = ChapServiceInfo(display_name="Test Service")
    runner = MockRunner(info, ChapConfig)
    database = MockDatabase()
    service = ChapService(runner, database)
    app = service.create_fastapi()
    client = TestClient(app)
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    assert response.json() == {"display_name": "Test Service"}
