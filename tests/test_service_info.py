from fastapi.testclient import TestClient

from chapkit.runner import ChapRunner
from chapkit.service import ChapService
from chapkit.storage import ChapStorage
from chapkit.types import ChapConfig, ChapServiceInfo


class MockRunner(ChapRunner):
    pass


class MockStorage(ChapStorage):
    def get_configs(self):
        return []

    def get_config(self, id):
        return None

    def add_config(self, cfg):
        pass

    def del_config(self, id):
        return True


def test_info_endpoint():
    info = ChapServiceInfo(display_name="Test Service")
    runner = MockRunner(info, ChapConfig)
    storage = MockStorage()
    service = ChapService(runner, storage)
    app = service.create_fastapi()
    client = TestClient(app)
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    assert response.json() == {"display_name": "Test Service"}
