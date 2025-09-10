from fastapi.testclient import TestClient

from chapkit.runner import ChapRunner
from chapkit.service import ChapService
from chapkit.database import ChapDatabase
from chapkit.types import AssessedStatus, ChapConfig, ChapServiceInfo


class MockRunner(ChapRunner):
    def __init__(self, info, config_type):
        super().__init__(info, config_type=config_type)


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


def test_health_check():
    info = ChapServiceInfo(
        display_name="Test Service",
        author="Test Author",
        author_note="Test Note",
        author_assessed_status=AssessedStatus.gray,
        contact_email="test@example.com",
        description="Test Description",
        organization="Test Organization",
        organization_logo_url="https://example.com/logo.png",
        citation_info="Test Citation",
    )
    runner = MockRunner(info, config_type=ChapConfig)
    database = MockDatabase()
    service = ChapService(runner, database)
    app = service.create_fastapi()
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up"}
