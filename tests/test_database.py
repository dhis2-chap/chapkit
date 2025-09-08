import unittest
from uuid import uuid4

from chapkit.database import SqlAlchemyChapDatabase
from chapkit.types import ChapConfig


class TestDatabase(unittest.TestCase):
    def test_cascade_delete(self):
        database = SqlAlchemyChapDatabase(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        artifact_id = uuid4()

        database.add_config(config)
        database.add_artifact(artifact_id, config, "artifact")

        self.assertIsNotNone(database.get_artifact(artifact_id))

        database.del_config(config.id)

        self.assertIsNone(database.get_artifact(artifact_id))

    def test_get_config_for_artifact(self):
        database = SqlAlchemyChapDatabase(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        artifact_id = uuid4()

        database.add_config(config)
        database.add_artifact(artifact_id, config, "artifact")

        retrieved_config = database.get_config_for_artifact(artifact_id)
        self.assertEqual(config, retrieved_config)

    def test_cascade_delete_multiple_artifacts(self):
        database = SqlAlchemyChapDatabase(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        artifact_id1 = uuid4()
        artifact_id2 = uuid4()

        database.add_config(config)
        database.add_artifact(artifact_id1, config, "artifact1")
        database.add_artifact(artifact_id2, config, "artifact2")

        self.assertIsNotNone(database.get_artifact(artifact_id1))
        self.assertIsNotNone(database.get_artifact(artifact_id2))

        database.del_config(config.id)

        self.assertIsNone(database.get_artifact(artifact_id1))
        self.assertIsNone(database.get_artifact(artifact_id2))

    def test_get_artifacts_for_config(self):
        database = SqlAlchemyChapDatabase(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        artifact_id1 = uuid4()
        artifact_id2 = uuid4()
        artifact_data1 = {"type": "model", "version": "1.0"}
        artifact_data2 = {"type": "model", "version": "2.0"}

        database.add_config(config)
        database.add_artifact(artifact_id1, config, artifact_data1)
        database.add_artifact(artifact_id2, config, artifact_data2)

        artifacts = database.get_artifacts_for_config(config.id)

        self.assertEqual(len(artifacts), 2)
        artifact_ids = [artifact_id for artifact_id, _ in artifacts]
        self.assertIn(artifact_id1, artifact_ids)
        self.assertIn(artifact_id2, artifact_ids)

    def test_get_artifacts_for_nonexistent_config(self):
        database = SqlAlchemyChapDatabase(ChapConfig, file=":memory:")
        nonexistent_config_id = uuid4()

        artifacts = database.get_artifacts_for_config(nonexistent_config_id)
        self.assertEqual(len(artifacts), 0)

    def test_artifact_crud_operations(self):
        database = SqlAlchemyChapDatabase(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        artifact_id = uuid4()
        artifact_data = {"weights": [1, 2, 3], "metadata": {"accuracy": 0.95}}

        database.add_config(config)

        # Test add
        database.add_artifact(artifact_id, config, artifact_data)
        retrieved_artifact = database.get_artifact(artifact_id)
        self.assertEqual(retrieved_artifact, artifact_data)

        # Test delete
        result = database.del_artifact(artifact_id)
        self.assertTrue(result)
        self.assertIsNone(database.get_artifact(artifact_id))

        # Test delete nonexistent
        result = database.del_artifact(artifact_id)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
