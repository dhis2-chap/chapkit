import unittest
from uuid import uuid4

from chapkit.storage import SqlAlchemyChapStorage
from chapkit.types import ChapConfig


class TestStorage(unittest.TestCase):
    def test_cascade_delete(self):
        storage = SqlAlchemyChapStorage(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        model_id = uuid4()

        storage.add_config(config)
        storage.add_model(model_id, config, "model")

        self.assertIsNotNone(storage.get_model(model_id))

        storage.del_config(config.id)

        self.assertIsNone(storage.get_model(model_id))

    def test_get_config_for_model(self):
        storage = SqlAlchemyChapStorage(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        model_id = uuid4()

        storage.add_config(config)
        storage.add_model(model_id, config, "model")

        retrieved_config = storage.get_config_for_model(model_id)
        self.assertEqual(config, retrieved_config)

    def test_cascade_delete_multiple_models(self):
        storage = SqlAlchemyChapStorage(ChapConfig, file=":memory:")
        config = ChapConfig(id=uuid4(), name="test")
        model_id1 = uuid4()
        model_id2 = uuid4()

        storage.add_config(config)
        storage.add_model(model_id1, config, "model1")
        storage.add_model(model_id2, config, "model2")

        self.assertIsNotNone(storage.get_model(model_id1))
        self.assertIsNotNone(storage.get_model(model_id2))

        storage.del_config(config.id)

        self.assertIsNone(storage.get_model(model_id1))
        self.assertIsNone(storage.get_model(model_id2))


if __name__ == "__main__":
    unittest.main()
