"""Config repository tests."""

from __future__ import annotations

from conftest import DemoConfig

from chapkit import Config, ConfigRepository, SqliteDatabaseBuilder


async def test_config_repository_find_by_name_round_trip() -> None:
    """ConfigRepository.find_by_name should return matching rows and None otherwise."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    async with db.session() as session:
        repo = ConfigRepository(session)

        assert await repo.find_by_name("missing") is None

        created = Config(name="feature", data=DemoConfig(x=1, y=2, z=3, tags=["feature"]))
        await repo.save(created)
        await repo.commit()
        await repo.refresh_many([created])

        found = await repo.find_by_name("feature")
        assert found is not None
        assert found.id == created.id
        assert found.name == "feature"
        assert found.data == {"x": 1, "y": 2, "z": 3, "tags": ["feature"]}

    await db.dispose()
