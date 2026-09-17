from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_core.core_schema import ValidationInfo
from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import BaseConfig, Config, ConfigOut

from .conftest import DemoConfig


class DemoConfigModel:
    """Tests for the Config model."""

    async def test_create_config_with_name_and_data(self) -> None:
        """Test creating a Config with name and data."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            test_data = DemoConfig(x=1, y=2, z=3, tags=["test"])
            config = Config(name="test_config", data=test_data)
            session.add(config)
            await session.commit()
            await session.refresh(config)

            assert config.id is not None
            assert isinstance(config.id, ULID)
            assert config.name == "test_config"
            config_data = config.data
            assert config_data is not None
            assert isinstance(config_data, dict)
            assert config_data["x"] == 1
            assert config_data["y"] == 2
            assert config_data["z"] == 3
            assert config_data["tags"] == ["test"]
            assert config.created_at is not None
            assert config.updated_at is not None

        await db.dispose()


def test_config_data_setter_rejects_invalid_type() -> None:
    """Config.data setter should reject unsupported types."""
    config = Config(name="invalid_type")
    with pytest.raises(TypeError):
        bad_value = cast(DemoConfig, 123)
        config.data = bad_value


def test_config_data_setter_accepts_dict() -> None:
    """Config.data setter should accept dict values."""
    config = Config(name="dict_data", data={})
    config.data = DemoConfig(x=1, y=2, z=3, tags=["test"])
    assert config.data["x"] == 1
    assert config.data["y"] == 2
    assert config.data["z"] == 3
    assert config.data["tags"] == ["test"]


def test_config_out_retains_dict_without_context() -> None:
    """ConfigOut should leave dict data unchanged when no context is provided."""
    payload = {"x": 1, "y": 2, "z": 3, "tags": ["raw"]}
    info = cast(ValidationInfo, SimpleNamespace(context=None))
    result = ConfigOut[DemoConfig].convert_dict_to_model(payload, info)
    assert result == payload


class TestConfigModelExtras:
    async def test_create_config_with_empty_data(self) -> None:
        """Test creating a Config with empty dict data."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config = Config(name="empty_config", data={})
            session.add(config)
            await session.commit()
            await session.refresh(config)

            assert config.id is not None
            assert config.name == "empty_config"
            assert config.data == {}

        await db.dispose()

    async def test_config_name_allows_duplicates(self) -> None:
        """Test that Config name field allows duplicates (no unique constraint)."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config1 = Config(name="duplicate_name", data=DemoConfig(x=1, y=2, z=3, tags=[]))
            session.add(config1)
            await session.commit()
            await session.refresh(config1)

        # Create another config with the same name - should succeed
        async with db.session() as session:
            config2 = Config(name="duplicate_name", data=DemoConfig(x=4, y=5, z=6, tags=[]))
            session.add(config2)
            await session.commit()
            await session.refresh(config2)

            # Verify they have different IDs but same name
            assert config1.id != config2.id
            assert config1.name == config2.name == "duplicate_name"

        await db.dispose()

    async def test_config_type_preservation(self) -> None:
        """Test Config stores data as dict and can be deserialized by application."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        test_data = DemoConfig(x=10, y=20, z=30, tags=["a", "b"])

        async with db.session() as session:
            config = Config(name="type_test", data=test_data)
            session.add(config)
            await session.commit()
            await session.refresh(config)

            # The data should be stored as dict
            config_data = config.data
            assert config_data is not None
            assert isinstance(config_data, dict)
            assert config_data["x"] == 10
            assert config_data["y"] == 20
            assert config_data["z"] == 30
            assert config_data["tags"] == ["a", "b"]

            # Application can deserialize it
            deserialized = DemoConfig.model_validate(config_data)
            assert deserialized.x == 10
            assert deserialized.y == 20
            assert deserialized.z == 30
            assert deserialized.tags == ["a", "b"]

        await db.dispose()

    async def test_config_id_is_ulid(self) -> None:
        """Test that Config ID is a ULID type."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config = Config(name="ulid_test", data={})
            session.add(config)
            await session.commit()
            await session.refresh(config)

            assert isinstance(config.id, ULID)
            # ULID string representation should be 26 characters
            assert len(str(config.id)) == 26

        await db.dispose()

    async def test_config_timestamps_auto_set(self) -> None:
        """Test that created_at and updated_at are automatically set."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config = Config(name="timestamp_test", data=DemoConfig(x=1, y=2, z=3, tags=[]))
            session.add(config)
            await session.commit()
            await session.refresh(config)

            assert config.created_at is not None
            assert config.updated_at is not None
            # Initially, created_at and updated_at should be very close
            time_diff = abs((config.updated_at - config.created_at).total_seconds())
            assert time_diff < 1  # Less than 1 second difference

        await db.dispose()

    async def test_config_update_modifies_data(self) -> None:
        """Test updating Config data field."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config = Config(name="update_test", data=DemoConfig(x=1, y=2, z=3, tags=["original"]))
            session.add(config)
            await session.commit()
            await session.refresh(config)

            original_id = config.id

            # Update the data
            config.data = DemoConfig(x=10, y=20, z=30, tags=["updated"])
            await session.commit()
            await session.refresh(config)

            assert config.id == original_id
            config_data = config.data
            assert config_data is not None
            assert isinstance(config_data, dict)
            assert config_data["x"] == 10
            assert config_data["y"] == 20
            assert config_data["z"] == 30
            assert config_data["tags"] == ["updated"]

        await db.dispose()

    async def test_config_tablename(self) -> None:
        """Test that Config uses correct table name."""
        assert Config.__tablename__ == "configs"

    async def test_config_inherits_from_entity(self) -> None:
        """Test that Config inherits from Entity and has expected fields."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config = Config(name="inheritance_test", data={})
            session.add(config)
            await session.commit()
            await session.refresh(config)

            # Check inherited fields from Entity
            assert hasattr(config, "id")
            assert hasattr(config, "created_at")
            assert hasattr(config, "updated_at")

            # Check Config-specific fields
            assert hasattr(config, "name")
            assert hasattr(config, "data")

        await db.dispose()

    async def test_multiple_configs_different_names(self) -> None:
        """Test creating multiple configs with different names."""
        db = SqliteDatabaseBuilder.in_memory().build()
        await db.init()

        async with db.session() as session:
            config1 = Config(name="config_1", data=DemoConfig(x=1, y=1, z=1, tags=[]))
            config2 = Config(name="config_2", data=DemoConfig(x=2, y=2, z=2, tags=[]))
            config3 = Config(name="config_3", data=DemoConfig(x=3, y=3, z=3, tags=[]))

            session.add_all([config1, config2, config3])
            await session.commit()
            await session.refresh(config1)
            await session.refresh(config2)
            await session.refresh(config3)

            assert config1.name == "config_1"
            assert config2.name == "config_2"
            assert config3.name == "config_3"

            # Each should have unique IDs
            assert config1.id != config2.id
            assert config2.id != config3.id
            assert config1.id != config3.id

        await db.dispose()


class TunableConfig(BaseConfig):
    """Config with two declared tunables, as a chapkit service would declare them."""

    prediction_periods: int = 3
    n_samples: int = 100
    log_transform: bool = True


def test_base_config_prediction_periods_defaults_to_three() -> None:
    """BaseConfig itself validates without prediction_periods, as chap-core never sends it."""
    config = BaseConfig()

    assert config.prediction_periods == 3
    assert config.additional_continuous_covariates == []


def test_base_config_hoists_into_inherited_prediction_periods() -> None:
    """A subclass that does not redeclare prediction_periods still receives it from the nested payload."""

    class InheritingConfig(BaseConfig):
        """Config relying on the base default for prediction_periods."""

        n_samples: int = 100

    default = InheritingConfig.model_validate({"user_option_values": {"n_samples": 7}})
    explicit = InheritingConfig.model_validate({"user_option_values": {"prediction_periods": 12}})

    assert default.prediction_periods == 3
    assert default.n_samples == 7
    assert explicit.prediction_periods == 12


def test_base_config_hoists_nested_user_option_values() -> None:
    """chap-core's nested user_option_values payload populates the declared fields."""
    config = TunableConfig.model_validate(
        {"name": "cfg_from_chap_core", "user_option_values": {"n_samples": 7, "log_transform": False}}
    )

    assert config.n_samples == 7
    assert config.log_transform is False
    assert config.prediction_periods == 3
    assert "user_option_values" not in config.model_dump()


def test_base_config_flat_fields_win_over_nested() -> None:
    """A flat key takes precedence over the same key under user_option_values."""
    config = TunableConfig.model_validate({"n_samples": 5, "user_option_values": {"n_samples": 7}})

    assert config.n_samples == 5


def test_base_config_flat_payload_unchanged() -> None:
    """Payloads without user_option_values validate exactly as before."""
    config = TunableConfig.model_validate({"prediction_periods": 6, "n_samples": 9})

    assert config.prediction_periods == 6
    assert config.n_samples == 9


def test_base_config_hoists_undeclared_user_options_as_extras() -> None:
    """Undeclared user options still surface as top-level extras, not one level deep."""

    class FieldlessConfig(BaseConfig):
        """Config that declares no tunables and relies on extra fields."""

        prediction_periods: int = 3

    config = FieldlessConfig.model_validate({"user_option_values": {"some_knob": 7}})
    dumped = config.model_dump()

    assert dumped["some_knob"] == 7
    assert "user_option_values" not in dumped


def test_base_config_non_dict_user_option_values_left_alone() -> None:
    """A non-dict user_option_values value is not treated as a nested payload."""
    config = TunableConfig.model_validate({"user_option_values": "not-a-dict"})

    assert config.n_samples == 100
    assert config.model_dump()["user_option_values"] == "not-a-dict"
