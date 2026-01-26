"""FastAPI service with config management, seeding, and custom metadata."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import EmailStr
from servicekit import Database
from ulid import ULID

from chapkit import BaseConfig, ConfigIn, ConfigManager, ConfigRepository
from chapkit.api import ServiceBuilder, ServiceInfo


class EnvironmentConfig(BaseConfig):
    """Environment configuration for API service settings."""

    prediction_periods: int = 3
    debug: bool
    api_host: str
    api_port: int
    max_connections: int


SEED_CONFIGS: tuple[tuple[str, ULID, EnvironmentConfig], ...] = (
    (
        "production",
        ULID.from_str("01K72P5N5KCRM6MD3BRE4P07N8"),
        EnvironmentConfig(
            debug=False,
            api_host="0.0.0.0",
            api_port=8080,
            max_connections=2000,
        ),
    ),
    (
        "staging",
        ULID.from_str("01K72P5N5KCRM6MD3BRE4P07N9"),
        EnvironmentConfig(
            debug=True,
            api_host="127.0.0.1",
            api_port=8001,
            max_connections=500,
        ),
    ),
    (
        "local",
        ULID.from_str("01K72P5N5KCRM6MD3BRE4P07NA"),
        EnvironmentConfig(
            debug=True,
            api_host="127.0.0.1",
            api_port=8000,
            max_connections=100,
        ),
    ),
)


class ConfigServiceInfo(ServiceInfo):
    """Extended service info with author and config metadata."""

    author: str | None = None
    contact_email: EmailStr | None = None
    config_schema: dict[str, object]
    seeded_configs: list[str]


async def seed_configs(app: FastAPI) -> None:
    """Seed database with predefined environment configurations on startup."""
    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        repo = ConfigRepository(session)
        manager = ConfigManager[EnvironmentConfig](repo, EnvironmentConfig)
        await manager.delete_all()
        await manager.save_all(
            ConfigIn[EnvironmentConfig](id=config_id, name=name, data=payload)
            for name, config_id, payload in SEED_CONFIGS
        )


info = ConfigServiceInfo(
    id="chapkit-config-service",
    display_name="Chapkit Config Service",
    summary="Environment configuration CRUD example",
    author="Morten Hansen",
    contact_email="morten@dhis2.org",
    contact={"email": "morten@dhis2.org"},
    config_schema=EnvironmentConfig.model_json_schema(),
    seeded_configs=[name for name, _, _ in SEED_CONFIGS],
)


app: FastAPI = (
    ServiceBuilder(info=info)
    .with_landing_page()
    .with_logging()
    .with_health()
    .with_system()
    .with_config(EnvironmentConfig)
    .on_startup(seed_configs)
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
