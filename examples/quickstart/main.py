"""Basic example: persist and retrieve typed configuration with ConfigManager."""

from __future__ import annotations

import asyncio

from servicekit import SqliteDatabaseBuilder
from ulid import ULID

from chapkit import BaseConfig, ConfigIn, ConfigManager, ConfigOut, ConfigRepository

EMAIL_SERVICE_CONFIG_ID = ULID.from_str("01K72P5N5KCRM6MD3BRE4P07N4")


class EmailServiceConfig(BaseConfig):
    """Email service configuration with sender details and retry settings."""

    prediction_periods: int = 3
    sender_name: str
    sender_address: str
    max_retries: int = 3
    enabled: bool = True


async def main() -> None:
    """Demonstrate storing and retrieving typed configuration."""
    db = SqliteDatabaseBuilder.in_memory().build()
    await db.init()

    try:
        async with db.session() as session:
            repo = ConfigRepository(session)
            manager = ConfigManager[EmailServiceConfig](repo, EmailServiceConfig)

            created: ConfigOut[EmailServiceConfig] = await manager.save(
                ConfigIn[EmailServiceConfig](
                    id=EMAIL_SERVICE_CONFIG_ID,
                    name="email_service",
                    data=EmailServiceConfig(
                        sender_name="Chapkit Notifications",
                        sender_address="no-reply@example.com",
                        max_retries=5,
                        enabled=True,
                    ),
                )
            )
            retrieved = await manager.find_by_name("email_service")
            if not retrieved or not retrieved.data:
                raise RuntimeError("Expected to retrieve the stored config")

            assert retrieved.data.sender_name == "Chapkit Notifications"
            assert retrieved.data.sender_address == "no-reply@example.com"
            assert retrieved.data.max_retries == 5
            assert retrieved.data.enabled is True

            print(f"✓ Stored config '{created.name}' (ULID: {created.id})")
            print("✓ Retrieved config matches stored values:")
            print(f"  sender_name   = {retrieved.data.sender_name}")
            print(f"  sender_address= {retrieved.data.sender_address}")
            print(f"  max_retries   = {retrieved.data.max_retries}")
            print(f"  enabled       = {retrieved.data.enabled}")

    finally:
        await db.dispose()


if __name__ == "__main__":
    asyncio.run(main())
