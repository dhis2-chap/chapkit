"""Example service with custom models and migrations."""

from models import Order, User
from servicekit.repository import BaseRepository

from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo

# Import models to ensure they're registered with Base.metadata
# This is important for Alembic autogenerate to detect them
__all__ = ["User", "Order"]


class AppConfig(BaseConfig):
    """Application configuration."""

    prediction_periods: int = 3


class UserRepository(BaseRepository[User]):
    """Repository for User operations."""


class OrderRepository(BaseRepository[Order]):
    """Repository for Order operations."""


# Build service
app = (
    ServiceBuilder(info=ServiceInfo(display_name="Custom Migrations Example"))
    .with_health()
    .with_database("sqlite+aiosqlite:///./custom_migrations.db")
    .with_config(AppConfig)
    .build()
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
