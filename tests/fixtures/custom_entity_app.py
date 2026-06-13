"""Service factory with a custom Entity, Repository, Manager, and CrudRouter for integration tests."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from pydantic import EmailStr
from servicekit import BaseManager, BaseRepository, Database, Entity, EntityIn, EntityOut
from servicekit.api import CrudRouter
from servicekit.api.dependencies import get_session
from sqlalchemy import JSON, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from ulid import ULID

from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo


class User(Entity):
    """Custom user model extending chapkit's Entity base class."""

    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}  # Allow repeated app builds within one process

    username: Mapped[str] = mapped_column(unique=True)  # unique creates an index automatically
    email: Mapped[str] = mapped_column(unique=True)
    full_name: Mapped[str | None] = mapped_column(nullable=True)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class ApiConfig(BaseConfig):
    """Service configuration using chapkit's BaseConfig."""

    prediction_periods: int = 3
    max_users: int
    registration_enabled: bool
    default_theme: str


class UserIn(EntityIn):
    """User creation model extending chapkit's EntityIn."""

    username: str
    email: EmailStr
    full_name: str | None = None
    preferences: dict[str, Any] = {}


class UserOut(EntityOut):
    """User response model extending chapkit's EntityOut."""

    username: str
    email: EmailStr
    full_name: str | None
    preferences: dict[str, Any]


class UserRepository(BaseRepository[User, ULID]):
    """Repository for User model operations extending chapkit's BaseRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize user repository with database session."""
        super().__init__(session, User)

    async def find_by_username(self, username: str) -> User | None:
        """Find user by username."""
        result = await self.s.scalars(select(User).where(User.username == username))
        return result.one_or_none()


class UserManager(BaseManager[User, UserIn, UserOut, ULID]):
    """Manager for User entities extending chapkit's BaseManager."""

    def __init__(self, repository: UserRepository) -> None:
        """Initialize user manager with repository."""
        super().__init__(repository, User, UserOut)
        self.repository: UserRepository = repository

    async def find_by_username(self, username: str) -> UserOut | None:
        """Find a user by username."""
        user = await self.repository.find_by_username(username)
        if user:
            return self._to_output_schema(user)
        return None


def get_user_manager(session: AsyncSession = Depends(get_session)) -> UserManager:
    """Dependency for user manager."""
    return UserManager(UserRepository(session))


async def seed_data(app: FastAPI) -> None:
    """Seed initial configuration and users."""
    from chapkit import ConfigIn, ConfigManager, ConfigRepository

    database: Database | None = getattr(app.state, "database", None)
    if database is None:
        return

    async with database.session() as session:
        config_repository = ConfigRepository(session)
        config_manager = ConfigManager[ApiConfig](config_repository, ApiConfig)

        existing = await config_manager.find_by_name("production")
        if not existing:
            await config_manager.save(
                ConfigIn[ApiConfig](
                    name="production",
                    data=ApiConfig(max_users=1000, registration_enabled=True, default_theme="dark"),
                )
            )

        user_manager = UserManager(UserRepository(session))
        existing_user = await user_manager.find_by_username("admin")
        if not existing_user:
            await user_manager.save(
                UserIn(
                    username="admin",
                    email="admin@example.com",
                    full_name="Administrator",
                    preferences={"theme": "dark", "notifications": True},
                )
            )


def build_custom_entity_app() -> FastAPI:
    """Build a service combining chapkit config with a custom User entity and CrudRouter."""
    info = ServiceInfo(
        id="custom-entity-fixture-service",
        display_name="Library Usage Example",
        description="Demonstrates chapkit with custom models",
        version="1.0.0",
    )
    user_router = CrudRouter.create(
        prefix="/api/v1/users",
        tags=["users"],
        entity_in_type=UserIn,
        entity_out_type=UserOut,
        manager_factory=get_user_manager,
    )
    return (
        ServiceBuilder(info=info)
        .with_database()  # Defaults to in-memory SQLite
        .with_landing_page()
        .with_logging()
        .with_health()
        .with_config(ApiConfig)
        .include_router(user_router)
        .on_startup(seed_data)
        .build()
    )
