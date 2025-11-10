"""Service builder with module integration (config, artifact, ml)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Coroutine, List, Self

from fastapi import Depends, FastAPI
from pydantic import EmailStr, HttpUrl
from servicekit import SqliteDatabaseBuilder
from servicekit.api.crud import CrudPermissions
from servicekit.api.dependencies import get_database, get_scheduler, get_session, set_scheduler
from servicekit.api.service_builder import BaseServiceBuilder, LifespanFactory, ServiceInfo
from sqlalchemy.ext.asyncio import AsyncSession

from chapkit import get_alembic_dir
from chapkit.artifact import (
    ArtifactHierarchy,
    ArtifactIn,
    ArtifactManager,
    ArtifactOut,
    ArtifactRepository,
    ArtifactRouter,
)
from chapkit.config import BaseConfig, ConfigIn, ConfigManager, ConfigOut, ConfigRepository, ConfigRouter
from chapkit.ml import MLManager, MLRouter, ModelRunnerProtocol
from chapkit.scheduler import ChapkitScheduler, InMemoryChapkitScheduler

from .dependencies import get_artifact_manager as default_get_artifact_manager
from .dependencies import get_config_manager as default_get_config_manager
from .dependencies import get_ml_manager as default_get_ml_manager

# Type alias for dependency factory functions
type DependencyFactory = Callable[..., Coroutine[Any, Any, Any]]


class AssessedStatus(StrEnum):
    """Status indicating the maturity and validation level of an ML service."""

    gray = "gray"  # Not intended for use, deprecated, or meant for legacy use only
    red = "red"  # Highly experimental prototype - not validated, only for early experimentation
    orange = "orange"  # Shows promise on limited data, needs manual configuration and careful evaluation
    yellow = "yellow"  # Ready for more rigorous testing
    green = "green"  # Validated and ready for production use


class MLServiceInfo(ServiceInfo):
    """Extended service metadata for ML services with author, organization, and assessment info."""

    author: str | None = None
    author_note: str | None = None
    author_assessed_status: AssessedStatus | None = None
    contact_email: EmailStr | None = None
    organization: str | None = None
    organization_logo_url: str | HttpUrl | None = None
    citation_info: str | None = None
    allow_free_additional_continuous_covariates: bool = False
    required_covariates: List[str] = field(default_factory=list)


@dataclass(slots=True)
class _ConfigOptions:
    """Internal config options for ServiceBuilder."""

    schema: type[BaseConfig]
    prefix: str = "/api/v1/configs"
    tags: List[str] = field(default_factory=lambda: ["Config"])
    permissions: CrudPermissions = field(default_factory=CrudPermissions)


@dataclass(slots=True)
class _ArtifactOptions:
    """Internal artifact options for ServiceBuilder."""

    hierarchy: ArtifactHierarchy
    prefix: str = "/api/v1/artifacts"
    tags: List[str] = field(default_factory=lambda: ["Artifacts"])
    enable_config_linking: bool = False
    permissions: CrudPermissions = field(default_factory=CrudPermissions)


@dataclass(slots=True)
class _MLOptions:
    """Internal ML options for ServiceBuilder."""

    runner: ModelRunnerProtocol
    prefix: str = "/api/v1/ml"
    tags: List[str] = field(default_factory=lambda: ["ML"])


class ServiceBuilder(BaseServiceBuilder):
    """Service builder with integrated module support (config, artifact, ml)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize service builder with module-specific state."""
        super().__init__(**kwargs)
        self._config_options: _ConfigOptions | None = None
        self._artifact_options: _ArtifactOptions | None = None
        self._ml_options: _MLOptions | None = None

    # --------------------------------------------------------------------- Module-specific fluent methods

    def with_config(
        self,
        schema: type[BaseConfig],
        *,
        prefix: str = "/api/v1/configs",
        tags: List[str] | None = None,
        permissions: CrudPermissions | None = None,
        allow_create: bool | None = None,
        allow_read: bool | None = None,
        allow_update: bool | None = None,
        allow_delete: bool | None = None,
    ) -> Self:
        base = permissions or CrudPermissions()
        perms = CrudPermissions(
            create=allow_create if allow_create is not None else base.create,
            read=allow_read if allow_read is not None else base.read,
            update=allow_update if allow_update is not None else base.update,
            delete=allow_delete if allow_delete is not None else base.delete,
        )
        self._config_options = _ConfigOptions(
            schema=schema,
            prefix=prefix,
            tags=list(tags) if tags else ["Config"],
            permissions=perms,
        )
        return self

    def with_artifacts(
        self,
        *,
        hierarchy: ArtifactHierarchy,
        prefix: str = "/api/v1/artifacts",
        tags: List[str] | None = None,
        enable_config_linking: bool = False,
        permissions: CrudPermissions | None = None,
        allow_create: bool | None = None,
        allow_read: bool | None = None,
        allow_update: bool | None = None,
        allow_delete: bool | None = None,
    ) -> Self:
        base = permissions or CrudPermissions()
        perms = CrudPermissions(
            create=allow_create if allow_create is not None else base.create,
            read=allow_read if allow_read is not None else base.read,
            update=allow_update if allow_update is not None else base.update,
            delete=allow_delete if allow_delete is not None else base.delete,
        )
        self._artifact_options = _ArtifactOptions(
            hierarchy=hierarchy,
            prefix=prefix,
            tags=list(tags) if tags else ["Artifacts"],
            enable_config_linking=enable_config_linking,
            permissions=perms,
        )
        return self

    def with_ml(
        self,
        runner: ModelRunnerProtocol,
        *,
        prefix: str = "/api/v1/ml",
        tags: List[str] | None = None,
    ) -> Self:
        """Enable ML train/predict endpoints with model runner."""
        self._ml_options = _MLOptions(
            runner=runner,
            prefix=prefix,
            tags=list(tags) if tags else ["ML"],
        )
        return self

    # --------------------------------------------------------------------- Extension point implementations

    def _validate_module_configuration(self) -> None:
        """Validate module-specific configuration."""
        if self._artifact_options and self._artifact_options.enable_config_linking and not self._config_options:
            raise ValueError(
                "Artifact config-linking requires a config schema. "
                "Call `with_config(...)` before enabling config linking in artifacts."
            )

        if self._ml_options:
            if not self._config_options:
                raise ValueError(
                    "ML operations require config for model configuration. "
                    "Call `with_config(...)` before `with_ml(...)`."
                )
            if not self._artifact_options:
                raise ValueError(
                    "ML operations require artifacts for model storage. "
                    "Call `with_artifacts(...)` before `with_ml(...)`."
                )
            if not self._job_options:
                raise ValueError(
                    "ML operations require job scheduler for async execution. "
                    "Call `with_jobs(...)` before `with_ml(...)`."
                )

    def _register_module_routers(self, app: FastAPI) -> None:
        """Register module-specific routers (config, artifact, task)."""
        if self._config_options:
            config_options = self._config_options
            config_schema = config_options.schema
            config_dep = self._build_config_dependency(config_schema)
            entity_in_type: type[ConfigIn[BaseConfig]] = ConfigIn[config_schema]  # type: ignore[valid-type]
            entity_out_type: type[ConfigOut[BaseConfig]] = ConfigOut[config_schema]  # type: ignore[valid-type]
            config_router = ConfigRouter.create(
                prefix=config_options.prefix,
                tags=config_options.tags,
                manager_factory=config_dep,
                entity_in_type=entity_in_type,
                entity_out_type=entity_out_type,
                permissions=config_options.permissions,
                enable_artifact_operations=(
                    self._artifact_options is not None and self._artifact_options.enable_config_linking
                ),
            )
            app.include_router(config_router)
            app.dependency_overrides[default_get_config_manager] = config_dep

        if self._artifact_options:
            artifact_options = self._artifact_options
            artifact_dep = self._build_artifact_dependency(
                hierarchy=artifact_options.hierarchy,
                include_config=artifact_options.enable_config_linking,
            )
            artifact_router = ArtifactRouter.create(
                prefix=artifact_options.prefix,
                tags=artifact_options.tags,
                manager_factory=artifact_dep,
                entity_in_type=ArtifactIn,
                entity_out_type=ArtifactOut,
                permissions=artifact_options.permissions,
                enable_config_access=self._config_options is not None and artifact_options.enable_config_linking,
            )
            app.include_router(artifact_router)
            app.dependency_overrides[default_get_artifact_manager] = artifact_dep

        if self._ml_options:
            ml_options = self._ml_options
            ml_dep = self._build_ml_dependency()
            ml_router = MLRouter.create(
                prefix=ml_options.prefix,
                tags=ml_options.tags,
                manager_factory=ml_dep,
            )
            app.include_router(ml_router)
            app.dependency_overrides[default_get_ml_manager] = ml_dep

    # --------------------------------------------------------------------- Module dependency builders

    @staticmethod
    def _build_config_dependency(
        schema: type[BaseConfig],
    ) -> DependencyFactory:
        async def _dependency(session: AsyncSession = Depends(get_session)) -> ConfigManager[BaseConfig]:
            repo = ConfigRepository(session)
            return ConfigManager[BaseConfig](repo, schema)

        return _dependency

    @staticmethod
    def _build_artifact_dependency(
        *,
        hierarchy: ArtifactHierarchy,
        include_config: bool,
    ) -> DependencyFactory:
        async def _dependency(session: AsyncSession = Depends(get_session)) -> ArtifactManager:
            artifact_repo = ArtifactRepository(session)
            return ArtifactManager(artifact_repo, hierarchy=hierarchy)

        return _dependency

    def _build_ml_dependency(self) -> DependencyFactory:
        ml_runner = self._ml_options.runner if self._ml_options else None
        config_schema = self._config_options.schema if self._config_options else None

        async def _dependency() -> MLManager:
            if ml_runner is None:
                raise RuntimeError("ML runner not configured")
            if config_schema is None:
                raise RuntimeError("Config schema not configured")

            runner: ModelRunnerProtocol = ml_runner
            scheduler_base = get_scheduler()
            # ChapkitScheduler extends AIOJobScheduler which extends JobScheduler
            if not isinstance(scheduler_base, ChapkitScheduler):
                raise RuntimeError("Scheduler must be ChapkitScheduler for ML operations")
            scheduler: ChapkitScheduler = scheduler_base
            database = get_database()
            return MLManager(runner, scheduler, database, config_schema)

        return _dependency

    def _build_lifespan(self) -> LifespanFactory:
        """Build lifespan context manager with InMemoryChapkitScheduler."""
        # Get parent lifespan factory
        parent_lifespan = super()._build_lifespan()

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            """Override scheduler creation to use InMemoryChapkitScheduler."""
            # Call parent lifespan which handles database and most setup
            async with parent_lifespan(app):
                # Use InMemoryChapkitScheduler if jobs are enabled
                if self._job_options is not None:
                    scheduler = InMemoryChapkitScheduler(max_concurrency=self._job_options.max_concurrency)
                    set_scheduler(scheduler)
                    app.state.scheduler = scheduler

                yield

        return lifespan


class MLServiceBuilder(ServiceBuilder):
    """Specialized service builder for ML services with all required components pre-configured."""

    def __init__(
        self,
        *,
        info: ServiceInfo | MLServiceInfo,
        config_schema: type[BaseConfig],
        hierarchy: ArtifactHierarchy,
        runner: ModelRunnerProtocol,
        database_url: str = "sqlite+aiosqlite:///:memory:",
        include_error_handlers: bool = True,
        include_logging: bool = True,
    ) -> None:
        """Initialize ML service builder with required ML components."""
        # Create database with migrations for file-based databases
        if ":memory:" in database_url:
            # In-memory database: no migrations needed
            db = SqliteDatabaseBuilder.in_memory().build()
        else:
            # File-based database: enable migrations with chapkit's bundled alembic
            # Extract file path from URL (e.g., "sqlite+aiosqlite:///chapkit.db" -> "chapkit.db")
            db_path = database_url.split("///")[-1] if "///" in database_url else database_url
            # Normalize path: strip leading "./" to avoid path interpretation issues
            if db_path.startswith("./"):
                db_path = db_path[2:]
            db = (
                SqliteDatabaseBuilder.from_file(db_path)
                .with_migrations(enabled=True, alembic_dir=get_alembic_dir())
                .build()
            )

        super().__init__(
            info=info,
            include_error_handlers=include_error_handlers,
            include_logging=include_logging,
        )

        # Configure database with migrations
        self.with_database(db)

        # Automatically configure required ML components
        self.with_health()
        self.with_system()
        self.with_config(config_schema)
        self.with_artifacts(hierarchy=hierarchy, enable_config_linking=True)
        self.with_jobs()
        self.with_landing_page()
        self.with_ml(runner=runner)
