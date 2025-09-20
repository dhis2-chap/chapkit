from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import ForeignKey, create_engine, event, func, select
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from ulid import ULID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.types import PickleType, TypeDecorator, CHAR

from chapkit.types import ChapConfig, TChapConfig

T = TChapConfig


class ULIDType(TypeDecorator):
    """
    Custom SQLAlchemy type for ULID.
    It stores ULID as a CHAR(26) in the database.
    """

    impl = CHAR(26)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return ULID.from_str(value)
        return value


class ChapDatabase[T: ChapConfig](ABC):
    @abstractmethod
    def add_config(self, cfg: T) -> None: ...
    @abstractmethod
    def update_config(self, cfg: T) -> bool: ...
    @abstractmethod
    def del_config(self, id: ULID) -> bool: ...
    @abstractmethod
    def get_config(self, id: ULID) -> T | None: ...
    @abstractmethod
    def get_configs(self) -> list[T]: ...
    @abstractmethod
    def add_artifact(self, id: ULID, cfg: T, obj: Any, parent_id: ULID | None = None) -> None: ...
    @abstractmethod
    def del_artifact(self, id: ULID) -> bool: ...
    @abstractmethod
    def get_artifact(self, id: ULID) -> Any | None: ...
    @abstractmethod
    def get_artifact_row(self, id: ULID) -> ArtifactRow | None: ...
    @abstractmethod
    def get_config_for_artifact(self, artifact_id: ULID) -> T | None: ...
    @abstractmethod
    def get_artifacts_for_config(self, config_id: ULID) -> list[tuple[ULID, Any]]: ...
    @abstractmethod
    def get_artifact_rows_for_config(self, config_id: ULID) -> list[ArtifactRow]: ...
    @abstractmethod
    def get_artifact_roots_for_config(self, config_id: ULID) -> list[ArtifactRow]: ...


# ---------- ORM base & rows ----------
class Base(DeclarativeBase):
    id: Mapped[ULID] = mapped_column(ULIDType, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class ConfigRow(Base):
    __tablename__ = "configs"
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    artifacts: Mapped[list["ArtifactRow"]] = relationship(back_populates="config", cascade="all, delete-orphan")


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    config_id: Mapped[ULID] = mapped_column(ForeignKey("configs.id", ondelete="CASCADE"), nullable=False)
    parent_id: Mapped[ULID | None] = mapped_column(ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=True)
    data: Mapped[Any] = mapped_column(PickleType, nullable=False)
    config: Mapped["ConfigRow"] = relationship(back_populates="artifacts")
    children: Mapped[list["ArtifactRow"]] = relationship(
        "ArtifactRow",
        back_populates="parent",
        cascade="all, delete-orphan",
    )
    parent: Mapped[Optional["ArtifactRow"]] = relationship(
        "ArtifactRow",
        back_populates="children",
        primaryjoin="ArtifactRow.parent_id == ArtifactRow.id",
        remote_side="ArtifactRow.id",
    )


# ---------- helper: engine factory ----------
def make_engine(db_path: Path) -> Engine:
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False, "timeout": 30.0},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record) -> None:
        cur = dbapi_connection.cursor()

        # WAL (Write-Ahead Logging) improves concurrency, allowing reads and writes to occur simultaneously.
        cur.execute("PRAGMA journal_mode=WAL;")
        # 'NORMAL' synchronous mode is a good balance of safety and performance with WAL.
        cur.execute("PRAGMA synchronous=NORMAL;")
        # Enforce foreign key constraints, which is off by default in SQLite.
        cur.execute("PRAGMA foreign_keys=ON;")
        # Set a busy timeout to avoid 'database is locked' errors during contention.
        cur.execute("PRAGMA busy_timeout=30000;")  # 30 seconds
        # Store temporary tables in memory for speed.
        cur.execute("PRAGMA temp_store=MEMORY;")
        # Increase the page cache size to reduce disk I/O. Value is in KiB.
        cur.execute("PRAGMA cache_size=-64000;")  # 64MB
        # Memory-map a portion of the database file for faster access.
        cur.execute("PRAGMA mmap_size=134217728;")  # 128 MiB

        cur.close()

    return engine


# ---------- Database (per-instance engine & session) ----------
class SqlAlchemyChapDatabase(ChapDatabase[T]):
    def __init__(self, file: str | Path = "target/chapkit.db", *, config_type: type[T]) -> None:
        self._config_type = config_type
        self._db_path = Path(file)
        os.makedirs(self._db_path.parent, exist_ok=True)

        self._engine: Engine = make_engine(self._db_path)
        Base.metadata.create_all(self._engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self._engine, expire_on_commit=False) as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise

    # --- Pydantic <-> JSON helpers ---
    def _cfg_to_dict(self, cfg: T) -> dict:
        d = cfg.model_dump(mode="python")

        if isinstance(d.get("id"), ULID):
            d["id"] = str(d["id"])

        return d

    def _cfg_from_dict(self, data: dict) -> T:
        return self._config_type.model_validate(data)

    # --- Config API (SQLite upsert) ---
    def add_config(self, cfg: T) -> None:
        payload = self._cfg_to_dict(cfg)
        with self._session() as s:
            stmt = sqlite_insert(ConfigRow).values(id=cfg.id, config=payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ConfigRow.id],
                set_={"config": stmt.excluded.config},
            )
            s.execute(stmt)

    def update_config(self, cfg: T) -> bool:
        with self._session() as s:
            row = s.get(ConfigRow, cfg.id)
            if row is None:
                return False
            row.config = self._cfg_to_dict(cfg)
            return True

    def del_config(self, id: ULID) -> bool:
        with self._session() as s:
            row = s.get(ConfigRow, id)

            if row is None:
                return False
            s.delete(row)

            return True

    def get_config(self, id: ULID) -> T | None:
        with self._session() as s:
            row = s.get(ConfigRow, id)
            return self._cfg_from_dict(row.config) if row else None

    def get_configs(self) -> list[T]:
        with self._session() as s:
            rows = s.scalars(select(ConfigRow)).all()
            return [self._cfg_from_dict(r.config) for r in rows]

    # --- Artifact API (pickled) ---

    def add_artifact(self, id: ULID, cfg: T, obj: Any, parent_id: ULID | None = None) -> None:
        with self._session() as s:
            stmt = sqlite_insert(ArtifactRow).values(id=id, config_id=cfg.id, data=obj, parent_id=parent_id)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ArtifactRow.id],
                set_={
                    "data": stmt.excluded.data,
                    "config_id": stmt.excluded.config_id,
                    "parent_id": stmt.excluded.parent_id,
                },
            )
            s.execute(stmt)

    def del_artifact(self, id: ULID) -> bool:
        with self._session() as s:
            row = s.get(ArtifactRow, id)

            if row is None:
                return False

            s.delete(row)

            return True

    def get_artifact(self, id: ULID) -> Any | None:
        with self._session() as s:
            row = s.get(ArtifactRow, id)
            return row.data if row else None

    def get_artifact_row(self, id: ULID) -> ArtifactRow | None:
        with self._session() as s:
            return s.get(ArtifactRow, id)

    def get_config_for_artifact(self, artifact_id: ULID) -> T | None:
        with self._session() as s:
            row = s.get(ArtifactRow, artifact_id)
            if not row:
                return None
            return self._cfg_from_dict(row.config.config)

    def get_artifacts_for_config(self, config_id: ULID) -> list[tuple[ULID, Any]]:
        with self._session() as s:
            rows = s.scalars(select(ArtifactRow).where(ArtifactRow.config_id == config_id)).all()
            return [(row.id, row.data) for row in rows]

    def get_artifact_rows_for_config(self, config_id: ULID) -> list[ArtifactRow]:
        with self._session() as s:
            return s.scalars(select(ArtifactRow).where(ArtifactRow.config_id == config_id)).all()

    def get_artifact_roots_for_config(self, config_id: ULID) -> list[ArtifactRow]:
        with self._session() as s:
            return s.scalars(
                select(ArtifactRow).where(ArtifactRow.config_id == config_id).where(ArtifactRow.parent_id.is_(None))
            ).all()
