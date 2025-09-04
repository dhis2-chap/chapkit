import os
import pickle
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from sqlalchemy import LargeBinary, create_engine, event, select
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from chapkit.types import ChapConfig, TChapConfig

T = TChapConfig


class ChapStorage[T: ChapConfig](ABC):
    @abstractmethod
    def add_config(self, cfg: T) -> None: ...
    @abstractmethod
    def update_config(self, cfg: T) -> bool: ...
    @abstractmethod
    def del_config(self, id: UUID) -> bool: ...
    @abstractmethod
    def get_config(self, id: UUID) -> T | None: ...
    @abstractmethod
    def get_configs(self) -> list[T]: ...
    @abstractmethod
    def add_model(self, id: UUID, obj: Any) -> None: ...
    @abstractmethod
    def del_model(self, id: UUID) -> bool: ...
    @abstractmethod
    def get_model(self, id: UUID) -> Any | None: ...


# ---------- ORM base & rows ----------
class Base(DeclarativeBase):
    pass


class ConfigRow(Base):
    __tablename__ = "configs"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)


class ModelRow(Base):
    __tablename__ = "models"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


# ---------- helper: engine factory ----------
def make_engine(db_path: Path) -> Engine:
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record) -> None:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-65536;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.execute("PRAGMA mmap_size=67108864;")
        cur.close()

    return engine


# ---------- Storage (per-instance engine & session) ----------
class SqlAlchemyChapStorage(ChapStorage[T]):
    def __init__(self, model_type: type[T], file: str | Path = "target/chapkit.db") -> None:
        self._model_type = model_type
        self._db_path = Path(file)
        os.makedirs(self._db_path.parent, exist_ok=True)

        self._engine: Engine = make_engine(self._db_path)
        Base.metadata.create_all(self._engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self._engine) as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise

    # --- Pydantic <-> JSON helpers ---
    def _cfg_to_dict(self, cfg: T) -> dict:
        d = cfg.model_dump(mode="python")

        if isinstance(d.get("id"), UUID):
            d["id"] = str(d["id"])

        return d

    def _cfg_from_dict(self, data: dict) -> T:
        return self._model_type.model_validate(data)

    # --- Pickle helpers (trusted data only) ---
    @staticmethod
    def _pickle(obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _unpickle(b: bytes) -> Any:
        return pickle.loads(b)

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

    def del_config(self, id: UUID) -> bool:
        with self._session() as s:
            row = s.get(ConfigRow, id)

            if row is None:
                return False
            s.delete(row)

            return True

    def get_config(self, id: UUID) -> T | None:
        with self._session() as s:
            row = s.get(ConfigRow, id)
            return self._cfg_from_dict(row.config) if row else None

    def get_configs(self) -> list[T]:
        with self._session() as s:
            rows = s.scalars(select(ConfigRow)).all()
            return [self._cfg_from_dict(r.config) for r in rows]

    # --- Model API (pickled) ---

    def add_model(self, id: UUID, obj: Any) -> None:
        blob = self._pickle(obj)

        with self._session() as s:
            stmt = sqlite_insert(ModelRow).values(id=id, data=blob)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ModelRow.id],
                set_={"data": stmt.excluded.data},
            )

            s.execute(stmt)

    def del_model(self, id: UUID) -> bool:
        with self._session() as s:
            row = s.get(ModelRow, id)

            if row is None:
                return False

            s.delete(row)

            return True

    def get_model(self, id: UUID) -> Any | None:
        with self._session() as s:
            row = s.get(ModelRow, id)
            return self._unpickle(row.data) if row else None
