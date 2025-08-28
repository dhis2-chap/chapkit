import json
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from chapkit.types import ChapConfig


@runtime_checkable
class ChapStorage[T: ChapConfig](Protocol):
    def add_config(self, cfg: T) -> None: ...
    def update_config(self, cfg: T) -> bool: ...
    def del_config(self, id: UUID) -> bool: ...
    def get_config(self, id: UUID) -> T | None: ...
    def get_configs(self) -> list[T]: ...


class JsonChapStorage[T: ChapConfig](ChapStorage[T]):
    def __init__(self, path: str | Path, model_type: type[T]) -> None:
        self._path = Path(path)
        self._model_type = model_type
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            self._write_all({"configs": {}})

    # ---- CRUD ----

    def add_config(self, cfg: T) -> None:
        """Insert or replace config."""
        data = self._read_all()
        configs: dict[str, dict] = data.get("configs", {})
        configs[str(cfg.id)] = cfg.model_dump(mode="json")
        data["configs"] = configs
        self._write_all(data)

    def update_config(self, cfg: T) -> bool:
        """Update only if config exists. Returns True if updated, False otherwise."""
        data = self._read_all()
        configs: dict[str, dict] = data.get("configs", {})
        id_str = str(cfg.id)

        if id_str not in configs:
            return False

        configs[id_str] = cfg.model_dump(mode="json")
        data["configs"] = configs
        self._write_all(data)

        return True

    def del_config(self, id: UUID) -> bool:
        """Delete config by ID. Returns True if deleted, False otherwise."""
        data = self._read_all()
        configs: dict[str, dict] = data.get("configs", {})
        id_str = str(id)

        if id_str in configs:
            del configs[id_str]
            data["configs"] = configs
            self._write_all(data)
            return True

        return False

    def get_config(self, id: UUID) -> T | None:
        data = self._read_all()
        configs: dict[str, dict] = data.get("configs", {})
        raw = configs.get(str(id))

        return self._model_type.model_validate(raw) if raw else None

    def get_configs(self) -> list[T]:
        data = self._read_all()
        configs: dict[str, dict] = data.get("configs", {})

        return [self._model_type.model_validate(raw) for raw in configs.values()]

    # ---- IO helpers ----

    def _read_all(self) -> dict:
        if not self._path.exists():
            return {"configs": {}}

        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write_all(self, data: dict) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._path)
