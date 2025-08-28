import json
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from chapkit.types import ChapConfig


@runtime_checkable
class ChapStorage[T: ChapConfig](Protocol):
    def add_config(self, cfg: T) -> None: ...
    def get_config(self, id: UUID) -> T | None: ...


class JsonChapStorage[T: ChapConfig](ChapStorage[T]):
    def __init__(self, path: str | Path, model_type: type[T]) -> None:
        self._path = Path(path)
        self._model_type = model_type
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            self._write_all({"configs": []})

    def get_configs(self) -> list[T]:
        data = self._read_all()
        configs: list[dict] = data.get("configs", [])
        return [self._model_type.model_validate(cfg) for cfg in configs]

    def get_config(self, cfg_id: UUID) -> T | None:
        data = self._read_all()
        configs: list[dict] = data.get("configs", [])
        for cfg in configs:
            if cfg.get("id") == str(cfg_id):
                return self._model_type.model_validate(cfg)
        return None

    def add_config(self, cfg: T) -> None:
        data = self._read_all()
        configs: list[dict] = data.get("configs", [])

        # replace if config with same id exists
        cfg_dict = cfg.model_dump(mode="json")
        configs = [c for c in configs if c.get("id") != str(cfg.id)]
        configs.append(cfg_dict)

        data["configs"] = configs
        self._write_all(data)

    # --- helpers ---
    def _read_all(self) -> dict:
        if not self._path.exists():
            return {"configs": []}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write_all(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
