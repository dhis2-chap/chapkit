"""Shared test stubs used across API router tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, TypeVar

from ulid import ULID

from chapkit import ArtifactIn, ArtifactOut, ArtifactTreeNode, BaseConfig, ConfigIn, ConfigOut
from servicekit import Manager

ConfigDataT = TypeVar("ConfigDataT", bound=BaseConfig)


class ConfigManagerStub(Manager[ConfigIn[ConfigDataT], ConfigOut[ConfigDataT], ULID], Generic[ConfigDataT]):
    """Config manager stub supporting name lookups for tests."""

    def __init__(
        self,
        *,
        items: dict[str, ConfigOut[ConfigDataT]] | None = None,
        linked_artifacts: dict[ULID, list[ArtifactOut]] | None = None,
    ) -> None:
        self._items_by_name = dict(items or {})
        self._items_by_id = {item.id: item for item in self._items_by_name.values()}
        self._linked_artifacts = linked_artifacts or {}
        self._link_error: str | None = None

    def set_link_error(self, error: str) -> None:
        """Set an error to be raised on link operations."""
        self._link_error = error

    async def link_artifact(self, config_id: ULID, artifact_id: ULID) -> None:
        """Link an artifact to a config."""
        if self._link_error:
            raise ValueError(self._link_error)

    async def unlink_artifact(self, artifact_id: ULID) -> None:
        """Unlink an artifact from its config."""
        if self._link_error:
            raise Exception(self._link_error)

    async def get_linked_artifacts(self, config_id: ULID) -> list[ArtifactOut]:
        """Get all artifacts linked to a config."""
        return self._linked_artifacts.get(config_id, [])

    async def get_config_for_artifact(self, artifact_id: ULID, artifact_repo: Any) -> ConfigOut[ConfigDataT] | None:
        """Get config for an artifact by traversing to root."""
        # For stub purposes, return the first config in items
        items = list(self._items_by_id.values())
        return items[0] if items else None

    async def find_by_name(self, name: str) -> ConfigOut[ConfigDataT] | None:
        return self._items_by_name.get(name)

    async def save(self, data: ConfigIn[ConfigDataT]) -> ConfigOut[ConfigDataT]:
        raise NotImplementedError

    async def save_all(self, items: Iterable[ConfigIn[ConfigDataT]]) -> list[ConfigOut[ConfigDataT]]:
        raise NotImplementedError

    async def find_all(self) -> list[ConfigOut[ConfigDataT]]:
        return list(self._items_by_id.values())

    async def find_paginated(self, page: int, size: int) -> tuple[list[ConfigOut[ConfigDataT]], int]:
        all_items = list(self._items_by_id.values())
        offset = (page - 1) * size
        paginated_items = all_items[offset : offset + size]
        return paginated_items, len(all_items)

    async def find_all_by_id(self, ids: Sequence[ULID]) -> list[ConfigOut[ConfigDataT]]:
        raise NotImplementedError

    async def find_by_id(self, id: ULID) -> ConfigOut[ConfigDataT] | None:
        return self._items_by_id.get(id)

    async def exists_by_id(self, id: ULID) -> bool:
        return id in self._items_by_id

    async def delete_by_id(self, id: ULID) -> None:
        self._items_by_id.pop(id, None)
        # Keep name map in sync
        names_to_remove = [name for name, item in self._items_by_name.items() if item.id == id]
        for name in names_to_remove:
            self._items_by_name.pop(name, None)

    async def delete_all(self) -> None:
        self._items_by_id.clear()
        self._items_by_name.clear()

    async def delete_all_by_id(self, ids: Sequence[ULID]) -> None:
        raise NotImplementedError

    async def count(self) -> int:
        return len(self._items_by_id)


class ArtifactManagerStub(Manager[ArtifactIn, ArtifactOut, ULID]):
    """Artifact manager stub providing tree data for tests."""

    def __init__(self, *, trees: dict[ULID, ArtifactTreeNode] | None = None) -> None:
        self._trees = trees or {}
        self.repo = None  # Will be set if needed for config lookups

    async def build_tree(self, id: ULID) -> ArtifactTreeNode | None:
        return self._trees.get(id)

    async def save(self, data: ArtifactIn) -> ArtifactOut:
        raise NotImplementedError

    async def save_all(self, items: Iterable[ArtifactIn]) -> list[ArtifactOut]:
        raise NotImplementedError

    async def find_all(self) -> list[ArtifactOut]:
        return []

    async def find_paginated(self, page: int, size: int) -> tuple[list[ArtifactOut], int]:
        return [], 0

    async def find_all_by_id(self, ids: Sequence[ULID]) -> list[ArtifactOut]:
        raise NotImplementedError

    async def find_by_id(self, id: ULID) -> ArtifactOut | None:
        raise NotImplementedError

    async def exists_by_id(self, id: ULID) -> bool:
        raise NotImplementedError

    async def delete_by_id(self, id: ULID) -> None:
        raise NotImplementedError

    async def delete_all(self) -> None:
        raise NotImplementedError

    async def delete_all_by_id(self, ids: Sequence[ULID]) -> None:
        raise NotImplementedError

    async def count(self) -> int:
        return len(self._trees)


def singleton_factory(instance: Any) -> Callable[[], Any]:
    """Return a dependency factory that always yields the provided instance."""

    def _provide() -> Any:
        return instance

    return _provide
