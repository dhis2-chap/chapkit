"""Geospatial helpers for chapkit services."""

from collections.abc import Iterator
from typing import Any

# A GeoJSON bounding box in spec order: [west, south, east, north].
BoundingBox = tuple[float, float, float, float]


def _iter_positions(coordinates: Any) -> Iterator[tuple[float, float]]:
    """Yield every (lon, lat) position from an arbitrarily nested coordinate array."""
    if (
        isinstance(coordinates, (list, tuple))
        and len(coordinates) >= 2
        and isinstance(coordinates[0], (int, float))
        and isinstance(coordinates[1], (int, float))
    ):
        yield (float(coordinates[0]), float(coordinates[1]))
        return
    if isinstance(coordinates, (list, tuple)):
        for item in coordinates:
            yield from _iter_positions(item)


def _geometry_positions(geometry: Any) -> Iterator[tuple[float, float]]:
    """Yield every position from a GeoJSON geometry, recursing into collections."""
    if not isinstance(geometry, dict):
        return
    if geometry.get("type") == "GeometryCollection":
        for sub in geometry.get("geometries") or []:
            yield from _geometry_positions(sub)
        return
    yield from _iter_positions(geometry.get("coordinates"))


def _as_dict(feature_collection: Any) -> dict[str, Any]:
    """Coerce a geojson-pydantic model or mapping into a plain dict."""
    if hasattr(feature_collection, "model_dump"):
        dumped: dict[str, Any] = feature_collection.model_dump()
        return dumped
    if isinstance(feature_collection, dict):
        return feature_collection
    raise TypeError(f"Expected a FeatureCollection or mapping, got {type(feature_collection)}")


def bounding_box(feature_collection: Any) -> BoundingBox | None:
    """Compute the GeoJSON bbox [west, south, east, north] of a FeatureCollection.

    Accepts a geojson-pydantic FeatureCollection or a plain GeoJSON mapping. Returns
    None when there are no coordinates to bound (no features, or all geometries empty
    or missing).
    """
    data = _as_dict(feature_collection)
    longitudes: list[float] = []
    latitudes: list[float] = []
    for feature in data.get("features") or []:
        if not isinstance(feature, dict):
            continue
        for lon, lat in _geometry_positions(feature.get("geometry")):
            longitudes.append(lon)
            latitudes.append(lat)

    if not longitudes:
        return None
    return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))
