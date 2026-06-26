"""Tests for the GeoJSON bounding-box helper and generator bbox emission."""

from __future__ import annotations

from chapkit.data import TestDataGenerator, bounding_box


class TestBoundingBox:
    """bounding_box derives [west, south, east, north] from feature geometries."""

    def test_points(self) -> None:
        """A bbox spans all point coordinates."""
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [10.0, 20.0]}, "properties": {}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-5.0, 35.0]}, "properties": {}},
            ],
        }
        assert bounding_box(fc) == (-5.0, 20.0, 10.0, 35.0)

    def test_polygon(self) -> None:
        """A bbox spans every vertex of a polygon ring."""
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0], [0.0, 0.0]]],
                    },
                    "properties": {},
                }
            ],
        }
        assert bounding_box(fc) == (0.0, 0.0, 2.0, 3.0)

    def test_geometry_collection(self) -> None:
        """A bbox recurses into a GeometryCollection's geometries."""
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "GeometryCollection",
                        "geometries": [
                            {"type": "Point", "coordinates": [1.0, 1.0]},
                            {"type": "Point", "coordinates": [4.0, 9.0]},
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        assert bounding_box(fc) == (1.0, 1.0, 4.0, 9.0)

    def test_empty_returns_none(self) -> None:
        """A collection with no coordinates yields None."""
        assert bounding_box({"type": "FeatureCollection", "features": []}) is None
        assert (
            bounding_box(
                {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": None, "properties": {}}]}
            )
            is None
        )


class TestGeneratorBbox:
    """The synthetic geo generator attaches a valid GeoJSON bbox."""

    def test_polygon_geo_has_consistent_bbox(self) -> None:
        """Generated geo carries a bbox that matches its own coordinates."""
        generator = TestDataGenerator(seed=42)
        geo = generator.generate_geo_data(num_features=4, geo_type="polygon")
        assert "bbox" in geo
        west, south, east, north = geo["bbox"]
        assert west <= east and south <= north
        assert tuple(geo["bbox"]) == bounding_box(geo)

    def test_point_geo_has_bbox(self) -> None:
        """Point geometries also produce a bbox."""
        generator = TestDataGenerator(seed=7)
        geo = generator.generate_geo_data(num_features=3, geo_type="point")
        assert len(geo["bbox"]) == 4
