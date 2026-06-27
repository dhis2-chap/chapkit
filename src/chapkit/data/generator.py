"""Synthetic data generator for ML workflows (shared by the CLI test runner and the console)."""

import math
import random
from typing import Any, Literal

from ulid import ULID


class TestDataGenerator:
    """Generate synthetic test data for ML workflows."""

    # Not a pytest test class despite the name; suppress collection.
    __test__ = False

    def __init__(self, seed: int | None = None) -> None:
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)

    def generate_training_data(
        self,
        num_locations: int = 5,
        num_periods: int = 12,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        additional_covariates: list[str] | None = None,
        extra_covariates: int = 0,
        start_year: int = 2020,
        start_period: int = 0,
        period_type: Literal["monthly", "weekly"] = "monthly",
    ) -> dict[str, Any]:
        """Generate training DataFrame with panel data structure for climate-health analysis."""
        # chap-core canonical always-present columns: time_period + location (indices),
        # disease_cases (target), population (non-optional per chap_core.predictor.feature_spec),
        # and the two common-optional climate covariates many models hardcode.
        # Emitted unconditionally so most models in the ecosystem Just Work under
        # `chapkit test` without needing to declare everything in required_covariates.
        columns: list[str] = [
            "time_period",
            "location",
            "disease_cases",
            "population",
            "rainfall",
            "mean_temperature",
        ]

        # Feature columns (climate/covariate data)
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates from service info - dedup against the always-present set.
        required_covariates = [c for c in (required_covariates or []) if c not in columns]
        for cov in required_covariates:
            columns.append(cov)

        # Additional continuous covariates from config - dedup against required + always-present.
        additional_covariates = [c for c in (additional_covariates or []) if c not in columns]
        for cov in additional_covariates:
            columns.append(cov)

        # Extra continuous covariates if allowed
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Stable per-location parameters (deterministic under the configured seed):
        # a constant population, a continuous seasonal phase so each location's
        # season peaks on an arbitrary month and can straddle the year boundary
        # (rather than every season starting in Jan and ending in Dec), plus a
        # baseline disease level and seasonal amplitude.
        periods_per_year = 52 if period_type == "weekly" else 12
        location_params = [
            {
                "population": random.randint(50_000, 1_500_000),
                "phase": random.random(),
                "base_cases": random.uniform(20.0, 70.0),
                "amplitude": random.uniform(0.45, 0.7),
            }
            for _ in range(num_locations)
        ]
        # Smooth seasonal covariates instead of static noise: every covariate
        # column after the canonical set gets its own phase offset.
        num_covariate_cols = len(columns) - 6

        # Generate panel data: periods x locations (periods grouped together).
        # Climate covariates and the disease target follow an annual cycle, and
        # disease_cases is correlated with the seasonal signal and rainfall so a
        # model has something real to learn.
        data: list[list[Any]] = []
        for period_idx in range(num_periods):
            # Absolute period index from the series origin; start_period lets a later
            # block (e.g. prediction's future) continue contiguously from an earlier
            # one so the calendar and seasonal cycle stay unbroken across the seam.
            absolute_idx = start_period + period_idx
            # Continuous position within the year, in [0, 1).
            year_fraction = (absolute_idx % periods_per_year) / periods_per_year
            for loc_idx in range(num_locations):
                params = location_params[loc_idx]
                row: list[Any] = []

                # Time period
                if period_type == "weekly":
                    year = start_year + (absolute_idx // 52)
                    week = (absolute_idx % 52) + 1
                    row.append(f"{year}-W{week:02d}")
                else:  # monthly (default)
                    year = start_year + (absolute_idx // 12)
                    month = (absolute_idx % 12) + 1
                    row.append(f"{year}-{month:02d}")

                # Location (matches geojson.properties.id)
                row.append(f"location_{loc_idx}")

                # Seasonal signal for this location/period in [-1, 1].
                season = math.sin(2.0 * math.pi * (year_fraction - params["phase"]))
                # rainfall (mm / period): wet season aligned to the location's phase.
                rainfall = max(0.0, 150.0 + 130.0 * season + random.uniform(-25.0, 25.0))
                # mean_temperature (degrees Celsius): a regional annual climate cycle.
                temperature = 24.0 + 7.0 * math.sin(2.0 * math.pi * (year_fraction - 0.08)) + random.uniform(-1.5, 1.5)
                # disease_cases (target): seasonal, rainfall-driven, per-location level.
                cases = (
                    params["base_cases"] * (1.0 + params["amplitude"] * season)
                    + 0.06 * rainfall
                    + random.gauss(0.0, 4.0)
                )

                # disease_cases (health outcome, whole number as float)
                row.append(float(max(0, round(cases))))
                # population (chap-core canonical, non-optional): stable per location
                row.append(params["population"])
                # rainfall
                row.append(round(rainfall, 2))
                # mean_temperature
                row.append(round(temperature, 2))

                # Remaining covariate columns (features, required, additional, extra),
                # each a smooth seasonal signal in [0, 100] with its own phase.
                for cov_idx in range(num_covariate_cols):
                    cov_phase = (cov_idx + 1) / (num_covariate_cols + 1)
                    value = (
                        50.0 + 35.0 * math.sin(2.0 * math.pi * (year_fraction - cov_phase)) + random.uniform(-8.0, 8.0)
                    )
                    row.append(round(min(100.0, max(0.0, value)), 2))

                data.append(row)

        return {"columns": columns, "data": data}

    def generate_prediction_data(
        self,
        num_locations: int = 5,
        num_periods: int = 12,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        additional_covariates: list[str] | None = None,
        extra_covariates: int = 0,
        period_type: Literal["monthly", "weekly"] = "monthly",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate historic and future DataFrames for prediction."""
        # Historic data: actual observations from earlier time periods
        historic = self.generate_training_data(
            num_locations=num_locations,
            num_periods=num_periods,
            num_features=num_features,
            required_covariates=required_covariates,
            additional_covariates=additional_covariates,
            extra_covariates=extra_covariates,
            start_year=2020,
            period_type=period_type,
        )

        # Future data: continue immediately after the historic span (same origin year,
        # offset by num_periods) so the future horizon is contiguous with history
        # instead of jumping to a fixed year and leaving a calendar gap. Then null out
        # disease_cases for prediction.
        future = self.generate_training_data(
            num_locations=num_locations,
            num_periods=num_periods,
            num_features=num_features,
            required_covariates=required_covariates,
            additional_covariates=additional_covariates,
            extra_covariates=extra_covariates,
            start_year=2020,
            start_period=num_periods,
            period_type=period_type,
        )

        disease_cases_idx = future["columns"].index("disease_cases")
        for row in future["data"]:
            row[disease_cases_idx] = None

        return historic, future

    def generate_config_data_from_schema(self, schema: dict[str, Any], variation: int = 0) -> dict[str, Any]:
        """Generate config data matching the JSON schema."""
        data: dict[str, Any] = {"id": str(ULID())}

        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name == "id":
                continue
            data[field_name] = self._generate_value_for_schema(field_schema, variation)

        return data

    def _generate_value_for_schema(self, field_schema: dict[str, Any], variation: int) -> Any:
        """Generate a value matching the field schema type."""
        if "default" in field_schema:
            default = field_schema["default"]
            # bool is a subclass of int, so it must be checked first to avoid
            # mutating a boolean default into an out-of-range integer (True + 1 -> 2).
            if isinstance(default, bool):
                return default
            if isinstance(default, int):
                return default + variation
            elif isinstance(default, float):
                return default + (variation * 0.1)
            return default

        field_type = field_schema.get("type", "string")

        if field_type == "integer":
            return max(1, variation)
        elif field_type == "number":
            return variation * 0.1
        elif field_type == "boolean":
            return variation % 2 == 0
        elif field_type == "string":
            if "enum" in field_schema:
                enum_values = field_schema["enum"]
                return enum_values[variation % len(enum_values)]
            return f"test_value_{variation}"
        elif field_type == "array":
            return []
        elif field_type == "object":
            return {}
        else:
            return None

    def generate_geo_data(
        self,
        num_features: int = 5,
        geo_type: Literal["polygon", "point"] = "polygon",
    ) -> dict[str, Any]:
        """Generate a GeoJSON FeatureCollection laid out as a contiguous grid of regions."""
        # Tile the locations as adjacent cells so they form a coherent, map-friendly
        # area (rather than tiny shapes scattered across the globe). Centered over a
        # plausible land region so it reads as real administrative areas on a basemap.
        columns = max(1, math.ceil(math.sqrt(num_features)))
        rows = max(1, math.ceil(num_features / columns))
        cell = 0.8  # degrees per cell
        base_lon, base_lat = 36.8, -1.3  # near Nairobi, East Africa
        origin_lon = base_lon - (columns * cell) / 2.0
        origin_lat = base_lat - (rows * cell) / 2.0

        features: list[dict[str, Any]] = []
        for i in range(num_features):
            col = i % columns
            row = i // columns
            min_lon = origin_lon + col * cell
            min_lat = origin_lat + row * cell

            if geo_type == "point":
                geometry: dict[str, Any] = {
                    "type": "Point",
                    "coordinates": [min_lon + cell / 2.0, min_lat + cell / 2.0],
                }
            else:  # polygon (default): a square cell that tiles with its neighbours
                geometry = {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [min_lon + cell, min_lat],
                            [min_lon + cell, min_lat + cell],
                            [min_lon, min_lat + cell],
                            [min_lon, min_lat],  # Close the ring
                        ]
                    ],
                }

            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {"id": f"location_{i}"},
                }
            )

        collection: dict[str, Any] = {"type": "FeatureCollection", "features": features}
        # Attach a GeoJSON bbox so consumers (e.g. a map view) can frame the data.
        from chapkit.data.geo import bounding_box

        bbox = bounding_box(collection)
        if bbox is not None:
            collection["bbox"] = list(bbox)
        return collection
