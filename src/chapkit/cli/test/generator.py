"""Test data generator for ML workflows."""

import random
from typing import Any

from ulid import ULID


class TestDataGenerator:
    """Generate synthetic test data for ML workflows."""

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
        extra_covariates: int = 0,
    ) -> dict[str, Any]:
        """Generate training DataFrame with panel data structure for climate-health analysis."""
        required_covariates = required_covariates or []

        # Build columns list
        columns: list[str] = ["location", "time_period", "disease_cases"]

        # Feature columns (climate/covariate data)
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates from service info
        for cov in required_covariates:
            if cov not in columns:
                columns.append(cov)

        # Extra continuous covariates if allowed
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Generate panel data: locations x periods
        data: list[list[Any]] = []
        for loc_idx in range(num_locations):
            for period_idx in range(num_periods):
                row: list[Any] = []

                # Location (matches geojson.properties.id)
                row.append(f"location_{loc_idx}")

                # Time period (YYYY-mm format)
                year = 2020 + (period_idx // 12)
                month = (period_idx % 12) + 1
                row.append(f"{year}-{month:02d}")

                # Disease cases (health outcome, always positive)
                row.append(random.uniform(1, 100))

                # Feature values (climate data)
                for _ in range(num_features):
                    row.append(random.uniform(0, 100))

                # Required covariate values
                for _ in required_covariates:
                    row.append(random.uniform(0, 100))

                # Extra covariate values
                for _ in range(extra_covariates):
                    row.append(random.uniform(0, 100))

                data.append(row)

        return {"columns": columns, "data": data}

    def generate_prediction_data(
        self,
        num_locations: int = 5,
        num_periods: int = 12,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        extra_covariates: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate historic and future DataFrames for prediction."""
        required_covariates = required_covariates or []

        # Build columns list (same structure as training)
        columns: list[str] = ["location", "time_period", "disease_cases"]

        # Feature columns
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates
        for cov in required_covariates:
            if cov not in columns:
                columns.append(cov)

        # Extra covariates
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Empty historic (matching scaffolded template pattern)
        historic: dict[str, Any] = {"columns": columns, "data": []}

        # Future data to predict (same panel structure)
        future = self.generate_training_data(
            num_locations=num_locations,
            num_periods=num_periods,
            num_features=num_features,
            required_covariates=required_covariates,
            extra_covariates=extra_covariates,
        )

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
            if isinstance(default, int):
                return default + variation
            elif isinstance(default, float):
                return default + (variation * 0.1)
            return default

        field_type = field_schema.get("type", "string")

        if field_type == "integer":
            return variation
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

    def generate_geo_data(self, num_features: int = 5) -> dict[str, Any]:
        """Generate simple GeoJSON FeatureCollection with Polygon geometries."""
        features: list[dict[str, Any]] = []

        for i in range(num_features):
            # Random center point (longitude, latitude)
            center_lon = random.uniform(-170, 170)
            center_lat = random.uniform(-80, 80)

            # Small polygon around center (roughly 1 degree square)
            size = 0.5
            coordinates = [
                [
                    [center_lon - size, center_lat - size],
                    [center_lon + size, center_lat - size],
                    [center_lon + size, center_lat + size],
                    [center_lon - size, center_lat + size],
                    [center_lon - size, center_lat - size],  # Close the ring
                ]
            ]

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": coordinates},
                    "properties": {"id": f"location_{i}"},
                }
            )

        return {"type": "FeatureCollection", "features": features}
