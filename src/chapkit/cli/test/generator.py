"""Test data generator for ML workflows."""

import random
from typing import Any, Literal

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
        additional_covariates: list[str] | None = None,
        extra_covariates: int = 0,
        start_year: int = 2020,
        period_type: Literal["monthly", "weekly"] = "monthly",
    ) -> dict[str, Any]:
        """Generate training DataFrame with panel data structure for climate-health analysis."""
        required_covariates = required_covariates or []
        # Additional continuous covariates the config declares (BaseConfig field).
        # Kept distinct from required_covariates so we can tell why each column is there.
        additional_covariates = [c for c in (additional_covariates or []) if c not in required_covariates]

        # Build columns list
        columns: list[str] = ["time_period", "location", "disease_cases"]

        # Feature columns (climate/covariate data)
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates from service info
        for cov in required_covariates:
            if cov not in columns:
                columns.append(cov)

        # Additional continuous covariates from config
        for cov in additional_covariates:
            if cov not in columns:
                columns.append(cov)

        # Extra continuous covariates if allowed
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Generate panel data: periods x locations (periods grouped together)
        data: list[list[Any]] = []
        for period_idx in range(num_periods):
            for loc_idx in range(num_locations):
                row: list[Any] = []

                # Time period
                if period_type == "weekly":
                    year = start_year + (period_idx // 52)
                    week = (period_idx % 52) + 1
                    row.append(f"{year}-W{week:02d}")
                else:  # monthly (default)
                    year = start_year + (period_idx // 12)
                    month = (period_idx % 12) + 1
                    row.append(f"{year}-{month:02d}")

                # Location (matches geojson.properties.id)
                row.append(f"location_{loc_idx}")

                # Disease cases (health outcome, whole number as float)
                row.append(float(random.randint(1, 100)))

                # Feature values (climate data)
                for _ in range(num_features):
                    row.append(random.uniform(0, 100))

                # Required covariate values
                for _ in required_covariates:
                    row.append(random.uniform(0, 100))

                # Additional continuous covariate values
                for _ in additional_covariates:
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

        # Future data: generate structure then null out disease_cases for prediction
        future = self.generate_training_data(
            num_locations=num_locations,
            num_periods=num_periods,
            num_features=num_features,
            required_covariates=required_covariates,
            additional_covariates=additional_covariates,
            extra_covariates=extra_covariates,
            start_year=2025,
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
        """Generate simple GeoJSON FeatureCollection."""
        features: list[dict[str, Any]] = []

        for i in range(num_features):
            # Random center point (longitude, latitude)
            center_lon = random.uniform(-170, 170)
            center_lat = random.uniform(-80, 80)

            if geo_type == "point":
                geometry: dict[str, Any] = {"type": "Point", "coordinates": [center_lon, center_lat]}
            else:  # polygon (default)
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
                geometry = {"type": "Polygon", "coordinates": coordinates}

            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {"id": f"location_{i}"},
                }
            )

        return {"type": "FeatureCollection", "features": features}
