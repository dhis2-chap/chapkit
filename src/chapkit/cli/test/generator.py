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
        num_rows: int = 100,
        num_dimensions: int = 2,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        extra_covariates: int = 0,
    ) -> dict[str, Any]:
        """Generate training DataFrame with dimension and numeric columns."""
        required_covariates = required_covariates or []

        # Build columns list
        columns: list[str] = []

        # Dimension columns
        for i in range(num_dimensions):
            columns.append(f"dim_{i}")

        # Feature columns
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates from service info
        for cov in required_covariates:
            if cov not in columns:
                columns.append(cov)

        # Extra continuous covariates if allowed
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Generate data
        data: list[list[Any]] = []
        for _ in range(num_rows):
            row: list[Any] = []

            # Dimension values (categorical)
            for i in range(num_dimensions):
                row.append(f"cat_{i}_{random.randint(0, 4)}")

            # Feature values (floats)
            for _ in range(num_features):
                row.append(random.uniform(0, 100))

            # Required covariate values (floats)
            for _ in required_covariates:
                row.append(random.uniform(0, 100))

            # Extra covariate values (floats)
            for _ in range(extra_covariates):
                row.append(random.uniform(0, 100))

            data.append(row)

        return {"columns": columns, "data": data}

    def generate_prediction_data(
        self,
        num_rows: int = 10,
        num_dimensions: int = 2,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        extra_covariates: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate historic and future DataFrames for prediction."""
        required_covariates = required_covariates or []

        # Build columns list (same as training but without target)
        columns: list[str] = []

        # Dimension columns
        for i in range(num_dimensions):
            columns.append(f"dim_{i}")

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

        # Future data to predict
        future = self.generate_training_data(
            num_rows=num_rows,
            num_dimensions=num_dimensions,
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
        """Generate simple GeoJSON FeatureCollection with Point geometries."""
        features: list[dict[str, Any]] = []

        for i in range(num_features):
            # Random coordinates (longitude, latitude)
            lon = random.uniform(-180, 180)
            lat = random.uniform(-90, 90)

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"id": f"location_{i}"},
                }
            )

        return {"type": "FeatureCollection", "features": features}
