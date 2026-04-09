"""Tests for TestDataGenerator in the CLI test module."""

from __future__ import annotations

from chapkit.cli.test.generator import TestDataGenerator
from chapkit.config.schemas import BaseConfig


class TestGeneratePredictionData:
    """Tests for generate_prediction_data method."""

    def test_historic_data_is_not_empty(self) -> None:
        """Verify historic data contains rows."""
        generator = TestDataGenerator(seed=42)
        historic, _ = generator.generate_prediction_data()
        assert len(historic["data"]) > 0

    def test_historic_data_has_disease_cases(self) -> None:
        """Verify historic rows have non-null disease_cases values."""
        generator = TestDataGenerator(seed=42)
        historic, _ = generator.generate_prediction_data()
        disease_cases_idx = historic["columns"].index("disease_cases")
        for row in historic["data"]:
            assert row[disease_cases_idx] is not None

    def test_future_data_has_null_disease_cases(self) -> None:
        """Verify future rows have null disease_cases for prediction."""
        generator = TestDataGenerator(seed=42)
        _, future = generator.generate_prediction_data()
        disease_cases_idx = future["columns"].index("disease_cases")
        for row in future["data"]:
            assert row[disease_cases_idx] is None

    def test_future_data_has_other_columns_populated(self) -> None:
        """Verify non-disease_cases columns in future data are populated."""
        generator = TestDataGenerator(seed=42)
        _, future = generator.generate_prediction_data()
        disease_cases_idx = future["columns"].index("disease_cases")
        for row in future["data"]:
            for i, value in enumerate(row):
                if i != disease_cases_idx:
                    assert value is not None

    def test_prediction_data_columns_match(self) -> None:
        """Verify historic and future have identical column lists."""
        generator = TestDataGenerator(seed=42)
        historic, future = generator.generate_prediction_data()
        assert historic["columns"] == future["columns"]


class TestGenerateConfigDataFromSchema:
    """Tests for generate_config_data_from_schema method."""

    def test_integer_field_without_default_is_at_least_one(self) -> None:
        """Verify integer fields without defaults produce values >= 1."""
        generator = TestDataGenerator(seed=42)
        schema = {"properties": {"count": {"type": "integer"}}}
        data = generator.generate_config_data_from_schema(schema, variation=0)
        assert data["count"] >= 1

    def test_integer_field_with_default_uses_default(self) -> None:
        """Verify integer fields with defaults use the default value."""
        generator = TestDataGenerator(seed=42)
        schema = {"properties": {"count": {"type": "integer", "default": 3}}}
        data = generator.generate_config_data_from_schema(schema, variation=0)
        assert data["count"] == 3

    def test_prediction_periods_from_base_config_schema(self) -> None:
        """Verify prediction_periods is at least 1 when using BaseConfig schema."""
        generator = TestDataGenerator(seed=42)
        schema = BaseConfig.model_json_schema()
        data = generator.generate_config_data_from_schema(schema, variation=0)
        assert data["prediction_periods"] >= 1
