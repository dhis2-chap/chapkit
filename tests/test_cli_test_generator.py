"""Tests for TestDataGenerator in the CLI test module."""

from __future__ import annotations

from chapkit.config.schemas import BaseConfig
from chapkit.data.generator import TestDataGenerator


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

    def test_future_periods_continue_contiguously_after_historic(self) -> None:
        """Verify future periods follow historic with no overlap or calendar gap."""
        generator = TestDataGenerator(seed=42)
        historic, future = generator.generate_prediction_data(num_locations=1, num_periods=24)
        time_idx = historic["columns"].index("time_period")
        historic_periods = [row[time_idx] for row in historic["data"]]
        future_periods = [row[time_idx] for row in future["data"]]
        # Lexicographic order matches calendar order for the YYYY-MM labels used here.
        assert max(historic_periods) < min(future_periods)
        assert historic_periods[-1] == "2021-12"
        assert future_periods[0] == "2022-01"

    def test_prediction_data_spans_multiple_years(self) -> None:
        """Verify the generated series covers more than one calendar year."""
        generator = TestDataGenerator(seed=42)
        historic, future = generator.generate_prediction_data(num_locations=1, num_periods=24)
        time_idx = historic["columns"].index("time_period")
        years = {row[time_idx][:4] for row in historic["data"] + future["data"]}
        assert len(years) >= 2


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

    def test_boolean_default_is_not_mutated_into_int(self) -> None:
        """Verify bool defaults stay boolean across variations (bool is a subclass of int)."""
        generator = TestDataGenerator(seed=42)
        schema = {"properties": {"flag": {"type": "boolean", "default": True}}}
        for variation in range(3):
            data = generator.generate_config_data_from_schema(schema, variation=variation)
            assert data["flag"] is True

    def test_boolean_false_default_is_preserved(self) -> None:
        """Verify a False boolean default is preserved as-is."""
        generator = TestDataGenerator(seed=42)
        schema = {"properties": {"flag": {"type": "boolean", "default": False}}}
        data = generator.generate_config_data_from_schema(schema, variation=2)
        assert data["flag"] is False
