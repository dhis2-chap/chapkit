"""Tests for ML schema discovery from external model scripts."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from chapkit.config.schemas import BaseConfig
from chapkit.ml.schema_discovery import (
    ModelInfo,
    create_config_from_schema,
    discover_model_info,
)


class TestCreateConfigFromSchema:
    """Tests for create_config_from_schema function."""

    def test_empty_schema_returns_base_config(self):
        """Empty schema should return a BaseConfig subclass that accepts any fields."""
        config_class = create_config_from_schema(None)
        assert issubclass(config_class, BaseConfig)

        # Should accept arbitrary fields due to extra="allow"
        instance = config_class(foo="bar", num=42)
        assert instance.foo == "bar"
        assert instance.num == 42

    def test_schema_with_string_property(self):
        """Schema with string property should create appropriate field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"}
            }
        }
        config_class = create_config_from_schema(schema, model_name="TestConfig")

        # Field should exist with None default (not required)
        instance = config_class()
        assert instance.name is None

        # Can set value
        instance = config_class(name="test")
        assert instance.name == "test"

    def test_schema_with_number_property(self):
        """Schema with number property should create float field."""
        schema = {
            "type": "object",
            "properties": {
                "rate": {"type": "number", "default": 0.5}
            }
        }
        config_class = create_config_from_schema(schema)

        instance = config_class()
        assert instance.rate == 0.5

    def test_schema_with_integer_property(self):
        """Schema with integer property should create int field."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": 10}
            }
        }
        config_class = create_config_from_schema(schema)

        instance = config_class()
        assert instance.count == 10

    def test_schema_with_boolean_property(self):
        """Schema with boolean property should create bool field."""
        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": True}
            }
        }
        config_class = create_config_from_schema(schema)

        instance = config_class()
        assert instance.enabled is True

    def test_schema_with_constraints(self):
        """Schema constraints should be converted to Pydantic validators."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            }
        }
        config_class = create_config_from_schema(schema)

        # Valid value
        instance = config_class(value=0.5)
        assert instance.value == 0.5

        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            config_class(value=-0.1)
        with pytest.raises(Exception):
            config_class(value=1.5)

    def test_schema_with_required_field(self):
        """Required fields should not have default values."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"}
            },
            "required": ["required_field"]
        }
        config_class = create_config_from_schema(schema)

        # Should raise error without required field
        with pytest.raises(Exception):
            config_class()

        # Should work with required field
        instance = config_class(required_field="value")
        assert instance.required_field == "value"

    def test_generated_class_name(self):
        """Generated class should have the specified name."""
        schema = {"type": "object", "properties": {}}
        config_class = create_config_from_schema(schema, model_name="MyCustomConfig")
        assert config_class.__name__ == "MyCustomConfig"


class TestDiscoverModelInfo:
    """Tests for discover_model_info function."""

    def test_successful_discovery(self):
        """Should successfully parse valid JSON output."""
        mock_output = json.dumps({
            "service_info": {
                "period_type": "month",
                "required_covariates": ["rainfall"],
                "allows_additional_continuous_covariates": True
            },
            "config_schema": {
                "type": "object",
                "properties": {
                    "smoothing": {"type": "number", "default": 0.5}
                }
            }
        })

        with patch("chapkit.ml.schema_discovery.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["test"],
                returncode=0,
                stdout=mock_output,
                stderr=""
            )

            model_info = discover_model_info("echo test")

            assert model_info.period_type == "month"
            assert model_info.required_covariates == ["rainfall"]
            assert model_info.allows_additional_continuous_covariates is True
            assert issubclass(model_info.config_class, BaseConfig)

    def test_handles_extra_output_before_json(self):
        """Should handle non-JSON output before the JSON object (like R package loading messages)."""
        mock_output = """
Loading required package: dplyr
Attaching package: 'dplyr'
{
    "service_info": {"period_type": "week"},
    "config_schema": null
}
"""
        with patch("chapkit.ml.schema_discovery.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["test"],
                returncode=0,
                stdout=mock_output,
                stderr=""
            )

            model_info = discover_model_info("echo test")
            assert model_info.period_type == "week"

    def test_command_failure_raises_error(self):
        """Should raise RuntimeError on command failure."""
        with patch("chapkit.ml.schema_discovery.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["test"],
                returncode=1,
                stdout="",
                stderr="Command failed"
            )

            with pytest.raises(RuntimeError, match="failed with exit code 1"):
                discover_model_info("failing_command")

    def test_invalid_json_raises_error(self):
        """Should raise RuntimeError on invalid JSON output."""
        with patch("chapkit.ml.schema_discovery.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["test"],
                returncode=0,
                stdout="not valid json without braces",
                stderr=""
            )

            with pytest.raises(RuntimeError, match="No JSON object found"):
                discover_model_info("test_command")

    def test_timeout_raises_error(self):
        """Should raise RuntimeError on timeout."""
        with patch("chapkit.ml.schema_discovery.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=1.0)

            with pytest.raises(RuntimeError, match="timed out"):
                discover_model_info("slow_command", timeout=1.0)

    def test_null_config_schema_handled(self):
        """Should handle null config_schema gracefully."""
        mock_output = json.dumps({
            "service_info": {"period_type": "any"},
            "config_schema": None
        })

        with patch("chapkit.ml.schema_discovery.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["test"],
                returncode=0,
                stdout=mock_output,
                stderr=""
            )

            model_info = discover_model_info("echo test")
            assert model_info.config_schema is None
            # Should still have a usable config class
            instance = model_info.config_class(arbitrary_field="value")
            assert instance.arbitrary_field == "value"


class TestModelInfo:
    """Tests for ModelInfo class."""

    def test_properties_return_defaults(self):
        """Properties should return sensible defaults when not in service_info."""
        model_info = ModelInfo(
            service_info={},
            config_schema=None,
            config_class=BaseConfig,
        )

        assert model_info.period_type == "any"
        assert model_info.required_covariates == []
        assert model_info.allows_additional_continuous_covariates is False

    def test_properties_return_values(self):
        """Properties should return values from service_info."""
        model_info = ModelInfo(
            service_info={
                "period_type": "month",
                "required_covariates": ["temp", "rainfall"],
                "allows_additional_continuous_covariates": True,
            },
            config_schema={"type": "object"},
            config_class=BaseConfig,
        )

        assert model_info.period_type == "month"
        assert model_info.required_covariates == ["temp", "rainfall"]
        assert model_info.allows_additional_continuous_covariates is True


@pytest.mark.skipif(
    subprocess.run(["which", "Rscript"], capture_output=True).returncode != 0,
    reason="Rscript not available"
)
class TestRealRIntegration:
    """Integration tests with actual R scripts (requires R and chap.r.sdk installed)."""

    def test_discover_from_r_example(self):
        """Test discovery from the actual R example in the repo."""
        # This test only runs if R is available
        try:
            model_info = discover_model_info(
                "Rscript examples/ml_r/model.R info --format json",
                cwd=Path(__file__).parent.parent,  # chapkit root
                model_name="TestMeanModel"
            )

            # Check service info
            assert model_info.period_type == "any"
            assert model_info.allows_additional_continuous_covariates is False

            # Check config class was created with expected fields
            schema = model_info.config_class.model_json_schema()
            assert "smoothing" in schema.get("properties", {})
            assert "min_observations" in schema.get("properties", {})

        except RuntimeError as e:
            if "chap.r.sdk" in str(e) or "there is no package" in str(e):
                pytest.skip("chap.r.sdk R package not installed")
            raise
