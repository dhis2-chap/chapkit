"""Tests for MLManager prediction_periods validation."""

import pytest

from chapkit.config import BaseConfig
from chapkit.ml import MLManager


class SampleConfig(BaseConfig):
    """Test configuration schema."""

    prediction_periods: int = 3


class TestValidatePredictionPeriods:
    """Tests for _validate_prediction_periods method."""

    def test_within_bounds_passes(self) -> None:
        """Test that prediction_periods within bounds does not raise."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 10

        config_data = SampleConfig(prediction_periods=5)
        manager._validate_prediction_periods(config_data)

    def test_at_minimum_bound_passes(self) -> None:
        """Test that prediction_periods at minimum bound does not raise."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 3
        manager.max_prediction_periods = 10

        config_data = SampleConfig(prediction_periods=3)
        manager._validate_prediction_periods(config_data)

    def test_at_maximum_bound_passes(self) -> None:
        """Test that prediction_periods at maximum bound does not raise."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 5

        config_data = SampleConfig(prediction_periods=5)
        manager._validate_prediction_periods(config_data)

    def test_below_minimum_raises_value_error(self) -> None:
        """Test that prediction_periods below minimum raises ValueError."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 5
        manager.max_prediction_periods = 10

        config_data = SampleConfig(prediction_periods=3)

        with pytest.raises(ValueError) as exc_info:
            manager._validate_prediction_periods(config_data)

        assert "prediction_periods (3)" in str(exc_info.value)
        assert "below the minimum" in str(exc_info.value)
        assert "(5)" in str(exc_info.value)

    def test_above_maximum_raises_value_error(self) -> None:
        """Test that prediction_periods above maximum raises ValueError."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 5

        config_data = SampleConfig(prediction_periods=10)

        with pytest.raises(ValueError) as exc_info:
            manager._validate_prediction_periods(config_data)

        assert "prediction_periods (10)" in str(exc_info.value)
        assert "exceeds the maximum" in str(exc_info.value)
        assert "(5)" in str(exc_info.value)

    def test_default_bounds(self) -> None:
        """Test that default bounds are 0 and 100."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 0
        manager.max_prediction_periods = 100

        # Should pass with values within default range
        config_data = SampleConfig(prediction_periods=50)
        manager._validate_prediction_periods(config_data)

        # Should pass at boundaries
        config_data_min = SampleConfig(prediction_periods=0)
        manager._validate_prediction_periods(config_data_min)

        config_data_max = SampleConfig(prediction_periods=100)
        manager._validate_prediction_periods(config_data_max)
