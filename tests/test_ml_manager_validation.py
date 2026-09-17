"""Tests for MLManager prediction_periods validation."""

import pytest

from chapkit.ml import MLManager


class TestValidatePredictionPeriods:
    """Tests for _validate_prediction_periods method."""

    def test_within_bounds_passes(self) -> None:
        """Test that prediction_periods within bounds does not raise."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 10

        manager._validate_prediction_periods(5, "config")

    def test_at_minimum_bound_passes(self) -> None:
        """Test that prediction_periods at minimum bound does not raise."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 3
        manager.max_prediction_periods = 10

        manager._validate_prediction_periods(3, "config")

    def test_at_maximum_bound_passes(self) -> None:
        """Test that prediction_periods at maximum bound does not raise."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 5

        manager._validate_prediction_periods(5, "config")

    def test_below_minimum_raises_value_error(self) -> None:
        """Test that prediction_periods below minimum raises ValueError."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 5
        manager.max_prediction_periods = 10

        with pytest.raises(ValueError) as exc_info:
            manager._validate_prediction_periods(3, "config")

        assert "prediction_periods (3, from config)" in str(exc_info.value)
        assert "below the minimum" in str(exc_info.value)
        assert "(5)" in str(exc_info.value)

    def test_above_maximum_raises_value_error(self) -> None:
        """Test that prediction_periods above maximum raises ValueError."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 5

        with pytest.raises(ValueError) as exc_info:
            manager._validate_prediction_periods(10, "config")

        assert "prediction_periods (10, from config)" in str(exc_info.value)
        assert "exceeds the maximum" in str(exc_info.value)
        assert "(5)" in str(exc_info.value)

    def test_message_names_the_resolution_source(self) -> None:
        """Test that the diagnostic message names where the horizon came from."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 1
        manager.max_prediction_periods = 10

        with pytest.raises(ValueError) as exc_info:
            manager._validate_prediction_periods(12, "run_info")

        assert "prediction_periods (12, from run_info) exceeds the maximum allowed value (10)" in str(exc_info.value)

    def test_default_bounds(self) -> None:
        """Test that default bounds are 0 and 100."""
        manager = MLManager.__new__(MLManager)
        manager.min_prediction_periods = 0
        manager.max_prediction_periods = 100

        # Should pass with values within default range
        manager._validate_prediction_periods(50, "config")

        # Should pass at boundaries
        manager._validate_prediction_periods(0, "config")
        manager._validate_prediction_periods(100, "config")
