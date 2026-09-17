"""Tests for chapkit test command helpers."""

from __future__ import annotations

from chapkit.cli.test.command import clamp_prediction_periods


class TestClampPredictionPeriods:
    """Tests for clamping the generated horizon into the service's declared bounds."""

    def test_within_bounds_is_unchanged(self) -> None:
        """A requested horizon inside the bounds is returned as is."""
        assert clamp_prediction_periods(30, 0, 100) == 30

    def test_above_maximum_is_lowered_to_maximum(self) -> None:
        """A service with a small maximum gets a correspondingly short horizon."""
        assert clamp_prediction_periods(30, 0, 12) == 12

    def test_below_minimum_is_raised_to_minimum(self) -> None:
        """A service with a minimum above the request gets the minimum."""
        assert clamp_prediction_periods(2, 4, 100) == 4

    def test_never_below_one(self) -> None:
        """A zero minimum still yields at least one period."""
        assert clamp_prediction_periods(0, 0, 100) == 1

    def test_maximum_below_minimum_uses_minimum(self) -> None:
        """Inconsistent bounds resolve to the minimum rather than an empty range."""
        assert clamp_prediction_periods(5, 8, 3) == 8
