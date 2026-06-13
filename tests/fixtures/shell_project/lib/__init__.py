"""Shared utilities for ML workflows.

This demonstrates how ShellModelRunner supports relative imports
by copying the entire project directory to an isolated workspace.
"""

from lib.preprocessing import engineer_features, normalize_data  # type: ignore[import-not-found]
from lib.validation import validate_predictions, validate_training_data  # type: ignore[import-not-found]

__all__ = [
    "engineer_features",
    "normalize_data",
    "validate_training_data",
    "validate_predictions",
]
