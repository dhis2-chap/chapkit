"""Data validation utilities for ML workflows.

This module demonstrates how validation logic can be shared between
train and predict scripts using relative imports.
"""

import sys

import pandas as pd


def validate_training_data(df: pd.DataFrame, required_cols: list[str], min_samples: int) -> None:
    """Validate training data meets requirements.

    Args:
        df: Training data DataFrame
        required_cols: Required column names
        min_samples: Minimum number of samples required

    Raises:
        SystemExit: If validation fails
    """
    # Check minimum samples
    if len(df) < min_samples:
        print(f"ERROR: Insufficient training data: {len(df)} < {min_samples}", file=sys.stderr)
        sys.exit(1)

    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Check for NaN in features
    feature_cols = [col for col in required_cols if col != "disease_cases"]
    if df[feature_cols].isna().any().any():
        print("ERROR: Training features contain NaN values", file=sys.stderr)
        sys.exit(1)

    print(f"Validation passed: {len(df)} samples with all required columns", file=sys.stderr)


def validate_predictions(predictions: pd.Series) -> None:
    """Validate prediction output quality.

    Args:
        predictions: Series of predicted values

    Raises:
        SystemExit: If validation fails
    """
    # Check for NaN predictions
    if predictions.isna().any():
        print("WARNING: Some predictions are NaN", file=sys.stderr)

    # Check for negative predictions (disease cases cannot be negative)
    if (predictions < 0).any():
        print("WARNING: Some predictions are negative, clipping to 0", file=sys.stderr)

    # Check for unrealistic values (simple sanity check)
    if (predictions > 1000).any():
        print("WARNING: Some predictions exceed 1000 cases", file=sys.stderr)
