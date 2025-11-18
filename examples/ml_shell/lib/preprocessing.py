"""Data preprocessing utilities shared between train and predict scripts.

This module demonstrates how ShellModelRunner enables code reuse through
relative imports. Both train_model.py and predict_model.py can import
these utilities because the entire project is copied to the workspace.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional features from raw climate data.

    Args:
        df: DataFrame with climate features

    Returns:
        DataFrame with engineered features added
    """
    result = df.copy()

    # Add interaction features (realistic for disease prediction)
    if "rainfall" in df.columns and "mean_temperature" in df.columns:
        result["temp_rainfall_interaction"] = df["rainfall"] * df["mean_temperature"]

    if "mean_temperature" in df.columns and "humidity" in df.columns:
        result["temp_humidity_interaction"] = df["mean_temperature"] * df["humidity"]

    return result


def normalize_data(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Normalize features using min-max scaling.

    Args:
        df: DataFrame with features
        feature_cols: Column names to normalize

    Returns:
        Tuple of (normalized DataFrame, normalization parameters dict)
    """
    result = df.copy()
    params = {}

    for col in feature_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()

            if max_val > min_val:
                result[col] = (df[col] - min_val) / (max_val - min_val)
                params[col] = {"min": min_val, "max": max_val}

    return result, params


def apply_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply pre-computed normalization parameters to new data.

    Args:
        df: DataFrame to normalize
        params: Normalization parameters from normalize_data()

    Returns:
        Normalized DataFrame
    """
    result = df.copy()

    for col, p in params.items():
        if col in df.columns:
            result[col] = (df[col] - p["min"]) / (p["max"] - p["min"])

    return result
