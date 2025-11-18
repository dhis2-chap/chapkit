#!/usr/bin/env python3
"""External training script for ML model training via ShellModelRunner.

This script demonstrates language-agnostic ML training by:
- Reading config from YAML
- Loading training data from CSV
- Training a model with feature engineering
- Saving the model to disk
- Using relative imports from shared lib/ utilities

Usage:
    python train_model.py --config config.yml --data data.csv --model model.pickle
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

# Add project root to sys.path for relative imports
# This works because ShellModelRunner sets cwd to the project root
sys.path.insert(0, str(Path.cwd()))

# Import shared utilities using relative imports
from lib.preprocessing import engineer_features  # type: ignore[import-not-found]
from lib.validation import validate_training_data  # type: ignore[import-not-found]


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML model from external script")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--model", required=True, help="Path to save trained model")
    parser.add_argument("--geo", default="", help="Path to GeoJSON file (optional)")

    args = parser.parse_args()

    try:
        # Load config
        with open(args.config) as f:
            config = yaml.safe_load(f)

        print(f"Training with config: {config}", file=sys.stderr)
        min_samples = config.get("min_samples", 3)

        # Load training data
        data = pd.read_csv(args.data)
        print(f"Loaded {len(data)} training samples", file=sys.stderr)

        # Extract features and target
        feature_cols = ["rainfall", "mean_temperature", "humidity"]
        target_col = "disease_cases"

        # VALIDATION: Use shared validation utility
        validate_training_data(data, feature_cols + [target_col], min_samples)

        # PREPROCESSING: Engineer features using shared utility
        data = engineer_features(data)
        print("Engineered additional features", file=sys.stderr)

        # Update feature columns to include engineered features
        engineered_features = [
            "temp_rainfall_interaction",
            "temp_humidity_interaction",
        ]
        all_features = feature_cols + engineered_features

        X = data[all_features]
        y = data[target_col].fillna(0)

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        print(f"Model trained with {len(all_features)} features", file=sys.stderr)
        print(f"Coefficients: {model.coef_.tolist()}", file=sys.stderr)

        # Save model with feature metadata
        model_data = {
            "model": model,
            "feature_cols": all_features,
            "config": config,
        }

        with open(args.model, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {args.model}", file=sys.stderr)
        print("SUCCESS: Training completed")

    except Exception as e:
        print(f"ERROR: Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
