#!/usr/bin/env python3
"""External prediction script for ML model inference via ShellModelRunner.

This script demonstrates language-agnostic ML prediction by:
- Reading config from YAML
- Loading trained model from disk
- Loading future data from CSV
- Applying feature engineering
- Making predictions
- Saving predictions to CSV
- Using relative imports from shared lib/ utilities

Usage:
    python predict_model.py --config config.yml --model model.pickle --future future.csv --output predictions.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to sys.path for relative imports
# This works because ShellModelRunner sets cwd to the project root
sys.path.insert(0, str(Path.cwd()))

# Import shared utilities using relative imports
from lib.preprocessing import engineer_features  # type: ignore[import-not-found]
from lib.validation import validate_predictions  # type: ignore[import-not-found]


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--historic", required=True, help="Path to historic data CSV")
    parser.add_argument("--future", required=True, help="Path to future data CSV")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    parser.add_argument("--geo", default="", help="Path to GeoJSON file (optional)")

    args = parser.parse_args()

    try:
        # Load config
        with open(args.config) as f:
            config = yaml.safe_load(f)

        print(f"Predicting with config: {config}", file=sys.stderr)

        # Load model
        with open(args.model, "rb") as f:
            model_data = pickle.load(f)

        # Handle both old and new model format for backward compatibility
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            feature_cols = model_data["feature_cols"]
            print(f"Loaded model with {len(feature_cols)} features", file=sys.stderr)
        else:
            # Old format: just the sklearn model
            model = model_data
            feature_cols = ["rainfall", "mean_temperature", "humidity"]
            print("Loaded legacy model format", file=sys.stderr)

        # Load future data
        future = pd.read_csv(args.future)
        print(f"Loaded {len(future)} prediction samples", file=sys.stderr)

        # PREPROCESSING: Engineer features using shared utility
        future = engineer_features(future)
        print("Engineered features for prediction", file=sys.stderr)

        # Extract features
        X = future[feature_cols]

        # Make predictions
        predictions = model.predict(X)  # pyright: ignore[reportAttributeAccessIssue]

        # VALIDATION: Validate predictions using shared utility
        validate_predictions(pd.Series(predictions))

        # Clip negative predictions
        predictions = pd.Series(predictions).clip(lower=0)

        # Add predictions to dataframe
        future["sample_0"] = predictions

        print(f"Made {len(predictions)} predictions", file=sys.stderr)

        # Save predictions
        future.to_csv(args.output, index=False)

        print(f"Predictions saved to {args.output}", file=sys.stderr)
        print("SUCCESS: Prediction completed")

    except Exception as e:
        print(f"ERROR: Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
