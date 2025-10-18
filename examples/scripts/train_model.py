#!/usr/bin/env python3
"""External training script for ML model training via ShellModelRunner.

This script demonstrates language-agnostic ML training by:
- Reading config from JSON
- Loading training data from CSV
- Training a model
- Saving the model to disk

Usage:
    python train_model.py --config config.json --data data.csv --model model.pickle
"""

import argparse
import json
import pickle
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML model from external script")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--data", required=True, help="Path to training data CSV")
    parser.add_argument("--model", required=True, help="Path to save trained model")
    parser.add_argument("--geo", default="", help="Path to GeoJSON file (optional)")

    args = parser.parse_args()

    try:
        # Load config
        with open(args.config) as f:
            config = json.load(f)

        print(f"Training with config: {config}", file=sys.stderr)

        # Load training data
        data = pd.read_csv(args.data)
        print(f"Loaded {len(data)} training samples", file=sys.stderr)

        # Extract features and target
        feature_cols = ["rainfall", "mean_temperature", "humidity"]
        target_col = "disease_cases"

        X = data[feature_cols]
        y = data[target_col].fillna(0)

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        print(f"Model trained with coefficients: {model.coef_.tolist()}", file=sys.stderr)

        # Save model
        with open(args.model, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved to {args.model}", file=sys.stderr)
        print("SUCCESS: Training completed")

    except Exception as e:
        print(f"ERROR: Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
