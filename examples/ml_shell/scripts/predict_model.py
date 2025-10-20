#!/usr/bin/env python3
"""External prediction script for ML model inference via ShellModelRunner.

This script demonstrates language-agnostic ML prediction by:
- Reading config from JSON
- Loading trained model from disk
- Loading future data from CSV
- Making predictions
- Saving predictions to CSV

Usage:
    python predict_model.py --config config.json --model model.pickle --future future.csv --output predictions.csv
"""

import argparse
import json
import pickle
import sys

import pandas as pd


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--historic", required=True, help="Path to historic data CSV")
    parser.add_argument("--future", required=True, help="Path to future data CSV")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")
    parser.add_argument("--geo", default="", help="Path to GeoJSON file (optional)")

    args = parser.parse_args()

    try:
        # Load config
        with open(args.config) as f:
            config = json.load(f)

        print(f"Predicting with config: {config}", file=sys.stderr)

        # Load model
        with open(args.model, "rb") as f:
            model = pickle.load(f)

        print(f"Model loaded from {args.model}", file=sys.stderr)

        # Load future data
        future = pd.read_csv(args.future)
        print(f"Loaded {len(future)} prediction samples", file=sys.stderr)

        # Extract features
        feature_cols = ["rainfall", "mean_temperature", "humidity"]
        X = future[feature_cols]

        # Make predictions
        predictions = model.predict(X)

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
