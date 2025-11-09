"""Shell-based ML example using external Python scripts for train/predict.

This example demonstrates:
- Using ShellModelRunner to execute external scripts
- Command template variable substitution
- Language-agnostic ML workflows (could use R, Julia, etc.)
- File-based data interchange (CSV, YAML, pickle)
- Integration with existing scripts without modification
"""

import sys
from pathlib import Path

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner

# Get absolute path to scripts directory
SCRIPTS_DIR = Path(__file__).parent / "scripts"


class DiseaseConfig(BaseConfig):
    """Configuration for disease prediction model."""

    # Config fields can be accessed by external scripts via config.yml
    min_samples: int = 3
    model_type: str = "linear_regression"


# Create shell-based runner with command templates
# Variables will be substituted with actual file paths at runtime:
#   {config_file} - YAML config
#   {data_file} - Training data CSV
#   {model_file} - Model pickle file
#   {historic_file} - Historic data CSV
#   {future_file} - Future data CSV
#   {output_file} - Predictions CSV

# Training command template
train_command = (
    f"{sys.executable} {SCRIPTS_DIR}/train_model.py "
    "--config {config_file} "
    "--data {data_file} "
    "--model {model_file}"
)

# Prediction command template
predict_command = (
    f"{sys.executable} {SCRIPTS_DIR}/predict_model.py "
    "--config {config_file} "
    "--model {model_file} "
    "--historic {historic_file} "
    "--future {future_file} "
    "--output {output_file}"
)

# Create shell model runner
runner: ShellModelRunner[DiseaseConfig] = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
    model_format="pickle",
)

# Create ML service info with metadata
info = MLServiceInfo(
    display_name="Shell-Based Disease Prediction Service",
    version="1.0.0",
    summary="ML service using external scripts for train/predict",
    description="Demonstrates language-agnostic ML workflows with file-based data interchange using Python scripts",
    author="ML Engineering Team",
    author_note="Language-agnostic approach allows integration with R, Julia, and other tools",
    author_assessed_status=AssessedStatus.orange,
    contact_email="mleng@example.com",
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="shell_ml_pipeline",
    level_labels={0: "ml_training", 1: "ml_prediction"},
)

# Build the FastAPI application
app = (
    MLServiceBuilder(
        info=info,
        config_schema=DiseaseConfig,
        hierarchy=HIERARCHY,
        runner=runner,
    )
    .with_monitoring()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app")
