"""Shell-based ML example using external Python scripts for train/predict.

This example demonstrates:
- Using ShellModelRunner to execute external scripts
- Command template variable substitution
- Language-agnostic ML workflows (could use R, Julia, etc.)
- File-based data interchange (CSV, YAML, pickle)
- Full project isolation with relative imports support
- Shared utility code (lib/) imported by both train and predict scripts
- Realistic ML workflow with feature engineering and validation
"""

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelCard, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner


class DiseaseConfig(BaseConfig):
    """Configuration for disease prediction model."""

    # Config fields can be accessed by external scripts via config.yml
    prediction_periods: int = 3
    min_samples: int = 3
    model_type: str = "linear_regression"


# Create shell-based runner with command templates
# The runner copies the entire project directory to an isolated workspace
# and executes commands with the workspace as the current directory.
# This allows scripts to:
#   1. Use relative paths to access files
#   2. Import shared utilities via relative imports (e.g., from lib import ...)
#   3. Access any project files needed for the workflow
#
# In this example, both train_model.py and predict_model.py import from lib/
# to share preprocessing and validation utilities. This demonstrates how
# ShellModelRunner enables proper project organization with reusable code.
#
# Variables will be substituted with actual file paths at runtime:
#   {data_file} - Training data CSV
#   {historic_file} - Historic data CSV
#   {future_file} - Future data CSV
#   {output_file} - Predictions CSV
#
# Files available in workspace (scripts can access directly):
#   config.yml - YAML config (always available)
#   model.pickle - Model file (create/use as needed)

# Training command template (using relative path to script)
train_command = "python scripts/train_model.py --data {data_file}"

# Prediction command template (using relative path to script)
predict_command = (
    "python scripts/predict_model.py --historic {historic_file} --future {future_file} --output {output_file}"
)

# Create shell model runner
runner: ShellModelRunner[DiseaseConfig] = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
)

# Create ML service info with metadata
info = MLServiceInfo(
    display_name="Shell-Based Disease Prediction Service",
    version="1.0.0",
    summary="ML service using external scripts for train/predict",
    description="Demonstrates language-agnostic ML workflows with file-based data interchange using Python scripts",
    model_card=ModelCard(
        author="ML Engineering Team",
        author_note="Language-agnostic approach allows integration with R, Julia, and other tools",
        author_assessed_status=AssessedStatus.orange,
        contact_email="mleng@example.com",
    ),
    period_type=PeriodType.monthly,
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="shell_ml_pipeline",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
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

    run_app("main:app", reload=False)
