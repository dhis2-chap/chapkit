"""R-based ML example using external R scripts for train/predict.

This example demonstrates:
- Using ShellModelRunner to execute R scripts via Rscript
- Integration with the chap.r.sdk R package
- File-based data interchange (CSV, YAML)
- The info command for model metadata and schema discovery

Prerequisites:
- R installed with Rscript available in PATH
- chap.r.sdk R package installed: install.packages("chap.r.sdk")

The R model script uses chap.r.sdk's create_chapkit_cli() which provides:
- Automatic CSV/YAML parsing
- tsibble conversion for time series data
- Configuration schema validation
- run_info handling
"""

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner
from pydantic import Field


class MeanModelConfig(BaseConfig):
    """Configuration for mean prediction model."""

    smoothing: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Smoothing factor for exponential smoothing (0 = no smoothing)",
    )
    min_observations: int = Field(
        default=1,
        ge=1,
        description="Minimum observations required per location",
    )


# Create shell-based runner for R scripts
# Uses chap.r.sdk's create_chapkit_cli() interface with named arguments
#
# The R SDK handles:
#   - Loading CSV data as tsibbles
#   - Parsing YAML configuration
#   - Loading run_info from YAML
#   - Saving predictions in the correct format
#
# Variables substituted at runtime:
#   {data_file} - Training data CSV
#   {historic_file} - Historic data CSV
#   {future_file} - Future data CSV
#   {output_file} - Predictions CSV output path
#   {run_info_file} - Run info YAML with prediction_length, etc.

train_command = (
    "Rscript model.R train "
    "--data {data_file} "
    "--run-info {run_info_file}"
)

predict_command = (
    "Rscript model.R predict "
    "--historic {historic_file} "
    "--future {future_file} "
    "--output {output_file} "
    "--run-info {run_info_file}"
)

# Create shell model runner for R
runner: ShellModelRunner[MeanModelConfig] = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
)

# Create ML service info with metadata
info = MLServiceInfo(
    display_name="R Mean Model Service",
    version="1.0.0",
    summary="Simple mean-based prediction model implemented in R",
    description=(
        "Demonstrates R model integration with chapkit using chap.r.sdk. "
        "Predicts future values based on historical mean per location."
    ),
    author="CHAP Team",
    author_note="Example R integration using chap.r.sdk package",
    author_assessed_status=AssessedStatus.yellow,
    contact_email="chap@example.com",
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="r_ml_pipeline",
    level_labels={0: "ml_training", 1: "ml_prediction"},
)

# Build the FastAPI application
app = (
    MLServiceBuilder(
        info=info,
        config_schema=MeanModelConfig,
        hierarchy=HIERARCHY,
        runner=runner,
    )
    .with_monitoring()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app", reload=False)
