"""R-based ML example using external R scripts for train/predict.

This example demonstrates:
- Using ShellModelRunner to execute R scripts via Rscript
- **Automatic schema discovery** from R model's `info --format json` command
- Integration with the chap.r.sdk R package
- File-based data interchange (CSV, YAML)

Prerequisites:
- R installed with Rscript available in PATH
- chap.r.sdk R package installed: install.packages("chap.r.sdk")

The R model script uses chap.r.sdk's create_chapkit_cli() which provides:
- Automatic CSV/YAML parsing
- tsibble conversion for time series data
- Configuration schema validation
- run_info handling
- `info --format json` for schema discovery
"""

from pathlib import Path

from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner, discover_model_info

# Get the directory where this script lives (for finding model.R)
SCRIPT_DIR = Path(__file__).parent

# ============================================================================
# Schema Discovery from R Model
# ============================================================================
# Instead of duplicating the config schema in Python, we discover it from
# the R model's `info --format json` command. This ensures the Python service
# always uses the same schema as the R model.

model_info = discover_model_info(
    "Rscript model.R info --format json",
    model_name="MeanModelConfig",
    cwd=SCRIPT_DIR,
)

# The discovered model_info contains:
# - model_info.config_class: Pydantic BaseConfig subclass for configuration
# - model_info.service_info: Dict with period_type, required_covariates, etc.
# - model_info.period_type: Shortcut to service_info["period_type"]
# - model_info.required_covariates: Shortcut to service_info["required_covariates"]

print(f"Discovered config schema: {model_info.config_class.model_json_schema()}")


# ============================================================================
# Shell Model Runner
# ============================================================================
# Uses chap.r.sdk's create_chapkit_cli() interface with named arguments
#
# Variables substituted at runtime:
#   {data_file} - Training data CSV
#   {historic_file} - Historic data CSV
#   {future_file} - Future data CSV
#   {output_file} - Predictions CSV output path
#   {run_info_file} - Run info YAML with prediction_length, etc.

MODEL_SCRIPT = SCRIPT_DIR / "model.R"

train_command = (
    f"Rscript {MODEL_SCRIPT} train "
    "--data {data_file} "
    "--run-info {run_info_file}"
)

predict_command = (
    f"Rscript {MODEL_SCRIPT} predict "
    "--historic {historic_file} "
    "--future {future_file} "
    "--output {output_file} "
    "--run-info {run_info_file}"
)

runner = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
)


# ============================================================================
# Service Configuration
# ============================================================================
# Use discovered model_info to populate service metadata

info = MLServiceInfo(
    display_name="R Mean Model Service",
    version="1.0.0",
    summary="Simple mean-based prediction model implemented in R",
    description=(
        "Demonstrates R model integration with chapkit using chap.r.sdk. "
        "Predicts future values based on historical mean per location. "
        "Configuration schema is automatically discovered from the R model."
    ),
    author="CHAP Team",
    author_note="Example R integration with automatic schema discovery",
    author_assessed_status=AssessedStatus.yellow,
    contact_email="chap@example.com",
    # Use discovered service info
    supported_period_type=PeriodType(model_info.period_type),
    required_covariates=model_info.required_covariates,
    allow_free_additional_continuous_covariates=model_info.allows_additional_continuous_covariates,
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="r_ml_pipeline",
    level_labels={0: "ml_training", 1: "ml_prediction"},
)

# Build the FastAPI application using the discovered config schema
app = (
    MLServiceBuilder(
        info=info,
        config_schema=model_info.config_class,  # Use discovered schema!
        hierarchy=HIERARCHY,
        runner=runner,
    )
    .with_monitoring()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app", reload=False)
