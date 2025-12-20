# R Model Integration Example

This example demonstrates how to integrate R models with chapkit using the `chap.r.sdk` package, including **automatic schema discovery** from R models.

## Prerequisites

1. **R** installed with `Rscript` available in PATH
2. **chap.r.sdk** R package installed:
   ```r
   # From GitHub (development version)
   remotes::install_github("dhis2/chap_r_sdk")

   # Or from local source
   install.packages("/path/to/chap_r_sdk", repos = NULL, type = "source")
   ```

## Project Structure

```
ml_r/
├── main.py      # Chapkit service definition with schema discovery
├── model.R      # R model implementation using chap.r.sdk
├── pyproject.toml
└── README.md
```

## Key Feature: Automatic Schema Discovery

Instead of duplicating the configuration schema in both R and Python, chapkit can **discover the schema from your R model** at startup:

```python
from chapkit.ml import ShellModelRunner, discover_model_info

# Discover schema and service info from R model
model_info = discover_model_info("Rscript model.R info --format json")

# Use the discovered config class
app = MLServiceBuilder(
    info=info,
    config_schema=model_info.config_class,  # Dynamically created from R schema!
    ...
)
```

This means you only define your schema once (in R) and chapkit automatically:
- Creates a Pydantic model with the correct fields
- Exposes it via `GET /api/v1/configs/$schema`
- Uses it for validation

## How It Works

### 1. R Model Script (`model.R`)

The R model uses `chap.r.sdk::create_chapkit_cli()` which provides:
- Automatic CSV loading and tsibble conversion
- YAML configuration parsing with schema validation
- run_info handling for runtime parameters
- Prediction output in chapkit-compatible format
- **`info --format json` command for schema discovery**

```r
library(chap.r.sdk)

# Define config schema (source of truth)
config_schema <- create_config_schema(
  title = "My Model Configuration",
  properties = list(
    smoothing = schema_number(default = 0.5, minimum = 0, maximum = 1),
    min_observations = schema_integer(default = 1, minimum = 1)
  )
)

# Define model info
model_info <- list(
  period_type = "month",
  required_covariates = c("rainfall"),
  allows_additional_continuous_covariates = TRUE
)

# Create CLI - enables info, train, and predict commands
create_chapkit_cli(train_fn, predict_fn, config_schema, model_info)
```

### 2. Python Service (`main.py`)

Uses `discover_model_info()` to get the schema from R:

```python
from chapkit.ml import ShellModelRunner, discover_model_info

# Discover schema from R model
model_info = discover_model_info(
    "Rscript model.R info --format json",
    model_name="MeanModelConfig"
)

# model_info contains:
# - config_class: Pydantic BaseConfig subclass
# - service_info: Dict with period_type, required_covariates, etc.
# - period_type, required_covariates, allows_additional_continuous_covariates

runner = ShellModelRunner(
    train_command="Rscript model.R train --data {data_file} --run-info {run_info_file}",
    predict_command="Rscript model.R predict --historic {historic_file} --future {future_file} --output {output_file} --run-info {run_info_file}"
)
```

### 3. Model Metadata Discovery

The R SDK's `info` command provides structured JSON output:

```bash
Rscript model.R info --format json
```

Output:
```json
{
  "service_info": {
    "period_type": "any",
    "required_covariates": [],
    "allows_additional_continuous_covariates": false
  },
  "config_schema": {
    "title": "Mean Model Configuration",
    "type": "object",
    "properties": { ... }
  }
}
```

## Running the Example

1. **Start the service**:
   ```bash
   cd examples/ml_r
   python main.py
   ```

2. **Test the R model directly**:
   ```bash
   # Show model info as JSON
   Rscript model.R info --format json

   # Show model info as YAML (human-readable)
   Rscript model.R info
   ```

3. **Use the API**:
   ```bash
   # Create a configuration
   curl -X POST http://localhost:8000/api/v1/configs \
     -H "Content-Type: application/json" \
     -d '{"data": {"smoothing": 0.5, "min_observations": 3}}'

   # Train the model
   curl -X POST http://localhost:8000/api/v1/ml/\$train \
     -H "Content-Type: application/json" \
     -d '{
       "config_id": "<config_id>",
       "run_info": {"prediction_length": 3},
       "data": {"columns": [...], "data": [...]}
     }'
   ```

## File Interchange

ShellModelRunner creates these files in the workspace:

| File | Description |
|------|-------------|
| `config.yml` | Model configuration in YAML format |
| `run_info.yml` | Runtime info (prediction_length, additional_covariates, etc.) |
| `data.csv` | Training data (for train command) |
| `historic.csv` | Historic data (for predict command) |
| `future.csv` | Future periods to predict (for predict command) |
| `predictions.csv` | Model output (created by predict command) |
| `model.rds` | Trained model object (created by train command) |

## Configuration Schema

The R model defines its configuration schema using `chap.r.sdk` helpers:

```r
config_schema <- create_config_schema(
  title = "Mean Model Configuration",
  properties = list(
    smoothing = schema_number(default = 0.0, minimum = 0.0, maximum = 1.0),
    min_observations = schema_integer(default = 1L, minimum = 1L)
  )
)
```

This schema is:
- Validated by the R SDK when loading configuration
- Exposed via `info --format json` for chapkit integration
- Used by chapkit for UI generation and validation
