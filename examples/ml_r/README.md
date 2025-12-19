# R Model Integration Example

This example demonstrates how to integrate R models with chapkit using the `chap.r.sdk` package.

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
├── main.py      # Chapkit service definition
├── model.R      # R model implementation using chap.r.sdk
├── pyproject.toml
└── README.md
```

## How It Works

### 1. R Model Script (`model.R`)

The R model uses `chap.r.sdk::create_chapkit_cli()` which provides:
- Automatic CSV loading and tsibble conversion
- YAML configuration parsing with schema validation
- run_info handling for runtime parameters
- Prediction output in chapkit-compatible format

```r
library(chap.r.sdk)

# Define your model functions
train_fn <- function(training_data, model_configuration, run_info) {
  # training_data is already a tsibble
  # Return model object (will be saved as RDS)
}

predict_fn <- function(historic_data, future_data, saved_model,
                       model_configuration, run_info) {
  # Return tibble with 'samples' list-column
}

# Create CLI with one line
create_chapkit_cli(train_fn, predict_fn, config_schema, model_info)
```

### 2. Python Service (`main.py`)

Uses `ShellModelRunner` to execute R scripts:

```python
from chapkit.ml import ShellModelRunner

runner = ShellModelRunner(
    train_command="Rscript model.R train --data {data_file} --run-info {run_info_file}",
    predict_command="Rscript model.R predict --historic {historic_file} --future {future_file} --output {output_file} --run-info {run_info_file}"
)
```

### 3. Model Metadata Discovery

The R SDK's `info` command provides structured JSON output for chapkit:

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
