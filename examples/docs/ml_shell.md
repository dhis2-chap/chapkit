# ml_shell.py - Shell-Based ML Service cURL Guide

Shell-based disease prediction ML service using external Python scripts for language-agnostic workflows.

## Quick Start

```bash
# Start the service
fastapi dev examples/ml_shell.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Model**: LinearRegression (via external Python script)
- **Features**: `rainfall`, `mean_temperature`
- **Target**: `disease_cases`
- **Runner**: ShellModelRunner (executes external scripts)
- **Assessment**: Orange (shows promise, needs evaluation)
- **Special**: Language-agnostic - can use Python, R, Julia, or any CLI tool

## Architecture

```
┌──────────────┐
│ FastAPI App  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ShellModelRunner │
└──────┬───────┘
       │
       ├─ Train ──▶ examples/scripts/train_model.py
       │            (reads CSV, writes pickle)
       │
       └─ Predict ▶ examples/scripts/predict_model.py
                    (reads pickle/CSV, writes CSV)
```

### External Scripts

Located in `examples/scripts/`:
- `train_model.py` - Training script
- `predict_model.py` - Prediction script

**Variable Substitution:**
- `{config_file}` - Config JSON path
- `{data_file}` - Training CSV path
- `{model_file}` - Model pickle path
- `{future_file}` - Future data CSV path
- `{output_file}` - Output CSV path

## Complete Workflow

### 1. Check Service Health

```bash
curl http://127.0.0.1:8000/health
```

### 2. View System Info

```bash
curl http://127.0.0.1:8000/api/v1/system
```

**Response:**
```json
{
  "current_time": "2025-10-11T17:30:00Z",
  "timezone": "CEST",
  "python_version": "3.13.8",
  "platform": "macOS-26.0.1-arm64-arm-64bit-Mach-O",
  "hostname": "mlaptop.local"
}
```

### 3. View Service Metadata

```bash
curl http://127.0.0.1:8000/api/v1/info
```

**Response:**
```json
{
  "display_name": "Shell-Based Disease Prediction Service",
  "version": "1.0.0",
  "summary": "ML service using external scripts for train/predict",
  "description": "Demonstrates language-agnostic ML workflows with file-based data interchange using Python scripts",
  "author": "ML Engineering Team",
  "author_note": "Language-agnostic approach allows integration with R, Julia, and other tools",
  "author_assessed_status": "orange",
  "contact_email": "mleng@example.com"
}
```

### 4. Get Config Schema

```bash
curl http://127.0.0.1:8000/api/v1/configs/\$schema
```

**Response:**
```json
{
  "type": "object",
  "properties": {
    "min_samples": {
      "type": "integer",
      "default": 3,
      "title": "Min Samples"
    },
    "model_type": {
      "type": "string",
      "default": "linear_regression",
      "title": "Model Type"
    }
  },
  "title": "DiseaseConfig"
}
```

### 5. Create Configuration

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "shell_disease_model",
    "data": {
      "min_samples": 3,
      "model_type": "linear_regression"
    }
  }'
```

**Response:**
```json
{
  "id": "01JAABC123XYZ456...",
  "name": "shell_disease_model",
  "data": {
    "min_samples": 3,
    "model_type": "linear_regression"
  },
  "created_at": "2025-10-11T17:30:00Z"
}
```

### 6. Train Model (Executes External Script)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "01JAABC123XYZ456...",
    "data": {
      "columns": ["rainfall", "mean_temperature", "disease_cases"],
      "data": [
        [100.5, 25.3, 12],
        [85.2, 27.1, 8],
        [120.8, 24.5, 15],
        [95.3, 26.2, 10],
        [110.0, 26.0, 13]
      ]
    }
  }'
```

**What happens:**
1. Service creates temp directory: `/tmp/chapkit_ml_train_XXXXX/`
2. Writes `config.json` with model config
3. Writes `data.csv` with training data
4. Executes: `python examples/scripts/train_model.py --config config.json --data data.csv --model model.pickle`
5. Script trains model and saves to `model.pickle`
6. Service reads pickle and stores in artifact
7. Cleanup temp directory

**Response:**
```json
{
  "job_id": "01JAABC789DEF012...",
  "model_artifact_id": "01JAABC789GHI345...",
  "status": "pending",
  "message": "Training job submitted"
}
```

### 7. Check Training Logs

Server logs show script execution:

```
executing_train_script command='python .../train_model.py ...' temp_dir='/tmp/chapkit_ml_train_XXXXX'
train_script_completed stdout='Training model...\nModel saved to model.pickle\n' stderr=''
```

### 8. Get Trained Model

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/01JAABC789GHI345...
```

**Response:**
```json
{
  "id": "01JAABC789GHI345...",
  "level": 0,
  "data": {
    "model_type": "sklearn.linear_model.LinearRegression",
    "trained_via": "shell_script",
    "script": "train_model.py",
    "trained_at": "2025-10-11T17:31:05Z"
  }
}
```

### 9. Make Predictions (Executes External Script)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_artifact_id": "01JAABC789GHI345...",
    "historic": {
      "columns": ["rainfall", "mean_temperature"],
      "data": []
    },
    "future": {
      "columns": ["rainfall", "mean_temperature"],
      "data": [
        [110.0, 26.0],
        [90.0, 28.0],
        [125.0, 24.0]
      ]
    }
  }'
```

**What happens:**
1. Service creates temp directory: `/tmp/chapkit_ml_predict_XXXXX/`
2. Writes `config.json` with model config
3. Writes `model.pickle` (from artifact)
4. Writes `future.csv` with prediction data
5. Executes: `python examples/scripts/predict_model.py --config config.json --model model.pickle --future future.csv --output predictions.csv`
6. Script loads model and makes predictions
7. Service reads `predictions.csv` and stores in artifact
8. Cleanup temp directory

**Response:**
```json
{
  "job_id": "01JAABC999JKL678...",
  "prediction_artifact_id": "01JAABC999MNO901...",
  "status": "pending",
  "message": "Prediction job submitted"
}
```

### 10. Get Predictions

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/01JAABC999MNO901...
```

**Response:**
```json
{
  "id": "01JAABC999MNO901...",
  "parent_id": "01JAABC789GHI345...",
  "level": 1,
  "data": {
    "predictions": {
      "columns": ["rainfall", "mean_temperature", "sample_0"],
      "data": [
        [110.0, 26.0, 13.2],
        [90.0, 28.0, 8.5],
        [125.0, 24.0, 16.8]
      ]
    },
    "predicted_via": "shell_script",
    "script": "predict_model.py"
  }
}
```

## External Script Interface

### Training Script Contract

**Input files:**
- `{config_file}` - JSON with config data
- `{data_file}` - CSV with training data
- `{geo_file}` - GeoJSON (optional, empty string if not provided)

**Output files:**
- `{model_file}` - Pickled model (must create this file)

**Example: train_model.py**
```python
#!/usr/bin/env python3
import argparse
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

# Load config
with open(args.config) as f:
    config = json.load(f)

# Load training data
data = pd.read_csv(args.data)

# Train model
X = data[["rainfall", "mean_temperature"]]
y = data["disease_cases"]
model = LinearRegression()
model.fit(X, y)

# Save model
with open(args.model, "wb") as f:
    pickle.dump(model, f)

print("Training completed")
```

### Prediction Script Contract

**Input files:**
- `{config_file}` - JSON with config data
- `{model_file}` - Pickled model
- `{historic_file}` - CSV with historic data (optional, empty string if not provided)
- `{future_file}` - CSV with future data
- `{geo_file}` - GeoJSON (optional, empty string if not provided)

**Output files:**
- `{output_file}` - CSV with predictions (must create this file)

**Example: predict_model.py**
```python
#!/usr/bin/env python3
import argparse
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--future", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Load model
with open(args.model, "rb") as f:
    model = pickle.load(f)

# Load future data
future = pd.read_csv(args.future)

# Make predictions
X = future[["rainfall", "mean_temperature"]]
future["sample_0"] = model.predict(X)

# Save predictions
future.to_csv(args.output, index=False)

print("Predictions completed")
```

## Language-Agnostic Examples

### R Script

**train_model.R:**
```r
#!/usr/bin/env Rscript
library(jsonlite)

args <- commandArgs(trailingOnly=TRUE)
config <- fromJSON(args[1])
data <- read.csv(args[2])

model <- lm(disease_cases ~ rainfall + mean_temperature, data=data)
saveRDS(model, args[3])

cat("Training completed\n")
```

**Command template:**
```python
train_command = "Rscript {script_dir}/train_model.R {config_file} {data_file} {model_file}"
```

### Julia Script

**train_model.jl:**
```julia
#!/usr/bin/env julia
using JSON, CSV, DataFrames, Serialization, GLM

config = JSON.parsefile(ARGS[1])
data = CSV.read(ARGS[2], DataFrame)

model = lm(@formula(disease_cases ~ rainfall + mean_temperature), data)
serialize(ARGS[3], model)

println("Training completed")
```

**Command template:**
```python
train_command = "julia {script_dir}/train_model.jl {config_file} {data_file} {model_file}"
```

### Docker Container

```python
train_command = (
    "docker run --rm -v {temp_dir}:/data "
    "my-ml-image:latest "
    "train --config /data/config.json --data /data/data.csv --output /data/model.pickle"
)
```

## Key Advantages

### 1. Language Independence
Use any tool that can:
- Read CSV/JSON files
- Write pickle/CSV files
- Execute via command line

### 2. Easy Integration
Integrate existing scripts without modification - just wrap with command template

### 3. Container Support
Run models in isolated containers with dependencies

### 4. Team Flexibility
Different team members can use their preferred languages:
- Data Scientists: Python/R
- ML Engineers: Julia/Scala
- Research: Specialized tools

### 5. Tool Diversity
Mix and match:
- sklearn, TensorFlow, PyTorch (Python)
- caret, ranger (R)
- MLJ.jl (Julia)
- XGBoost CLI, LightGBM CLI
- Custom C++/Rust binaries

## Configuration Options

### Model Format

```python
# Default: pickle
runner = ShellModelRunner(
    train_command="...",
    predict_command="...",
    model_format="pickle"  # or "joblib"
)
```

### Script Parameters

Config fields are accessible in scripts:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs -d '{
  "name": "advanced",
  "data": {
    "min_samples": 10,
    "model_type": "random_forest",
    "n_estimators": 100,
    "max_depth": 5
  }
}'
```

Scripts can read these:
```python
config = json.load(open(args.config))
model_type = config["model_type"]
n_estimators = config.get("n_estimators", 100)
```

## Sample Datasets

### Training Data (5 samples)

```json
{
  "columns": ["rainfall", "mean_temperature", "disease_cases"],
  "data": [
    [100, 25, 10],
    [120, 24, 15],
    [80, 27, 5],
    [110, 26, 12],
    [90, 28, 8]
  ]
}
```

### Prediction Data

```json
{
  "columns": ["rainfall", "mean_temperature"],
  "data": [
    [110, 26],
    [95, 27],
    [125, 23]
  ]
}
```

## Debugging

### Check Script Execution

```bash
# Enable logging to see script output
fastapi dev examples/ml_shell.py

# Logs show:
# executing_train_script command='python ...' temp_dir='/tmp/...'
# train_script_completed stdout='...' stderr='...'
```

### Test Script Manually

```bash
# Create test files
echo '{"min_samples": 3}' > config.json
echo 'rainfall,mean_temperature,disease_cases
100,25,10
120,24,15' > data.csv

# Run training script
python examples/scripts/train_model.py \
  --config config.json \
  --data data.csv \
  --model model.pickle

# Check output
ls -lh model.pickle
```

### Script Failures

If script fails, check job result:

```bash
curl http://127.0.0.1:8000/api/v1/jobs/YOUR_JOB_ID
```

**Error Response:**
```json
{
  "id": "...",
  "status": "failed",
  "result": "Training script failed with exit code 1: ModuleNotFoundError: No module named 'sklearn'",
  "created_at": "...",
  "updated_at": "..."
}
```

## Performance Considerations

### Temp File Overhead
- Creates/deletes temp directory per operation
- Use for: External tools, containers, language barriers
- Avoid for: Pure Python workflows (use FunctionalModelRunner or BaseModelRunner)

### Process Spawning
- Each train/predict spawns subprocess
- Use for: CPU-bound operations, isolated environments
- Consider: Thread pool for concurrent operations

### Serialization
- Model must be pickleable (or use joblib)
- CSV overhead for large datasets
- Consider: Parquet for large data, protocol buffers for efficiency

## Tips

1. **Script Permissions**: Ensure scripts are executable (`chmod +x`)
2. **Dependencies**: Scripts must have all required libraries
3. **Error Handling**: Scripts should exit with non-zero on failure
4. **Output Required**: Must create model/prediction files
5. **Temp Cleanup**: Automatic - don't rely on temp files persisting
6. **Testing**: Test scripts independently before integration

## Troubleshooting

### "Training script failed with exit code 1"
- Check script has required dependencies
- Test script manually with sample data
- Check stderr in job result

### "Training script did not create model file"
- Verify script writes to `{model_file}` path
- Check script exit code (must be 0)
- Check file permissions

### "ModuleNotFoundError"
- Install required packages: `pip install scikit-learn pandas`
- Or use virtual environment in command template

### "Permission denied"
- Make scripts executable: `chmod +x examples/scripts/*.py`
- Or use `python script.py` instead of `./script.py`

## Next Steps

- Compare with **[ml_basic.md](ml_basic.md)** (pure Python functional)
- Compare with **[ml_class.md](ml_class.md)** (pure Python class-based)
- Check **[../scripts/train_model.py](../scripts/train_model.py)** for script implementation
- Try with R or Julia scripts
- Integrate with Docker containers
