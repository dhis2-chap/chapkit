# ML Workflows

Chapkit provides a complete ML workflow system for training models and making predictions with artifact-based model storage, job scheduling, and hierarchical model lineage tracking.

## Quick Start

### Functional Approach (Recommended for Simple Models)

```python
from chapkit.artifact import ArtifactHierarchy

from chapkit import BaseConfig
from chapkit.api import MLServiceBuilder, MLServiceInfo
from chapkit.ml import FunctionalModelRunner
import pandas as pd
from sklearn.linear_model import LinearRegression

class ModelConfig(BaseConfig):
    pass

async def on_train(config: ModelConfig, data: pd.DataFrame, geo=None):
    X = data[["feature1", "feature2"]]
    y = data["target"]
    model = LinearRegression()
    model.fit(X, y)
    return model

async def on_predict(config: ModelConfig, model, historic, future, geo=None):
    X = future[["feature1", "feature2"]]
    future["sample_0"] = model.predict(X)
    return future

runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)

app = (
    MLServiceBuilder(
        info=MLServiceInfo(display_name="My ML Service"),
        config_schema=ModelConfig,
        hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "ml_training", 1: "ml_prediction"}),
        runner=runner,
    )
    .build()
)
```

**Run:** `fastapi dev your_file.py`

### Class-Based Approach (Recommended for Complex Models)

```python
from chapkit.ml import BaseModelRunner
from sklearn.preprocessing import StandardScaler

class CustomModelRunner(BaseModelRunner[ModelConfig]):
    def __init__(self):
        self.scaler = StandardScaler()

    async def on_train(self, config: ModelConfig, data, geo=None):
        X = data[["feature1", "feature2"]]
        y = data["target"]

        X_scaled = self.scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)

        return {"model": model, "scaler": self.scaler}

    async def on_predict(self, config: ModelConfig, model, historic, future, geo=None):
        X = future[["feature1", "feature2"]]
        X_scaled = model["scaler"].transform(X)
        future["sample_0"] = model["model"].predict(X_scaled)
        return future

runner = CustomModelRunner()
# Use same MLServiceBuilder setup as above
```

### Shell-Based Approach (Language-Agnostic)

```python
from chapkit.ml import ShellModelRunner
import sys

train_command = (
    f"{sys.executable} scripts/train.py "
    "--config {config_file} --data {data_file} --model {model_file}"
)

predict_command = (
    f"{sys.executable} scripts/predict.py "
    "--config {config_file} --model {model_file} "
    "--future {future_file} --output {output_file}"
)

runner = ShellModelRunner(
    train_command=train_command,
    predict_command=predict_command,
    model_format="pickle"
)
# Use same MLServiceBuilder setup as above
```

---

## Architecture

### Train/Predict Flow

```
1. TRAIN                           2. PREDICT
   POST /api/v1/ml/$train             POST /api/v1/ml/$predict
   ├─> Submit job                     ├─> Load trained model artifact
   ├─> Load config                    ├─> Load config
   ├─> Execute runner.on_train()      ├─> Execute runner.on_predict()
   └─> Store model in artifact        └─> Store predictions in artifact
       (level 0, parent_id=None)           (level 1, parent_id=model_id)
```

### Artifact Hierarchy

```
Config
  └─> Trained Model (level 0)
       ├─> Predictions 1 (level 1)
       ├─> Predictions 2 (level 1)
       └─> Predictions 3 (level 1)
```

**Benefits:**
- Complete model lineage tracking
- Multiple predictions from same model
- Config linked to all model artifacts
- Immutable model versioning

### Job Scheduling

All train/predict operations are asynchronous:
- Submit returns immediately with `job_id` and `artifact_id`
- Monitor progress via Job API or SSE streaming
- Results stored in artifacts when complete

---

## Model Runners

### BaseModelRunner

Abstract base class for custom model runners with lifecycle hooks.

```python
from chapkit.ml import BaseModelRunner
from chapkit import BaseConfig

class MyConfig(BaseConfig):
    """Your config schema."""
    pass

class MyRunner(BaseModelRunner[MyConfig]):
    async def on_init(self):
        """Called before train or predict (optional)."""
        pass

    async def on_cleanup(self):
        """Called after train or predict (optional)."""
        pass

    async def on_train(self, config: MyConfig, data, geo=None):
        """Train and return model (must be pickleable)."""
        # config is typed as MyConfig - no casting needed!
        # Your training logic
        return trained_model

    async def on_predict(self, config: MyConfig, model, historic, future, geo=None):
        """Make predictions and return DataFrame."""
        # config is typed as MyConfig - autocomplete works!
        # Your prediction logic
        return predictions_df
```

**Key Points:**
- Model must be pickleable (stored in artifact)
- Return value from `on_train` is passed to `on_predict` as `model` parameter
- `historic` parameter is required (must be provided, can be empty DataFrame)
- GeoJSON support via `geo` parameter

### FunctionalModelRunner

Wraps train/predict functions for functional-style ML workflows.

```python
from chapkit.ml import FunctionalModelRunner

async def train_fn(config, data, geo=None):
    # Training logic
    return model

async def predict_fn(config, model, historic, future, geo=None):
    # Prediction logic
    return predictions

runner = FunctionalModelRunner(on_train=train_fn, on_predict=predict_fn)
```

**Use Cases:**
- Simple models without state
- Quick prototypes
- Pure function workflows

### ShellModelRunner

Executes external scripts for language-agnostic ML workflows.

```python
from chapkit.ml import ShellModelRunner

runner = ShellModelRunner(
    train_command="python train.py --config {config_file} --data {data_file} --model {model_file}",
    predict_command="python predict.py --config {config_file} --model {model_file} --future {future_file} --output {output_file}",
    model_format="pickle"  # or "joblib", "json", etc.
)
```

**Variable Substitution:**
- `{config_file}` - JSON config file
- `{data_file}` - Training data CSV
- `{model_file}` - Model file (format specified)
- `{future_file}` - Future data CSV
- `{historic_file}` - Historic data CSV (required)
- `{output_file}` - Predictions output CSV
- `{geo_file}` - GeoJSON file (if provided)

**Script Requirements:**
- **Training script:** Read data/config, train model, save model to `{model_file}`
- **Prediction script:** Read model/data/config, make predictions, save to `{output_file}`
- Exit code 0 on success, non-zero on failure
- Use stderr for logging

**Example Training Script (Python):**
```python
#!/usr/bin/env python3
import argparse, json, pickle
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

# Load data
data = pd.read_csv(args.data)

# Train
X = data[["feature1", "feature2"]]
y = data["target"]
model = LinearRegression()
model.fit(X, y)

# Save
with open(args.model, "wb") as f:
    pickle.dump(model, f)
```

**Use Cases:**
- Integration with R, Julia, or other languages
- Legacy scripts without modification
- Containerized ML pipelines
- Team collaboration across languages

---

## ServiceBuilder Setup

### MLServiceBuilder (Recommended)

Bundles health, config, artifacts, jobs, and ML in one builder.

```python
from chapkit.artifact import ArtifactHierarchy

from chapkit.api import MLServiceBuilder, MLServiceInfo, AssessedStatus

info = MLServiceInfo(
    display_name="Disease Prediction Service",
    version="1.0.0",
    summary="ML service for disease prediction",
    description="Train and predict disease cases using weather data",
    author="ML Team",
    author_assessed_status=AssessedStatus.green,
    contact_email="ml-team@example.com",
)

hierarchy = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={0: "ml_training", 1: "ml_prediction"},
)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=ModelConfig,
        hierarchy=hierarchy,
        runner=runner,
    )
    .with_monitoring()  # Optional: Prometheus metrics
    .build()
)
```

**MLServiceBuilder automatically includes:**
- Health check (`/health`)
- Config CRUD (`/api/v1/configs`)
- Artifact CRUD (`/api/v1/artifacts`)
- Job scheduler (`/api/v1/jobs`) with concurrency control
- ML endpoints (`/api/v1/ml/$train`, `/api/v1/ml/$predict`)

### ServiceBuilder (Manual Configuration)

For fine-grained control:

```python
from chapkit.api import ServiceBuilder, ServiceInfo

app = (
    ServiceBuilder(info=ServiceInfo(display_name="Custom ML Service"))
    .with_health()
    .with_config(ModelConfig)
    .with_artifacts(hierarchy=hierarchy)
    .with_jobs(max_concurrency=3)
    .with_ml(runner=runner)
    .build()
)
```

**Requirements:**
- `.with_config()` must be called before `.with_ml()`
- `.with_artifacts()` must be called before `.with_ml()`
- `.with_jobs()` must be called before `.with_ml()`

### Configuration Options

```python
MLServiceBuilder(
    info=info,
    config_schema=YourConfig,
    hierarchy=hierarchy,
    runner=runner,
    max_concurrency=5,       # Limit concurrent jobs (default: unlimited)
    database_url="ml.db",    # Persistent storage (default: in-memory)
)
```

---

## API Reference

### POST /api/v1/ml/$train

Train a model asynchronously.

**Request:**
```json
{
  "config_id": "01JCONFIG...",
  "data": {
    "columns": ["feature1", "feature2", "target"],
    "data": [
      [1.0, 2.0, 10.0],
      [2.0, 3.0, 15.0],
      [3.0, 4.0, 20.0]
    ]
  },
  "geo": null
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "01JOB123...",
  "artifact_id": "01MODEL456...",
  "message": "Training job submitted. Job ID: 01JOB123..."
}
```

**cURL Example:**
```bash
# Create config first
CONFIG_ID=$(curl -s -X POST http://localhost:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name": "my_config", "data": {}}' | jq -r '.id')

# Submit training job
curl -X POST http://localhost:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "'$CONFIG_ID'",
    "data": {
      "columns": ["rainfall", "temperature", "cases"],
      "data": [[10.0, 25.0, 5.0], [15.0, 28.0, 8.0]]
    }
  }' | jq
```

### POST /api/v1/ml/$predict

Make predictions using a trained model.

**Request:**
```json
{
  "training_artifact_id": "01MODEL456...",
  "historic": {
    "columns": ["feature1", "feature2"],
    "data": []
  },
  "future": {
    "columns": ["feature1", "feature2"],
    "data": [
      [1.5, 2.5],
      [2.5, 3.5]
    ]
  },
  "geo": null
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "01JOB789...",
  "artifact_id": "01PRED012...",
  "message": "Prediction job submitted. Job ID: 01JOB789..."
}
```

**cURL Example:**
```bash
# Use model from training
curl -X POST http://localhost:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "training_artifact_id": "'$MODEL_ARTIFACT_ID'",
    "historic": {
      "columns": ["rainfall", "temperature"],
      "data": []
    },
    "future": {
      "columns": ["rainfall", "temperature"],
      "data": [[12.0, 26.0], [18.0, 29.0]]
    }
  }' | jq
```

### Monitor Job Status

```bash
# Poll job status
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq

# Stream status updates (SSE)
curl -N http://localhost:8000/api/v1/jobs/$JOB_ID/\$stream

# Get results from artifact
ARTIFACT_ID=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.artifact_id')
curl http://localhost:8000/api/v1/artifacts/$ARTIFACT_ID | jq
```

---

## Data Formats

### DataFrame Schema

All tabular data uses the `DataFrame` schema:

```json
{
  "columns": ["col1", "col2", "col3"],
  "data": [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ],
  "index": null,
  "column_types": null
}
```

**Python Usage:**
```python
from servicekit.data import DataFrame

# Create from DataFrame
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
data_frame = DataFrame.from_pandas(df)

# Convert to DataFrame
df = data_frame.to_pandas()
```

### GeoJSON Support

Optional geospatial data via `geojson-pydantic`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.4194, 37.7749]
      },
      "properties": {
        "name": "San Francisco",
        "population": 883305
      }
    }
  ]
}
```

---

## Artifact Structure

### TrainedModelArtifactData

Stored at hierarchy level 0:

```json
{
  "ml_type": "ml_training",
  "config_id": "01CONFIG...",
  "model": "<pickled model object>",
  "model_type": "sklearn.linear_model.LinearRegression",
  "model_size_bytes": 1234,
  "started_at": "2025-10-14T10:00:00Z",
  "completed_at": "2025-10-14T10:00:15Z",
  "duration_seconds": 15.23
}
```

**Fields:**
- `ml_type`: Always `"ml_training"`
- `config_id`: Config used for training
- `model`: Pickled model object (any Python object)
- `model_type`: Fully qualified class name (e.g., `sklearn.linear_model.LinearRegression`)
- `model_size_bytes`: Serialized pickle size
- `started_at`, `completed_at`: ISO timestamps
- `duration_seconds`: Training duration (rounded to 2 decimals)

### PredictionArtifactData

Stored at hierarchy level 1 (linked to model):

```json
{
  "ml_type": "ml_prediction",
  "config_id": "01CONFIG...",
  "training_artifact_id": "01MODEL...",
  "predictions": {
    "columns": ["feature1", "feature2", "sample_0"],
    "data": [[1.5, 2.5, 12.3], [2.5, 3.5, 17.8]]
  },
  "started_at": "2025-10-14T10:05:00Z",
  "completed_at": "2025-10-14T10:05:02Z",
  "duration_seconds": 2.15
}
```

**Fields:**
- `ml_type`: Always `"ml_prediction"`
- `config_id`: Config used for prediction
- `training_artifact_id`: Parent trained model artifact
- `predictions`: Result DataFrame (DataFrame schema)
- `started_at`, `completed_at`: ISO timestamps
- `duration_seconds`: Prediction duration (rounded to 2 decimals)

---

## Complete Workflow Examples

### Basic Functional Workflow

```bash
# 1. Start service
fastapi dev examples/ml_basic.py

# 2. Create config
CONFIG_ID=$(curl -s -X POST http://localhost:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name": "weather_model", "data": {}}' | jq -r '.id')

# 3. Train model
TRAIN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "'$CONFIG_ID'",
    "data": {
      "columns": ["rainfall", "mean_temperature", "disease_cases"],
      "data": [
        [10.0, 25.0, 5.0],
        [15.0, 28.0, 8.0],
        [8.0, 22.0, 3.0],
        [20.0, 30.0, 12.0],
        [12.0, 26.0, 6.0]
      ]
    }
  }')

JOB_ID=$(echo $TRAIN_RESPONSE | jq -r '.job_id')
MODEL_ARTIFACT_ID=$(echo $TRAIN_RESPONSE | jq -r '.artifact_id')

echo "Training Job ID: $JOB_ID"
echo "Model Artifact ID: $MODEL_ARTIFACT_ID"

# 4. Wait for training completion
curl -N http://localhost:8000/api/v1/jobs/$JOB_ID/\$stream

# 5. View trained model
curl http://localhost:8000/api/v1/artifacts/$MODEL_ARTIFACT_ID | jq

# 6. Make predictions
PREDICT_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "training_artifact_id": "'$MODEL_ARTIFACT_ID'",
    "historic": {
      "columns": ["rainfall", "mean_temperature"],
      "data": []
    },
    "future": {
      "columns": ["rainfall", "mean_temperature"],
      "data": [
        [11.0, 26.0],
        [14.0, 27.0],
        [9.0, 24.0]
      ]
    }
  }')

PRED_JOB_ID=$(echo $PREDICT_RESPONSE | jq -r '.job_id')
PRED_ARTIFACT_ID=$(echo $PREDICT_RESPONSE | jq -r '.artifact_id')

# 7. Wait for predictions
curl -N http://localhost:8000/api/v1/jobs/$PRED_JOB_ID/\$stream

# 8. View predictions
curl http://localhost:8000/api/v1/artifacts/$PRED_ARTIFACT_ID | jq '.data.predictions'
```

### Class-Based with Preprocessing

```python
# examples/ml_class.py demonstrates:
# - StandardScaler for feature normalization
# - State management (scaler shared between train/predict)
# - Lifecycle hooks (on_init, on_cleanup)
# - Model artifact containing multiple objects

from chapkit.ml import BaseModelRunner
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class WeatherModelRunner(BaseModelRunner[WeatherConfig]):
    def __init__(self):
        self.feature_names = ["rainfall", "mean_temperature", "humidity"]
        self.scaler = None

    async def on_train(self, config: WeatherConfig, data, geo=None):
        X = data[self.feature_names].fillna(0)
        y = data["disease_cases"].fillna(0)

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Return dict with model and preprocessing artifacts
        return {
            "model": model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }

    async def on_predict(self, config: WeatherConfig, model, historic, future, geo=None):
        # Extract artifacts
        trained_model = model["model"]
        scaler = model["scaler"]
        feature_names = model["feature_names"]

        # Apply same preprocessing
        X = future[feature_names].fillna(0)
        X_scaled = scaler.transform(X)

        # Predict
        future["sample_0"] = trained_model.predict(X_scaled)
        return future
```

**Benefits:**
- Consistent preprocessing between train/predict
- Model artifacts include all necessary objects
- Type safety and validation
- Easy testing and debugging

### Shell-Based Language-Agnostic

```python
# examples/ml_shell.py demonstrates:
# - External R/Julia/Python scripts
# - File-based data interchange
# - No code modification required
# - Container-friendly workflows

from chapkit.ml import ShellModelRunner
import sys

runner = ShellModelRunner(
    train_command=f"{sys.executable} scripts/train_model.py --config {{config_file}} --data {{data_file}} --model {{model_file}}",
    predict_command=f"{sys.executable} scripts/predict_model.py --config {{config_file}} --model {{model_file}} --future {{future_file}} --output {{output_file}}",
    model_format="pickle"
)
```

**External Script Example (R):**
```r
#!/usr/bin/env Rscript
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
config_file <- args[which(args == "--config") + 1]
data_file <- args[which(args == "--data") + 1]
model_file <- args[which(args == "--model") + 1]

# Load data
config <- fromJSON(config_file)
data <- read.csv(data_file)

# Train model
model <- lm(disease_cases ~ rainfall + mean_temperature, data = data)

# Save model
saveRDS(model, model_file)
cat("SUCCESS: Model trained\n")
```

---

## Testing

### Manual Testing

**Terminal 1:**
```bash
fastapi dev examples/ml_basic.py
```

**Terminal 2:**
```bash
# Complete workflow test
CONFIG_ID=$(curl -s -X POST http://localhost:8000/api/v1/configs \
  -d '{"name":"test","data":{}}' | jq -r '.id')

TRAIN=$(curl -s -X POST http://localhost:8000/api/v1/ml/\$train -d '{
  "config_id":"'$CONFIG_ID'",
  "data":{"columns":["a","b","y"],"data":[[1,2,10],[2,3,15],[3,4,20]]}
}')

MODEL_ID=$(echo $TRAIN | jq -r '.artifact_id')
JOB_ID=$(echo $TRAIN | jq -r '.job_id')

# Wait for completion
sleep 2
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.status'

# Predict
PRED=$(curl -s -X POST http://localhost:8000/api/v1/ml/\$predict -d '{
  "training_artifact_id":"'$MODEL_ID'",
  "historic":{"columns":["a","b"],"data":[]},
  "future":{"columns":["a","b"],"data":[[1.5,2.5],[2.5,3.5]]}
}')

PRED_ID=$(echo $PRED | jq -r '.artifact_id')
sleep 2

# View results
curl http://localhost:8000/api/v1/artifacts/$PRED_ID | jq '.data.predictions'
```

### Automated Testing

```python
import time
from fastapi.testclient import TestClient

def wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0):
    """Poll until job completes."""
    start = time.time()
    while time.time() - start < timeout:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ["completed", "failed", "canceled"]:
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} timeout")


def test_train_predict_workflow(client: TestClient):
    """Test complete ML workflow."""
    # Create config
    config_resp = client.post("/api/v1/configs", json={
        "name": "test_config",
        "data": {}
    })
    config_id = config_resp.json()["id"]

    # Train
    train_resp = client.post("/api/v1/ml/$train", json={
        "config_id": config_id,
        "data": {
            "columns": ["x1", "x2", "y"],
            "data": [[1, 2, 10], [2, 3, 15], [3, 4, 20]]
        }
    })
    assert train_resp.status_code == 202

    train_data = train_resp.json()
    job_id = train_data["job_id"]
    model_id = train_data["artifact_id"]

    # Wait for training
    job = wait_for_job(client, job_id)
    assert job["status"] == "completed"

    # Verify model artifact
    model_artifact = client.get(f"/api/v1/artifacts/{model_id}").json()
    assert model_artifact["data"]["ml_type"] == "ml_training"
    assert model_artifact["level"] == 0

    # Predict
    pred_resp = client.post("/api/v1/ml/$predict", json={
        "training_artifact_id": model_id,
        "historic": {
            "columns": ["x1", "x2"],
            "data": []
        },
        "future": {
            "columns": ["x1", "x2"],
            "data": [[1.5, 2.5], [2.5, 3.5]]
        }
    })
    assert pred_resp.status_code == 202

    pred_data = pred_resp.json()
    pred_job_id = pred_data["job_id"]
    pred_id = pred_data["artifact_id"]

    # Wait for prediction
    pred_job = wait_for_job(client, pred_job_id)
    assert pred_job["status"] == "completed"

    # Verify predictions
    pred_artifact = client.get(f"/api/v1/artifacts/{pred_id}").json()
    assert pred_artifact["data"]["ml_type"] == "ml_prediction"
    assert pred_artifact["parent_id"] == model_id
    assert pred_artifact["level"] == 1
    assert "sample_0" in pred_artifact["data"]["predictions"]["columns"]
```

### Browser Testing (Swagger UI)

1. Open http://localhost:8000/docs
2. Create config via POST `/api/v1/configs`
3. Train via POST `/api/v1/ml/$train`
4. Monitor job via GET `/api/v1/jobs/{job_id}`
5. Predict via POST `/api/v1/ml/$predict`
6. View artifacts via GET `/api/v1/artifacts/{artifact_id}`

---

## Production Deployment

### Concurrency Control

```python
MLServiceBuilder(
    info=info,
    config_schema=config_schema,
    hierarchy=hierarchy,
    runner=runner,
    max_concurrency=3,  # Limit concurrent training jobs
)
```

**Recommendations:**
- **CPU-intensive models**: Set to CPU core count (4-8)
- **GPU models**: Set to GPU count (1-4)
- **Memory-intensive**: Lower limits (2-3)
- **I/O-bound**: Higher limits OK (10-20)

### Database Configuration

```python
MLServiceBuilder(
    info=info,
    config_schema=config_schema,
    hierarchy=hierarchy,
    runner=runner,
    database_url="/data/ml.db",  # Persistent storage
)
```

**Best Practices:**
- Mount persistent volume for `/data`
- Regular backups (models + artifacts)
- Monitor database size growth
- Implement artifact retention policies

### Model Versioning

```python
# Use config name for version tracking
config = {
    "name": "weather_model_v1.2.3",
    "data": {
        "version": "1.2.3",
        "features": ["rainfall", "temperature"],
        "hyperparameters": {"alpha": 0.01}
    }
}
```

**Artifact Hierarchy for Versions:**
```
weather_model_v1.0.0 (config)
  └─> trained_model_1 (artifact level 0)
       └─> predictions_* (artifact level 1)

weather_model_v1.1.0 (config)
  └─> trained_model_2 (artifact level 0)
       └─> predictions_* (artifact level 1)
```

### Monitoring

```python
app = (
    MLServiceBuilder(info=info, config_schema=config, hierarchy=hierarchy, runner=runner)
    .with_monitoring()  # Prometheus metrics at /metrics
    .build()
)
```

**Available Metrics:**
- `ml_train_jobs_total` - Total training jobs submitted
- `ml_predict_jobs_total` - Total prediction jobs submitted
- Job scheduler metrics (see Job Scheduler guide)

**Custom Metrics:**
```python
from prometheus_client import Histogram

model_training_duration = Histogram(
    'model_training_duration_seconds',
    'Model training duration'
)

# Training durations already tracked in artifact metadata
# Query via artifact API
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8000

CMD ["fastapi", "run", "ml_service.py", "--host", "0.0.0.0"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  ml-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ml-data:/data
    environment:
      - DATABASE_URL=/data/ml.db
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G

volumes:
  ml-data:
```

### GPU Support

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04
FROM python:3.13

# Install ML libraries with GPU support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu120

# Your ML code
COPY . /app
```

**docker-compose.yml:**
```yaml
services:
  ml-service:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Troubleshooting

### "Config not found" Error

**Problem:** Training fails with "Config {id} not found"

**Cause:** Invalid or deleted config ID

**Solution:**
```bash
# List configs
curl http://localhost:8000/api/v1/configs | jq

# Verify config exists
curl http://localhost:8000/api/v1/configs/$CONFIG_ID
```

### "Model artifact not found" Error

**Problem:** Prediction fails with "Model artifact {id} not found"

**Cause:** Invalid model artifact ID or training failed

**Solution:**
```bash
# Check training job status
curl http://localhost:8000/api/v1/jobs/$TRAIN_JOB_ID | jq

# If training failed, check error
curl http://localhost:8000/api/v1/jobs/$TRAIN_JOB_ID | jq '.error'

# List artifacts
curl http://localhost:8000/api/v1/artifacts | jq
```

### Training Job Fails Immediately

**Problem:** Job status shows "failed" right after submission

**Causes:**
1. Model not pickleable
2. Missing required columns in data
3. Insufficient training data
4. Config validation errors

**Solution:**
```bash
# Check job error message
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.error, .error_traceback'

# Common fixes:
# - Ensure model is pickleable (no lambda functions, local classes)
# - Verify DataFrame columns match feature expectations
# - Check config schema validation
```

### Prediction Returns Wrong Shape

**Problem:** Predictions DataFrame has incorrect columns

**Cause:** `on_predict` must add prediction columns to input DataFrame

**Solution:**
```python
async def on_predict(self, config, model, historic, future, geo=None):
    X = future[["feature1", "feature2"]]
    predictions = model.predict(X)

    # IMPORTANT: Add predictions to future DataFrame
    future["sample_0"] = predictions  # Required column name

    return future  # Return modified DataFrame
```

### Shell Runner Script Fails

**Problem:** ShellModelRunner returns "script failed with exit code 1"

**Causes:**
1. Script not executable
2. Wrong interpreter
3. Missing dependencies
4. File path issues

**Solution:**
```bash
# Make script executable
chmod +x scripts/train_model.py

# Test script manually
python scripts/train_model.py \
  --config /tmp/test_config.json \
  --data /tmp/test_data.csv \
  --model /tmp/test_model.pkl

# Check script stderr output
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.error'
```

### High Memory Usage

**Problem:** Service consuming excessive memory

**Causes:**
1. Large models in memory
2. Too many concurrent jobs
3. Artifact accumulation

**Solution:**
```python
# Limit concurrent jobs
MLServiceBuilder(..., max_concurrency=2)

# Implement artifact cleanup
async def cleanup_old_artifacts(app):
    # Delete artifacts older than 30 days
    cutoff = datetime.now() - timedelta(days=30)
    # Implementation depends on your needs

app.on_startup(cleanup_old_artifacts)
```

### Model Size Too Large

**Problem:** "Model size exceeds limit" or slow artifact storage

**Cause:** Large models (>100MB) stored in SQLite

**Solution:**
```python
# Option 1: External model storage
async def on_train(self, config, data, geo=None):
    model = train_large_model(data)

    # Save to external storage (S3, etc.)
    model_url = save_to_s3(model)

    # Return metadata instead of model
    return {
        "model_url": model_url,
        "model_metadata": {...}
    }

# Option 2: Use PostgreSQL instead of SQLite
MLServiceBuilder(..., database_url="postgresql://...")
```

### DataFrame Validation Errors

**Problem:** "Invalid DataFrame schema" during train/predict

**Cause:** Incorrect data format in request

**Solution:**
```json
// Correct format
{
  "columns": ["col1", "col2"],
  "data": [
    [1.0, 2.0],
    [3.0, 4.0]
  ]
}

// Wrong formats:
// {"col1": [1, 3], "col2": [2, 4]}  (dict format - not supported)
// [{"col1": 1, "col2": 2}]  (records format - not supported)
```

---

## Next Steps

- **Job Monitoring:** See Job Scheduler guide for SSE streaming
- **Task Execution:** Combine with Tasks for preprocessing pipelines
- **Authentication:** Secure ML endpoints with API keys
- **Monitoring:** Track model performance with Prometheus metrics

For more examples:
- `examples/ml_basic.py` - Functional runner with LinearRegression
- `examples/ml_class.py` - Class-based runner with preprocessing
- `examples/ml_shell.py` - Shell-based runner with external scripts
- `tests/test_example_ml_basic.py` - Complete test suite
