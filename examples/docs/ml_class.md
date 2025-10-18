# ml_class.py - Class-Based ML Service cURL Guide

Weather-based prediction ML service using class-based model runner with feature preprocessing and StandardScaler.

## Quick Start

```bash
# Start the service
fastapi dev examples/ml_class.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Model**: LinearRegression with StandardScaler preprocessing
- **Features**: `rainfall`, `mean_temperature`, `humidity` (3 features vs 2 in basic)
- **Target**: `disease_cases`
- **Runner**: WeatherModelRunner (class-based with lifecycle hooks)
- **Assessment**: Green (validated for production)
- **Special**: Feature normalization with shared scaler state between train/predict

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
  "display_name": "Weather-Based Prediction Service",
  "version": "1.0.0",
  "summary": "Class-based ML service with preprocessing",
  "description": "Train and predict disease cases using normalized weather features with StandardScaler",
  "author": "Data Science Team",
  "author_note": "Improved feature normalization for better prediction accuracy",
  "author_assessed_status": "green",
  "contact_email": "datascience@example.com"
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
      "default": 5,
      "title": "Min Samples"
    },
    "normalize_features": {
      "type": "boolean",
      "default": true,
      "title": "Normalize Features"
    }
  },
  "title": "WeatherConfig"
}
```

### 5. Create Configuration

```bash
# With normalization (recommended)
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "weather_model_normalized",
    "data": {
      "min_samples": 5,
      "normalize_features": true
    }
  }'

# Without normalization (for comparison)
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "weather_model_raw",
    "data": {
      "min_samples": 3,
      "normalize_features": false
    }
  }'
```

**Response:**
```json
{
  "id": "01JAABC123XYZ456...",
  "name": "weather_model_normalized",
  "data": {
    "min_samples": 5,
    "normalize_features": true
  },
  "created_at": "2025-10-11T17:30:00Z",
  "updated_at": "2025-10-11T17:30:00Z"
}
```

### 6. Train Model with Preprocessing

**Important**: Training data must include all 3 features: `rainfall`, `mean_temperature`, `humidity`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "01JAABC123XYZ456...",
    "data": {
      "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
      "data": [
        [100.5, 25.3, 65.2, 12],
        [85.2, 27.1, 58.9, 8],
        [120.8, 24.5, 72.1, 15],
        [95.3, 26.2, 61.5, 10],
        [110.0, 26.0, 68.3, 13],
        [75.5, 28.0, 55.7, 6],
        [130.2, 23.8, 75.9, 18],
        [88.9, 27.5, 60.2, 9]
      ]
    }
  }'
```

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

If logging is enabled, check server logs for detailed training info:

```
runner_initializing features=['rainfall', 'mean_temperature', 'humidity']
training_started config={'min_samples': 5, 'normalize_features': True} sample_count=8
features_normalized mean=[100.675, 26.05, 64.725] scale=[16.789, 1.312, 6.458]
model_trained features=['rainfall', 'mean_temperature', 'humidity'] coefficients=[0.45, -0.23, 0.12]
runner_cleanup
```

### 8. Get Model Artifact (with Preprocessing State)

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
    "scaler": "StandardScaler",
    "feature_names": ["rainfall", "mean_temperature", "humidity"],
    "config": {
      "min_samples": 5,
      "normalize_features": true
    },
    "trained_at": "2025-10-11T17:31:05Z"
  },
  "created_at": "2025-10-11T17:31:00Z"
}
```

Note the `scaler` and `feature_names` stored with the model!

### 9. Make Predictions (Auto-Applies Same Preprocessing)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_artifact_id": "01JAABC789GHI345...",
    "historic": {
      "columns": ["rainfall", "mean_temperature", "humidity"],
      "data": []
    },
    "future": {
      "columns": ["rainfall", "mean_temperature", "humidity"],
      "data": [
        [110.0, 26.0, 67.0],
        [90.0, 28.0, 58.0],
        [125.0, 24.0, 74.0],
        [80.0, 29.0, 52.0]
      ]
    }
  }'
```

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
      "columns": ["rainfall", "mean_temperature", "humidity", "sample_0"],
      "data": [
        [110.0, 26.0, 67.0, 13.5],
        [90.0, 28.0, 58.0, 8.2],
        [125.0, 24.0, 74.0, 17.3],
        [80.0, 29.0, 52.0, 5.9]
      ]
    }
  }
}
```

## Key Differences from ml_basic

### 1. Additional Feature
- ml_basic: 2 features (`rainfall`, `mean_temperature`)
- ml_class: 3 features (`rainfall`, `mean_temperature`, `humidity`)

### 2. Feature Normalization
- ml_basic: Raw values used directly
- ml_class: StandardScaler normalization (configurable)

### 3. Model State
- ml_basic: Only model saved
- ml_class: Model + scaler + feature names saved together

### 4. Configuration
- ml_basic: Empty config
- ml_class: `min_samples` and `normalize_features` parameters

### 5. Implementation Style
- ml_basic: Functional (simple functions)
- ml_class: Class-based (lifecycle hooks, shared state)

## Configuration Options

### Min Samples Validation

```bash
# Will fail if < 5 samples with default config
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "YOUR_CONFIG_ID",
    "data": {
      "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
      "data": [
        [100, 25, 65, 10],
        [120, 24, 72, 15]
      ]
    }
  }'
```

**Error Response:**
```json
{
  "type": "urn:chapkit:error:validation-failed",
  "title": "Validation Error",
  "status": 400,
  "detail": "Insufficient training data: 2 < 5"
}
```

### Normalize vs Raw Features

```bash
# Create two configs for comparison
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name": "normalized", "data": {"normalize_features": true}}'

curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name": "raw", "data": {"normalize_features": false}}'

# Train both and compare prediction accuracy
```

## Sample Datasets

### Minimal (5 samples - meets min_samples requirement)

```json
{
  "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
  "data": [
    [100, 25, 65, 10],
    [120, 24, 72, 15],
    [80, 27, 58, 5],
    [110, 26, 68, 12],
    [90, 28, 60, 8]
  ]
}
```

### Realistic (12 samples with seasonal variation)

```json
{
  "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
  "data": [
    [95.5, 28.2, 62.3, 8],
    [112.3, 26.7, 68.5, 14],
    [78.9, 29.5, 55.2, 4],
    [105.2, 27.1, 64.8, 11],
    [88.4, 30.0, 58.9, 6],
    [125.8, 25.3, 73.1, 18],
    [98.7, 28.9, 61.7, 9],
    [115.6, 26.0, 70.2, 15],
    [82.1, 29.8, 56.5, 5],
    [108.3, 27.5, 66.4, 12],
    [92.5, 28.6, 60.8, 7],
    [118.9, 25.8, 71.9, 16]
  ]
}
```

### With Outliers (tests robustness)

```json
{
  "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
  "data": [
    [100, 25, 65, 10],
    [120, 24, 72, 15],
    [80, 27, 58, 5],
    [250, 22, 95, 35],
    [110, 26, 68, 12],
    [15, 32, 30, 1],
    [90, 28, 60, 8]
  ]
}
```

## Advanced Features

### Lifecycle Hooks

The class-based runner uses lifecycle hooks:

```python
async def on_init(self) -> None:
    # Called before train/predict
    # Setup resources, logging, etc.

async def on_cleanup(self) -> None:
    # Called after train/predict
    # Clean up resources
```

Check logs for:
```
runner_initializing features=['rainfall', 'mean_temperature', 'humidity']
...
runner_cleanup
```

### Shared State Between Train/Predict

The runner maintains state:
- `feature_names`: Ensures prediction uses same features
- `scaler`: Applies same normalization to new data
- Configuration is validated and logged

## Comparison Experiment

Compare normalized vs raw features:

```bash
# 1. Train with normalization
curl -X POST http://127.0.0.1:8000/api/v1/configs -d '{"name":"norm","data":{"normalize_features":true}}'
# Train and save model_artifact_id as MODEL_NORM

# 2. Train without normalization
curl -X POST http://127.0.0.1:8000/api/v1/configs -d '{"name":"raw","data":{"normalize_features":false}}'
# Train and save model_artifact_id as MODEL_RAW

# 3. Predict with same test data using both models
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict -d '{"model_artifact_id":"MODEL_NORM","historic":{...},"future":{...}}'
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict -d '{"model_artifact_id":"MODEL_RAW","historic":{...},"future":{...}}'

# 4. Compare prediction accuracy
```

## Tips

1. **Always Include 3 Features**: `rainfall`, `mean_temperature`, `humidity` required
2. **Normalization Recommended**: Generally improves model performance
3. **Min Samples**: Increase for production models (default: 5)
4. **Preprocessing Consistency**: Scaler from training automatically applied to predictions
5. **Green Status**: This example is marked production-ready
6. **Check Logs**: Detailed training metrics in server logs

## Troubleshooting

### "Insufficient training data"
Increase sample count or lower `min_samples` in config

### "Missing required columns"
Must include: `rainfall`, `mean_temperature`, `humidity`, `disease_cases` (training)
Must include: `rainfall`, `mean_temperature`, `humidity` (prediction)

### "Feature mismatch"
Prediction features must match training features (stored in artifact)

### Unexpected predictions
- Try with `normalize_features: true`
- Check for outliers in training data
- Increase training sample size

## Next Steps

- Compare with **[ml_basic.md](ml_basic.md)** (simpler, 2 features)
- Try **[ml_shell.md](ml_shell.md)** for external script integration
- Check **[../ml_class.py](../ml_class.py)** source code
- Experiment with different `min_samples` and `normalize_features` values
