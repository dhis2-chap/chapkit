# ml_basic.py - Basic ML Service cURL Guide

Disease prediction ML service using functional model runner with LinearRegression.

## Quick Start

```bash
# Start the service
fastapi dev examples/ml_basic.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Model**: LinearRegression from sklearn
- **Features**: `rainfall`, `mean_temperature`
- **Target**: `disease_cases`
- **Runner**: FunctionalModelRunner (functional programming style)
- **Assessment**: Yellow (ready for testing)

## Complete Workflow

### 1. Check Service Health

```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy"
  }
}
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
  "display_name": "Disease Prediction ML Service",
  "version": "1.0.0",
  "summary": "ML service for disease prediction using weather data",
  "description": "Train and predict disease cases based on rainfall and temperature data using Linear Regression",
  "author": "ML Team",
  "author_assessed_status": "yellow",
  "contact_email": "ml-team@example.com"
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
  "properties": {},
  "title": "DiseaseConfig"
}
```

*Note: DiseaseConfig is minimal - no extra parameters needed for this basic example*

### 5. Create Configuration

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
    "name": "basic_disease_model",
    "data": {}
  }'
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "name": "basic_disease_model",
  "data": {},
  "created_at": "2025-10-11T17:30:00Z",
  "updated_at": "2025-10-11T17:30:00Z"
}
```

### 6. Train Model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "01K79YAHJ7BR4E87VVTG8FNBMA",
    "data": {
      "columns": ["rainfall", "mean_temperature", "disease_cases"],
      "data": [
        [100.5, 25.3, 12],
        [85.2, 27.1, 8],
        [120.8, 24.5, 15],
        [95.3, 26.2, 10],
        [110.0, 26.0, 13],
        [75.5, 28.0, 6],
        [130.2, 23.8, 18],
        [88.9, 27.5, 9]
      ]
    }
  }'
```

**Response:**
```json
{
  "job_id": "01K79YAHJ7BR4E87VVTG8FNBMB",
  "model_artifact_id": "01K79YAHJ7BR4E87VVTG8FNBMC",
  "status": "pending",
  "message": "Training job submitted"
}
```

### 7. Poll Job Status

```bash
# Poll every 1-2 seconds until status is "completed"
curl http://127.0.0.1:8000/api/v1/jobs/01K79YAHJ7BR4E87VVTG8FNBMB
```

**Response (pending):**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMB",
  "status": "pending",
  "created_at": "2025-10-11T17:31:00Z"
}
```

**Response (completed):**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMB",
  "status": "completed",
  "result": "Training completed successfully",
  "created_at": "2025-10-11T17:31:00Z",
  "updated_at": "2025-10-11T17:31:05Z"
}
```

### 8. Get Trained Model Artifact

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/01K79YAHJ7BR4E87VVTG8FNBMC
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMC",
  "parent_id": null,
  "level": 0,
  "data": {
    "model_type": "sklearn.linear_model.LinearRegression",
    "trained_at": "2025-10-11T17:31:05Z"
  },
  "created_at": "2025-10-11T17:31:00Z",
  "updated_at": "2025-10-11T17:31:05Z"
}
```

### 9. View Artifact Tree

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/01K79YAHJ7BR4E87VVTG8FNBMC/\$tree
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMC",
  "level": 0,
  "level_label": "trained_model",
  "data": {...},
  "children": [],
  "created_at": "2025-10-11T17:31:00Z",
  "updated_at": "2025-10-11T17:31:05Z"
}
```

### 10. Make Predictions

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_artifact_id": "01K79YAHJ7BR4E87VVTG8FNBMC",
    "historic": {
      "columns": ["rainfall", "mean_temperature"],
      "data": []
    },
    "future": {
      "columns": ["rainfall", "mean_temperature"],
      "data": [
        [110.0, 26.0],
        [90.0, 28.0],
        [125.0, 24.0],
        [80.0, 29.0]
      ]
    }
  }'
```

**Response:**
```json
{
  "job_id": "01K79YAHJ7BR4E87VVTG8FNBMD",
  "prediction_artifact_id": "01K79YAHJ7BR4E87VVTG8FNBME",
  "status": "pending",
  "message": "Prediction job submitted"
}
```

### 11. Poll Prediction Job

```bash
curl http://127.0.0.1:8000/api/v1/jobs/01K79YAHJ7BR4E87VVTG8FNBMD
```

Wait until status is `completed`.

### 12. Get Predictions

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/01K79YAHJ7BR4E87VVTG8FNBME
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBME",
  "parent_id": "01K79YAHJ7BR4E87VVTG8FNBMC",
  "level": 1,
  "data": {
    "predictions": {
      "columns": ["rainfall", "mean_temperature", "sample_0"],
      "data": [
        [110.0, 26.0, 13.2],
        [90.0, 28.0, 8.5],
        [125.0, 24.0, 16.8],
        [80.0, 29.0, 6.3]
      ]
    }
  },
  "created_at": "2025-10-11T17:35:00Z",
  "updated_at": "2025-10-11T17:35:02Z"
}
```

The `sample_0` column contains predicted disease cases!

## Advanced Workflows

### List All Configs

```bash
# Simple list
curl http://127.0.0.1:8000/api/v1/configs

# Paginated
curl "http://127.0.0.1:8000/api/v1/configs?page=1&size=10"
```

### Update Config

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/configs/01K79YAHJ7BR4E87VVTG8FNBMA \
  -H "Content-Type: application/json" \
  -d '{
    "name": "updated_model_config",
    "data": {}
  }'
```

### List All Jobs

```bash
# All jobs
curl http://127.0.0.1:8000/api/v1/jobs

# Filter by status
curl "http://127.0.0.1:8000/api/v1/jobs?status_filter=completed"
curl "http://127.0.0.1:8000/api/v1/jobs?status_filter=failed"
```

### List All Artifacts

```bash
# All artifacts
curl http://127.0.0.1:8000/api/v1/artifacts

# Paginated
curl "http://127.0.0.1:8000/api/v1/artifacts?page=1&size=20"
```

### Delete Resources

```bash
# Delete config
curl -X DELETE http://127.0.0.1:8000/api/v1/configs/01K79YAHJ7BR4E87VVTG8FNBMA

# Delete artifact (cascades to children)
curl -X DELETE http://127.0.0.1:8000/api/v1/artifacts/01K79YAHJ7BR4E87VVTG8FNBMC

# Cancel/delete job
curl -X DELETE http://127.0.0.1:8000/api/v1/jobs/01K79YAHJ7BR4E87VVTG8FNBMB
```

## Sample Datasets

### Minimal Training Set (4 samples)

```json
{
  "columns": ["rainfall", "mean_temperature", "disease_cases"],
  "data": [
    [100, 25, 10],
    [120, 24, 15],
    [80, 27, 5],
    [110, 26, 12]
  ]
}
```

### Realistic Training Set (12 samples, seasonal variation)

```json
{
  "columns": ["rainfall", "mean_temperature", "disease_cases"],
  "data": [
    [95.5, 28.2, 8],
    [112.3, 26.7, 14],
    [78.9, 29.5, 4],
    [105.2, 27.1, 11],
    [88.4, 30.0, 6],
    [125.8, 25.3, 18],
    [98.7, 28.9, 9],
    [115.6, 26.0, 15],
    [82.1, 29.8, 5],
    [108.3, 27.5, 12],
    [92.5, 28.6, 7],
    [118.9, 25.8, 16]
  ]
}
```

## Tips

1. **Model Simplicity**: This is a basic LinearRegression - assumes linear relationship between rainfall/temperature and disease cases
2. **No Preprocessing**: Data is used as-is (see ml_class.py for normalization example)
3. **Async Operations**: Both train and predict are async - always poll job status
4. **Artifact Hierarchy**: Predictions are children of trained models (level 0 → level 1)
5. **Logging**: Enabled via `.with_logging()` - check server logs for training details

## Troubleshooting

### "Config not found"
```bash
# Verify config exists
curl http://127.0.0.1:8000/api/v1/configs/YOUR_CONFIG_ID
```

### "Model artifact not found"
```bash
# Verify artifact exists
curl http://127.0.0.1:8000/api/v1/artifacts/YOUR_MODEL_ARTIFACT_ID

# Check job completed successfully
curl http://127.0.0.1:8000/api/v1/jobs/YOUR_JOB_ID
```

### "Missing required columns"
Ensure training data has: `rainfall`, `mean_temperature`, `disease_cases`
Ensure prediction data has: `rainfall`, `mean_temperature`

## Next Steps

- Try **[ml_class.md](ml_class.md)** for preprocessing and normalization
- Try **[ml_shell.md](ml_shell.md)** for external script integration
- Check **[../ml_basic.py](../ml_basic.py)** source code
- Read **[../../CLAUDE.md](../../CLAUDE.md)** for API reference
