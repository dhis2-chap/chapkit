# auth_ml_service.py - Authenticated ML Service cURL Guide

Secure ML service with API key authentication for training and predictions.

## Quick Start

```bash
# Start the service with API keys
export CHAPKIT_API_KEYS="sk_ml_train_abc123,sk_ml_predict_xyz789"
fastapi dev examples/auth_ml_service.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **ML Operations**: Train and predict with disease prediction model
- **Authentication**: All ML endpoints require valid API key
- **Model**: LinearRegression predicting disease cases from weather data
- **Secure**: Health checks public, everything else authenticated
- **Audit**: Structured logging with API key prefix for tracking

## Complete ML Workflow with Authentication

### 1. Check Service Health (No Auth Required)

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

### 2. Try Accessing Config Without Auth (Fails)

```bash
curl http://127.0.0.1:8000/api/v1/configs
```

**Response (401 Unauthorized):**
```json
{
  "type": "urn:chapkit:error:unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "Missing authentication header: X-API-Key",
  "instance": "/api/v1/configs"
}
```

### 3. View System Info (Requires Auth)

```bash
curl -H "X-API-Key: sk_ml_train_abc123" \
  http://127.0.0.1:8000/api/v1/system
```

**Response:**
```json
{
  "current_time": "2025-10-12T16:00:00Z",
  "timezone": "CEST",
  "python_version": "3.13.8",
  "platform": "macOS-26.0.1-arm64-arm-64bit-Mach-O",
  "hostname": "mlaptop.local"
}
```

### 4. Create ML Configuration (With Auth)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "X-API-Key: sk_ml_train_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "01K7CXYHJ7BR4E87VVTG8FNBMA",
    "name": "secure_disease_model",
    "data": {}
  }'
```

**Response:**
```json
{
  "id": "01K7CXYHJ7BR4E87VVTG8FNBMA",
  "name": "secure_disease_model",
  "data": {},
  "created_at": "2025-10-12T16:00:00Z",
  "updated_at": "2025-10-12T16:00:00Z"
}
```

### 5. Train Model (Requires Auth)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "X-API-Key: sk_ml_train_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "01K7CXYHJ7BR4E87VVTG8FNBMA",
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
  "job_id": "01K7CXYHJ7BR4E87VVTG8FNBMB",
  "model_artifact_id": "01K7CXYHJ7BR4E87VVTG8FNBMC",
  "status": "pending",
  "message": "Training job submitted"
}
```

### 6. Poll Training Job Status (Requires Auth)

```bash
# Poll every 1-2 seconds until status is "completed"
curl -H "X-API-Key: sk_ml_train_abc123" \
  http://127.0.0.1:8000/api/v1/jobs/01K7CXYHJ7BR4E87VVTG8FNBMB
```

**Response (completed):**
```json
{
  "id": "01K7CXYHJ7BR4E87VVTG8FNBMB",
  "status": "completed",
  "result": "Training completed successfully",
  "created_at": "2025-10-12T16:01:00Z",
  "updated_at": "2025-10-12T16:01:05Z"
}
```

### 7. Get Trained Model Artifact (Requires Auth)

```bash
curl -H "X-API-Key: sk_ml_train_abc123" \
  http://127.0.0.1:8000/api/v1/artifacts/01K7CXYHJ7BR4E87VVTG8FNBMC
```

**Response:**
```json
{
  "id": "01K7CXYHJ7BR4E87VVTG8FNBMC",
  "parent_id": null,
  "level": 0,
  "data": {
    "model_type": "sklearn.linear_model.LinearRegression",
    "trained_at": "2025-10-12T16:01:05Z"
  },
  "created_at": "2025-10-12T16:01:00Z",
  "updated_at": "2025-10-12T16:01:05Z"
}
```

### 8. Make Predictions (Requires Auth - Can Use Different Key!)

```bash
# Using prediction-specific key
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict \
  -H "X-API-Key: sk_ml_predict_xyz789" \
  -H "Content-Type: application/json" \
  -d '{
    "model_artifact_id": "01K7CXYHJ7BR4E87VVTG8FNBMC",
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
  "job_id": "01K7CXYHJ7BR4E87VVTG8FNBMD",
  "prediction_artifact_id": "01K7CXYHJ7BR4E87VVTG8FNBME",
  "status": "pending",
  "message": "Prediction job submitted"
}
```

### 9. Poll Prediction Job (Requires Auth)

```bash
curl -H "X-API-Key: sk_ml_predict_xyz789" \
  http://127.0.0.1:8000/api/v1/jobs/01K7CXYHJ7BR4E87VVTG8FNBMD
```

Wait until status is `completed`.

### 10. Get Predictions (Requires Auth)

```bash
curl -H "X-API-Key: sk_ml_predict_xyz789" \
  http://127.0.0.1:8000/api/v1/artifacts/01K7CXYHJ7BR4E87VVTG8FNBME
```

**Response:**
```json
{
  "id": "01K7CXYHJ7BR4E87VVTG8FNBME",
  "parent_id": "01K7CXYHJ7BR4E87VVTG8FNBMC",
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
  "created_at": "2025-10-12T16:05:00Z",
  "updated_at": "2025-10-12T16:05:02Z"
}
```

## Multiple API Keys Strategy

Use different keys for different operations:

```bash
# Training key (expensive operations, limited distribution)
TRAIN_KEY="sk_ml_train_abc123"

# Prediction key (cheaper operations, wider distribution)
PREDICT_KEY="sk_ml_predict_xyz789"

# Admin key (full access)
ADMIN_KEY="sk_ml_admin_master001"

# Set environment
export CHAPKIT_API_KEYS="$TRAIN_KEY,$PREDICT_KEY,$ADMIN_KEY"
```

**Benefits:**
- **Rate limiting per key**: Different quotas for train vs predict
- **Usage tracking**: Know which teams/services are using what
- **Security**: Revoke prediction keys without affecting training
- **Cost allocation**: Track costs per API key
- **Compliance**: Audit trails showing who did what

## Production Use Cases

### Case 1: Separate Train/Predict Keys

```bash
# Data science team - training access
export DS_API_KEY="sk_ml_train_abc123"
curl -H "X-API-Key: $DS_API_KEY" -X POST .../ml/\$train ...

# Application team - prediction access only
export APP_API_KEY="sk_ml_predict_xyz789"
curl -H "X-API-Key: $APP_API_KEY" -X POST .../ml/\$predict ...
```

### Case 2: Per-Service Keys

```bash
# Service A: Weather analysis
export SERVICE_A_KEY="sk_ml_service_a_key123"

# Service B: Disease tracking
export SERVICE_B_KEY="sk_ml_service_b_key456"

# Both can predict, tracked separately
```

### Case 3: Per-Customer Keys (SaaS)

```bash
# Customer 1
export CUSTOMER_1_KEY="sk_ml_customer_1_abc123"

# Customer 2
export CUSTOMER_2_KEY="sk_ml_customer_2_xyz789"

# Track usage for billing
```

## Authenticated Endpoints Reference

### Require Authentication ✅

```bash
# Config endpoints
POST   /api/v1/configs
GET    /api/v1/configs
GET    /api/v1/configs/{id}
PUT    /api/v1/configs/{id}
DELETE /api/v1/configs/{id}
GET    /api/v1/configs/$schema

# Artifact endpoints
POST   /api/v1/artifacts
GET    /api/v1/artifacts
GET    /api/v1/artifacts/{id}
GET    /api/v1/artifacts/{id}/$tree
DELETE /api/v1/artifacts/{id}

# Job endpoints
GET    /api/v1/jobs
GET    /api/v1/jobs/{id}
DELETE /api/v1/jobs/{id}

# ML endpoints
POST   /api/v1/ml/$train
POST   /api/v1/ml/$predict

# System endpoints
GET    /api/v1/system
```

### Public (No Auth) ⭐

```bash
# Health and documentation
GET    /health
GET    /docs
GET    /redoc
GET    /openapi.json
GET    /
```

## Security Best Practices

### 1. Least Privilege Access

```bash
# Give training keys only to data scientists
TRAIN_KEYS="sk_ml_train_ds_team_abc123"

# Give prediction keys only to applications
PREDICT_KEYS="sk_ml_predict_app_xyz789"

# Separate keys = separate permissions (future feature)
```

### 2. Key Rotation for ML Services

```bash
# Month 1: Both old and new keys work
export CHAPKIT_API_KEYS="sk_ml_train_old,sk_ml_train_new"

# Update all training jobs to use new key

# Month 2: Remove old key
export CHAPKIT_API_KEYS="sk_ml_train_new"
```

### 3. Monitoring and Alerting

```bash
# Server logs show key prefix for audit
# grep "sk_ml_train" logs/app.log
# grep "auth.invalid_key" logs/app.log

# Track failed authentication attempts
# curl ... | grep "auth.missing_key"
```

### 4. Cost Tracking

```python
# Track API calls per key for billing
import logging

logger = logging.getLogger(__name__)
logger.info(
    "ml_operation",
    operation="train",
    key_prefix=request.state.api_key_prefix,  # "sk_ml_tra"
    cost_estimate=42.50
)
```

## Docker Deployment with ML Authentication

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  ml-api:
    image: your-ml-api-image
    ports:
      - "8000:8000"
    environment:
      CHAPKIT_API_KEYS: ${ML_API_KEYS}
      # Or use secrets (more secure)
    secrets:
      - ml_api_keys
    # Mount model artifacts
    volumes:
      - ml_models:/app/models

secrets:
  ml_api_keys:
    file: ./secrets/ml_api_keys.txt

volumes:
  ml_models:
```

**.env:**
```bash
ML_API_KEYS=sk_ml_train_abc123,sk_ml_predict_xyz789
```

## Troubleshooting

### "Training fails with 401"

**Problem:** Training key not provided or invalid

**Solution:**
```bash
# Verify key
echo $CHAPKIT_API_KEYS

# Use training key
curl -H "X-API-Key: sk_ml_train_abc123" -X POST .../ml/\$train ...
```

### "Prediction fails with 401"

**Problem:** Prediction key not provided

**Solution:**
```bash
# Use prediction key (can be different from training key)
curl -H "X-API-Key: sk_ml_predict_xyz789" -X POST .../ml/\$predict ...
```

### "Job status fails with 401"

**Problem:** Forgot to authenticate job status check

**Solution:**
```bash
# Job status requires authentication
curl -H "X-API-Key: sk_ml_train_abc123" http://127.0.0.1:8000/api/v1/jobs/{id}
```

### "Artifact access fails with 401"

**Problem:** Artifact endpoints require authentication

**Solution:**
```bash
# Use valid key for artifact access
curl -H "X-API-Key: sk_ml_train_abc123" \
  http://127.0.0.1:8000/api/v1/artifacts/{id}
```

## Performance Tips

1. **Reuse connections** with same API key
2. **Batch predictions** when possible
3. **Cache model artifacts** by ID
4. **Monitor key usage** for rate limiting
5. **Use async clients** for parallel requests

## Next Steps

- Try **[auth_envvar.md](auth_envvar.md)** for basic auth setup
- Try **[auth_docker_secrets.md](auth_docker_secrets.md)** for Docker secrets
- Read **[../auth_ml_service.py](../auth_ml_service.py)** source code
- See **[ml_basic.md](ml_basic.md)** for ML workflow without auth
- Check **[../../docs/authentication.md](../../docs/authentication.md)** for comprehensive guide
