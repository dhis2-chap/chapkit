# ML Service Quick Reference

One-page cheat sheet for common ML service operations.

## Start Service

```bash
fastapi dev examples/ml_basic.py   # Basic functional
fastapi dev examples/ml_class.py   # Class-based with preprocessing
fastapi dev examples/ml_shell.py   # Shell-based external scripts
```

## Common Operations

### Health Check
```bash
# One-time health check
curl http://127.0.0.1:8000/health

# Streaming health monitoring
curl -N "http://127.0.0.1:8000/health/\$stream?poll_interval=2.0"
```

### Create Config
```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name":"my_config","data":{}}'
```

### Train Model
```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "CONFIG_ID",
    "data": {
      "columns": ["feature1", "feature2", "target"],
      "data": [[1, 2, 3], [4, 5, 6]]
    }
  }'
```

### Check Job Status
```bash
curl http://127.0.0.1:8000/api/v1/jobs/JOB_ID
```

### Make Predictions
```bash
curl -X POST http://127.0.0.1:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_artifact_id": "MODEL_ID",
    "historic": {
      "columns": ["feature1", "feature2"],
      "data": []
    },
    "future": {
      "columns": ["feature1", "feature2"],
      "data": [[7, 8], [9, 10]]
    }
  }'
```

### Get Artifact
```bash
curl http://127.0.0.1:8000/api/v1/artifacts/ARTIFACT_ID
```

## ML Example Comparison

| Feature | ml_basic | ml_class | ml_shell |
|---------|----------|----------|----------|
| **Style** | Functional | Class-based | External Scripts |
| **Features** | 2 (rainfall, temp) | 3 (rainfall, temp, humidity) | 2 (rainfall, temp) |
| **Preprocessing** | None | StandardScaler | Script-dependent |
| **Config** | Empty | min_samples, normalize | min_samples, model_type |
| **Assessment** | Yellow | Green | Orange |
| **Use Case** | Simple prototypes | Production with preprocessing | Multi-language, containers |
| **State** | Stateless | Shared state (scaler) | File-based |

## Data Format Examples

### Basic Training Data
```json
{
  "columns": ["rainfall", "mean_temperature", "disease_cases"],
  "data": [
    [100, 25, 10],
    [120, 24, 15],
    [80, 27, 5]
  ]
}
```

### Class Training Data (3 features)
```json
{
  "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
  "data": [
    [100, 25, 65, 10],
    [120, 24, 72, 15],
    [80, 27, 58, 5]
  ]
}
```

### Prediction Data
```json
{
  "columns": ["rainfall", "mean_temperature"],
  "data": [
    [110, 26],
    [90, 28]
  ]
}
```

## Endpoint Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/health/$stream` | Health check (streaming) |
| GET | `/api/v1/system` | Service info |
| POST | `/api/v1/configs` | Create config |
| GET | `/api/v1/configs` | List configs |
| GET | `/api/v1/configs/{id}` | Get config |
| PUT | `/api/v1/configs/{id}` | Update config |
| DELETE | `/api/v1/configs/{id}` | Delete config |
| GET | `/api/v1/configs/$schema` | Config schema |
| POST | `/api/v1/ml/$train` | Train model (async) |
| POST | `/api/v1/ml/$predict` | Predict (async) |
| GET | `/api/v1/jobs` | List jobs |
| GET | `/api/v1/jobs/{id}` | Get job status |
| DELETE | `/api/v1/jobs/{id}` | Cancel job |
| GET | `/api/v1/artifacts` | List artifacts |
| GET | `/api/v1/artifacts/{id}` | Get artifact |
| GET | `/api/v1/artifacts/{id}/$tree` | Get artifact tree |
| DELETE | `/api/v1/artifacts/{id}` | Delete artifact |

## Pagination

```bash
# Without pagination (returns array)
curl http://127.0.0.1:8000/api/v1/configs

# With pagination (returns PaginatedResponse)
curl "http://127.0.0.1:8000/api/v1/configs?page=1&size=10"
```

## Using jq for JSON Parsing

```bash
# Extract config ID
CONFIG_ID=$(curl -s -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name":"test","data":{}}' | jq -r '.id')

# Extract job status
STATUS=$(curl -s http://127.0.0.1:8000/api/v1/jobs/$JOB_ID | jq -r '.status')

# Pretty print
curl -s http://127.0.0.1:8000/health | jq '.'
```

## Complete Workflow Script

```bash
#!/bin/bash
set -e

# 1. Create config
CONFIG_ID=$(curl -s -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name":"auto","data":{}}' | jq -r '.id')
echo "Config ID: $CONFIG_ID"

# 2. Train model
TRAIN_RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d "{
    \"config_id\": \"$CONFIG_ID\",
    \"data\": {
      \"columns\": [\"rainfall\", \"mean_temperature\", \"disease_cases\"],
      \"data\": [[100, 25, 10], [120, 24, 15], [80, 27, 5], [110, 26, 12]]
    }
  }")
JOB_ID=$(echo $TRAIN_RESPONSE | jq -r '.job_id')
MODEL_ID=$(echo $TRAIN_RESPONSE | jq -r '.model_artifact_id')
echo "Job ID: $JOB_ID"
echo "Model ID: $MODEL_ID"

# 3. Poll job status
while true; do
  STATUS=$(curl -s http://127.0.0.1:8000/api/v1/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && exit 1
  sleep 1
done

# 4. Make predictions
PRED_RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d "{
    \"model_artifact_id\": \"$MODEL_ID\",
    \"historic\": {
      \"columns\": [\"rainfall\", \"mean_temperature\"],
      \"data\": []
    },
    \"future\": {
      \"columns\": [\"rainfall\", \"mean_temperature\"],
      \"data\": [[110, 26], [90, 28]]
    }
  }")
PRED_JOB_ID=$(echo $PRED_RESPONSE | jq -r '.job_id')
PRED_ID=$(echo $PRED_RESPONSE | jq -r '.prediction_artifact_id')

# 5. Poll prediction job
while true; do
  STATUS=$(curl -s http://127.0.0.1:8000/api/v1/jobs/$PRED_JOB_ID | jq -r '.status')
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && exit 1
  sleep 1
done

# 6. Get predictions
curl -s http://127.0.0.1:8000/api/v1/artifacts/$PRED_ID | jq '.'
```

## Troubleshooting

### Service Won't Start
```bash
# Check port availability
lsof -i :8000

# Kill existing process
kill $(lsof -t -i :8000)
```

### Invalid ULID Error
```bash
# ULIDs must be 26 characters, base32 encoded
# Let service generate IDs - don't specify "id" in POST requests
```

### Job Stays Pending
```bash
# Check service logs for errors
# Verify job scheduler is running (requires .with_jobs())
```

### Config Not Found
```bash
# List all configs
curl http://127.0.0.1:8000/api/v1/configs

# Verify ID is correct (26 chars)
```

## Interactive API Docs

Visit: http://127.0.0.1:8000/docs

- Swagger UI with all endpoints
- Try operations in browser
- See request/response schemas
- Download OpenAPI spec

## Further Reading

- [README.md](README.md) - Overview and common workflows
- [ml_basic.md](ml_basic.md) - Basic functional runner guide
- [ml_class.md](ml_class.md) - Class-based runner with preprocessing
- [ml_shell.md](ml_shell.md) - External script integration
- [../../CLAUDE.md](../../CLAUDE.md) - Full architecture and API reference
