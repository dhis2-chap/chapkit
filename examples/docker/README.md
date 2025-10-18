# Docker Deployment Examples for Chapkit ML Services

Docker Compose examples for deploying Chapkit ML services with full observability.

## Available Examples

### 1. Basic ML Service (`compose.ml-basic.yml`)

**Use case:** Simple disease prediction ML service using Linear Regression.

**Features:**
- Disease prediction using weather data
- REST API for training and prediction
- Health checks and metrics

**Start:**
```bash
cd examples/docker
docker compose -f compose.ml-basic.yml up
```

**Test:**
```bash
# Health check
curl http://localhost:8000/health

# Train model (requires config and training data)
curl -X POST http://localhost:8000/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d @../../tests/fixtures/train_request.json

# Make predictions
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d @../../tests/fixtures/predict_request.json
```

### 2. Full-Featured ML Service (`compose.ml-full.yml`)

**Use case:** Production ML service with preprocessing, feature engineering, and persistence.

**Features:**
- Class-based ML runner with lifecycle hooks
- Feature normalization with StandardScaler
- Artifact storage for models and predictions
- Database persistence

**Start:**
```bash
cd examples/docker
docker compose -f compose.ml-full.yml up
```

### 3. ML with Monitoring (`compose.ml-monitoring.yml`)

**Use case:** Complete ML service with Prometheus metrics and Grafana dashboards.

**Features:**
- ML training and prediction workflows
- Prometheus metrics collection
- Grafana dashboards for ML metrics
- Full observability stack

**Start:**
```bash
cd examples/docker
docker compose -f compose.ml-monitoring.yml up
```

**Access:**
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Quick Start - ML Basic Example

```bash
# 1. Navigate to docker examples
cd examples/docker

# 2. Start the ML service
docker compose -f compose.ml-basic.yml up -d

# 3. Check health
curl http://localhost:8000/health

# 4. View logs
docker compose -f compose.ml-basic.yml logs -f

# 5. Train a model
curl -X POST http://localhost:8000/api/v1/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "config": {},
    "data": {
      "rainfall": [100, 120, 80, 90, 110],
      "mean_temperature": [25, 28, 22, 24, 26],
      "disease_cases": [10, 15, 7, 9, 13]
    }
  }'

# 6. Make predictions
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "<model_id_from_training>",
    "future": {
      "rainfall": [95, 105],
      "mean_temperature": [23, 25]
    }
  }'
```

## Architecture

### ML Basic Service
```
┌─────────────┐
│   Chapkit   │
│  ML Service │
│             │
│  - Config   │
│  - Artifacts│
│  - ML APIs  │
└─────────────┘
```

### ML with Monitoring
```
┌─────────────┐     ┌────────────┐     ┌──────────┐
│   Chapkit   │────▶│ Prometheus │────▶│ Grafana  │
│  ML Service │     │  (metrics) │     │(dashboard│
│             │     └────────────┘     └──────────┘
│  /metrics   │
└─────────────┘
```

## ML Workflow

1. **Configure**: Set up ML parameters via config API
2. **Train**: Post training data to `/api/v1/ml/train`
3. **Artifacts**: Model stored automatically in artifact hierarchy
4. **Predict**: Use trained model with `/api/v1/ml/predict`
5. **Monitor**: View metrics at `/metrics` or in Grafana

## Environment Variables

### ML Service Configuration
```bash
# Example module (ml_basic, ml_class, ml_pipeline)
EXAMPLE_MODULE=examples.ml_basic:app

# Logging
LOG_FORMAT=json  # or console
LOG_LEVEL=INFO   # DEBUG, INFO, WARNING, ERROR

# Server
PORT=8000
WORKERS=4
TIMEOUT=60
```

## Data Persistence

ML services store artifacts (models, predictions) in the database. To persist data:

```yaml
volumes:
  - ml-data:/app/data
```

## Stopping Services

```bash
# Stop and remove containers
docker compose -f compose.ml-basic.yml down

# Stop and remove with volumes (clears all ML artifacts)
docker compose -f compose.ml-basic.yml down -v
```

## Production Deployment

For production ML deployments:

1. **Use persistent volumes** for model storage
2. **Enable authentication** (see servicekit auth examples)
3. **Configure monitoring** with Prometheus/Grafana
4. **Set appropriate resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```
5. **Use secrets** for API keys and credentials

## Troubleshooting

### Model Training Fails

**Problem:** Training endpoint returns 500 error

**Solution:**
```bash
# Check logs
docker compose logs api

# Verify data format
# Ensure features match expected schema
```

### Out of Memory

**Problem:** Container killed during training

**Solution:**
```yaml
# Increase memory limit
deploy:
  resources:
    limits:
      memory: 8G
```

### Slow Predictions

**Problem:** Prediction latency is high

**Solution:**
- Use `ml_basic` instead of `ml_class` for simpler models
- Increase worker count
- Cache models in memory

## Next Steps

- See `../ml_basic.py` for simple functional ML example
- See `../ml_class.py` for class-based ML runner
- See `../ml_pipeline.py` for complete ML workflow
- Check Servicekit docs for auth and monitoring patterns
