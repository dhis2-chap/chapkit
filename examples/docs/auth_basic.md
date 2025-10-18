# auth_basic.py - API Key Authentication cURL Guide

Simple config service with API key authentication for service-to-service communication.

## Quick Start

```bash
# Start the service
export CHAPKIT_API_KEYS="sk_dev_abc123,sk_dev_xyz789"
fastapi dev examples/auth_basic.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Authentication**: API key via `X-API-Key` header
- **Keys**: Multiple keys supported (for rotation)
- **Unauthenticated paths**: `/docs`, `/health`, `/`
- **Error format**: RFC 9457 Problem Details

## Complete Workflow

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

*Note: Health checks don't require authentication by default*

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

### 3. Access Config With Invalid Key (Fails)

```bash
curl -H "X-API-Key: invalid_key" http://127.0.0.1:8000/api/v1/configs
```

**Response (401 Unauthorized):**
```json
{
  "type": "urn:chapkit:error:unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "Invalid API key",
  "instance": "/api/v1/configs"
}
```

### 4. Access Config With Valid Key (Success)

```bash
curl -H "X-API-Key: sk_dev_abc123" http://127.0.0.1:8000/api/v1/configs
```

**Response:**
```json
[]
```

*Empty list - no configs created yet*

### 5. Create Configuration (With Auth)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "X-API-Key: sk_dev_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
    "name": "production_config",
    "data": {
      "environment": "production",
      "debug": false
    }
  }'
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "name": "production_config",
  "data": {
    "environment": "production",
    "debug": false
  },
  "created_at": "2025-10-12T12:00:00Z",
  "updated_at": "2025-10-12T12:00:00Z"
}
```

### 6. Get Config Schema

```bash
curl -H "X-API-Key: sk_dev_abc123" \
  http://127.0.0.1:8000/api/v1/configs/\$schema
```

**Response:**
```json
{
  "type": "object",
  "properties": {
    "environment": {
      "type": "string",
      "title": "Environment"
    },
    "debug": {
      "default": false,
      "type": "boolean",
      "title": "Debug"
    }
  },
  "required": ["environment"],
  "title": "AppConfig"
}
```

### 7. Get Config By ID

```bash
curl -H "X-API-Key: sk_dev_abc123" \
  http://127.0.0.1:8000/api/v1/configs/01K79YAHJ7BR4E87VVTG8FNBMA
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "name": "production_config",
  "data": {
    "environment": "production",
    "debug": false
  },
  "created_at": "2025-10-12T12:00:00Z",
  "updated_at": "2025-10-12T12:00:00Z"
}
```

### 8. Update Config

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/configs/01K79YAHJ7BR4E87VVTG8FNBMA \
  -H "X-API-Key: sk_dev_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production_config_v2",
    "data": {
      "environment": "production",
      "debug": true
    }
  }'
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "name": "production_config_v2",
  "data": {
    "environment": "production",
    "debug": true
  },
  "created_at": "2025-10-12T12:00:00Z",
  "updated_at": "2025-10-12T12:05:00Z"
}
```

### 9. List All Configs

```bash
# Simple list
curl -H "X-API-Key: sk_dev_abc123" http://127.0.0.1:8000/api/v1/configs

# Paginated
curl -H "X-API-Key: sk_dev_abc123" \
  "http://127.0.0.1:8000/api/v1/configs?page=1&size=10"
```

### 10. Delete Config

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/configs/01K79YAHJ7BR4E87VVTG8FNBMA \
  -H "X-API-Key: sk_dev_abc123"
```

**Response:** `204 No Content`

## Key Rotation Example

Support multiple keys simultaneously for zero-downtime rotation:

```bash
# Both keys work
curl -H "X-API-Key: sk_dev_abc123" http://127.0.0.1:8000/api/v1/configs
curl -H "X-API-Key: sk_dev_xyz789" http://127.0.0.1:8000/api/v1/configs
```

**Rotation workflow:**
1. **Add new key** to `CHAPKIT_API_KEYS` (keep old key)
2. **Update clients** to use new key
3. **Remove old key** once all clients updated

## Unauthenticated Endpoints

These endpoints don't require authentication:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Swagger docs
curl http://127.0.0.1:8000/docs

# ReDoc
curl http://127.0.0.1:8000/redoc

# OpenAPI schema
curl http://127.0.0.1:8000/openapi.json

# Landing page
curl http://127.0.0.1:8000/
```

## Environment Variables

### Option 1: Direct Export (Development)

```bash
export CHAPKIT_API_KEYS="sk_dev_abc123,sk_dev_xyz789"
fastapi dev examples/auth_basic.py
```

### Option 2: .env File

```bash
# .env
CHAPKIT_API_KEYS=sk_prod_abc123,sk_prod_xyz789
```

```bash
# Load and run
set -a; source .env; set +a
fastapi run examples/auth_basic.py
```

### Option 3: Docker Secrets (Production)

See `docs/authentication.md` for Docker Compose and Kubernetes examples.

## Key Format Convention

**Recommended format:** `sk_{environment}_{random}`

```
sk_prod_a1b2c3d4e5f6g7h8     # Production
sk_staging_x1y2z3a4b5c6d7e8  # Staging
sk_dev_test123               # Development
```

**Why?**
- `sk_` prefix: Easily identifiable as secret key
- `environment`: Know which environment
- `random`: 16+ characters recommended

## Security Features

1. **Prefix logging only**: Only first 7 chars logged (`sk_prod_****`)
2. **RFC 9457 errors**: Standard Problem Details format
3. **Request tracing**: Key prefix attached to request state
4. **No database storage**: Stateless authentication

## Tips

1. **Never commit keys**: Add `.env` to `.gitignore`
2. **Rotate regularly**: Use multiple keys for zero-downtime rotation
3. **Different keys per service**: Don't reuse across environments
4. **Minimum 16 chars**: For random portion of key
5. **Monitor logs**: Watch for authentication failures

## Troubleshooting

### "Missing authentication header"
**Problem:** Forgot to include `-H "X-API-Key: ..."` in request

**Solution:** Add header to all protected endpoints:
```bash
curl -H "X-API-Key: sk_dev_abc123" http://127.0.0.1:8000/api/v1/configs
```

### "Invalid API key"
**Problem:** Wrong key or typo

**Solution:**
1. Check `CHAPKIT_API_KEYS` environment variable: `echo $CHAPKIT_API_KEYS`
2. Verify key matches exactly (no extra spaces)
3. Check server logs for key prefix being used

### Health check returns 401
**Problem:** Custom `unauthenticated_paths` removed default paths

**Solution:** Ensure `/health` is in unauthenticated paths list

## Service-to-Service Communication

### Service A (Client)

```python
import httpx
import os

headers = {"X-API-Key": os.getenv("SERVICE_B_API_KEY")}

async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://service-b:8000/api/v1/configs",
        headers=headers
    )
    print(response.json())
```

### Service B (Server)

```python
from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo

class AppConfig(BaseConfig):
    environment: str
    debug: bool = False

app = (
    ServiceBuilder(info=ServiceInfo(display_name="Service B"))
    .with_health()
    .with_config(AppConfig)
    .with_auth()  # Reads from CHAPKIT_API_KEYS
    .build()
)
```

## Next Steps

- Read **[../auth_basic.py](../auth_basic.py)** source code
- Check **[../../docs/authentication.md](../../docs/authentication.md)** for comprehensive guide
- See **[../../CLAUDE.md](../../CLAUDE.md)** for API reference
