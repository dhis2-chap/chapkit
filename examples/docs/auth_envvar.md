# auth_envvar.py - Environment Variable Authentication cURL Guide

Production-ready API using environment variables for authentication (recommended approach).

## Quick Start

```bash
# Start the service with API keys
export CHAPKIT_API_KEYS="sk_prod_a1b2c3d4e5f6g7h8,sk_prod_x1y2z3a4b5c6d7e8"
fastapi run examples/auth_envvar.py

# Or for development
export CHAPKIT_API_KEYS="sk_dev_test123,sk_dev_test456"
fastapi dev examples/auth_envvar.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Authentication**: Environment variable (CHAPKIT_API_KEYS)
- **Production-ready**: Recommended approach for all deployments
- **Multiple keys**: Supports comma-separated list for rotation
- **Secure logging**: Only first 7 chars logged (sk_prod_****)
- **RFC 9457 errors**: Standard Problem Details format

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

### 3. Access Config With Valid Key (Success)

```bash
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" http://127.0.0.1:8000/api/v1/configs
```

**Response:**
```json
[]
```

*Empty list - no configs created yet*

### 4. Get Config Schema

```bash
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
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
    "region": {
      "default": "us-east-1",
      "type": "string",
      "title": "Region"
    },
    "enable_analytics": {
      "default": true,
      "type": "boolean",
      "title": "Enable Analytics"
    },
    "max_requests_per_minute": {
      "default": 1000,
      "type": "integer",
      "title": "Max Requests Per Minute"
    }
  },
  "required": ["environment"],
  "title": "ProductionConfig"
}
```

### 5. Create Configuration (With Auth)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "01K7AXYHJ7BR4E87VVTG8FNBMA",
    "name": "production_us_east",
    "data": {
      "environment": "production",
      "region": "us-east-1",
      "enable_analytics": true,
      "max_requests_per_minute": 5000
    }
  }'
```

**Response:**
```json
{
  "id": "01K7AXYHJ7BR4E87VVTG8FNBMA",
  "name": "production_us_east",
  "data": {
    "environment": "production",
    "region": "us-east-1",
    "enable_analytics": true,
    "max_requests_per_minute": 5000
  },
  "created_at": "2025-10-12T14:00:00Z",
  "updated_at": "2025-10-12T14:00:00Z"
}
```

### 6. Get Config By ID

```bash
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  http://127.0.0.1:8000/api/v1/configs/01K7AXYHJ7BR4E87VVTG8FNBMA
```

**Response:**
```json
{
  "id": "01K7AXYHJ7BR4E87VVTG8FNBMA",
  "name": "production_us_east",
  "data": {
    "environment": "production",
    "region": "us-east-1",
    "enable_analytics": true,
    "max_requests_per_minute": 5000
  },
  "created_at": "2025-10-12T14:00:00Z",
  "updated_at": "2025-10-12T14:00:00Z"
}
```

### 7. Update Config

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/configs/01K7AXYHJ7BR4E87VVTG8FNBMA \
  -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production_us_east_v2",
    "data": {
      "environment": "production",
      "region": "us-east-1",
      "enable_analytics": true,
      "max_requests_per_minute": 10000
    }
  }'
```

**Response:**
```json
{
  "id": "01K7AXYHJ7BR4E87VVTG8FNBMA",
  "name": "production_us_east_v2",
  "data": {
    "environment": "production",
    "region": "us-east-1",
    "enable_analytics": true,
    "max_requests_per_minute": 10000
  },
  "created_at": "2025-10-12T14:00:00Z",
  "updated_at": "2025-10-12T14:05:00Z"
}
```

### 8. List All Configs

```bash
# Simple list
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  http://127.0.0.1:8000/api/v1/configs

# Paginated
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  "http://127.0.0.1:8000/api/v1/configs?page=1&size=10"
```

### 9. Delete Config

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/configs/01K7AXYHJ7BR4E87VVTG8FNBMA \
  -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8"
```

**Response:** `204 No Content`

## Key Rotation Example

Support multiple keys simultaneously for zero-downtime rotation:

```bash
# Both keys work
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" http://127.0.0.1:8000/api/v1/configs
curl -H "X-API-Key: sk_prod_x1y2z3a4b5c6d7e8" http://127.0.0.1:8000/api/v1/configs
```

**Rotation workflow:**

```bash
# Step 1: Add new key (both old and new work)
export CHAPKIT_API_KEYS="sk_prod_old123,sk_prod_new456"
fastapi run examples/auth_envvar.py

# Step 2: Update all clients to use new key
# Verify clients work with new key

# Step 3: Remove old key
export CHAPKIT_API_KEYS="sk_prod_new456"
fastapi run examples/auth_envvar.py
```

## Environment Variable Formats

### Option 1: Direct Export

```bash
export CHAPKIT_API_KEYS="sk_prod_abc123,sk_prod_xyz789"
fastapi run examples/auth_envvar.py
```

### Option 2: .env File

```bash
# .env
CHAPKIT_API_KEYS=sk_prod_abc123,sk_prod_xyz789
```

```bash
# Load and run
set -a; source .env; set +a
fastapi run examples/auth_envvar.py
```

### Option 3: Inline (Testing Only)

```bash
CHAPKIT_API_KEYS="sk_dev_test123" fastapi dev examples/auth_envvar.py
```

## Docker Compose Deployment

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    image: your-api-image
    ports:
      - "8000:8000"
    environment:
      CHAPKIT_API_KEYS: ${CHAPKIT_API_KEYS}
      # Or directly:
      # CHAPKIT_API_KEYS: "sk_prod_abc123,sk_prod_xyz789"
```

**.env:**
```bash
CHAPKIT_API_KEYS=sk_prod_abc123,sk_prod_xyz789
```

Run:
```bash
docker-compose up
```

## Key Format Convention

**Recommended format:** `sk_{environment}_{random}`

```
sk_prod_a1b2c3d4e5f6g7h8     # Production
sk_staging_x1y2z3a4b5c6d7e8  # Staging
sk_dev_test123               # Development
```

**Why?**
- `sk_` prefix: Easily identifiable as secret key
- `environment`: Know which environment the key belongs to
- `random`: Unique identifier (16+ characters recommended)

## Security Features

1. **Prefix logging only**: Only first 7 chars logged (`sk_prod_****`)
2. **RFC 9457 errors**: Standard Problem Details format
3. **Request tracing**: Key prefix attached to request state
4. **No database storage**: Stateless authentication
5. **Multiple keys**: Enable rotation without downtime

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

## Service-to-Service Communication

### Client Service (Python)

```python
import httpx
import os

headers = {"X-API-Key": os.getenv("API_SERVICE_KEY")}

async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://api-service:8000/api/v1/configs",
        headers=headers
    )
    print(response.json())
```

### Server Service (Your API)

Already configured with `.with_auth()` - reads from `CHAPKIT_API_KEYS`

## Troubleshooting

### "Missing authentication header"

**Problem:** Forgot to include `-H "X-API-Key: ..."` in request

**Solution:** Add header to all protected endpoints:
```bash
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" http://127.0.0.1:8000/api/v1/configs
```

### "Invalid API key"

**Problem:** Wrong key or typo

**Solution:**
1. Check environment variable: `echo $CHAPKIT_API_KEYS`
2. Verify key matches exactly (no extra spaces)
3. Check server logs for key prefix being used
4. Ensure no spaces around commas in key list

### "No API keys configured" (Startup Error)

**Problem:** Environment variable not set

**Solution:**
```bash
export CHAPKIT_API_KEYS="sk_dev_test123"
fastapi dev examples/auth_envvar.py
```

### Health check returns 401

**Problem:** Custom `unauthenticated_paths` removed default paths

**Solution:** Ensure `/health` is in unauthenticated paths list

## Best Practices

### ✅ DO

- **Use environment variables** in all environments (dev, staging, prod)
- **Use different keys** for each environment
- **Rotate keys quarterly** or after security incidents
- **Use minimum 16 characters** for key randomness
- **Monitor authentication logs** for failed attempts
- **Keep `.env` files in `.gitignore`**
- **Use `sk_env_random` format** for easy identification

### ❌ DON'T

- **Commit API keys to git** (use `.gitignore`)
- **Reuse keys across environments**
- **Use weak/short keys** (minimum 16 characters)
- **Share keys via email/Slack** (use secrets management)
- **Hardcode keys in source code**
- **Use spaces in key list** (use `key1,key2` not `key1, key2`)

## Next Steps

- Try **[auth_docker_secrets.md](auth_docker_secrets.md)** for Docker secrets approach
- Try **[auth_ml.md](auth_ml.md)** for authenticated ML service
- Read **[../auth_envvar.py](../auth_envvar.py)** source code
- See **[../../docs/authentication.md](../../docs/authentication.md)** for comprehensive guide
