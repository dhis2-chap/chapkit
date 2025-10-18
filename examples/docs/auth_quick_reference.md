# Authentication Quick Reference

One-page cheat sheet for Chapkit API authentication.

## Setup Methods

### 1. Environment Variable (Recommended)

```bash
export CHAPKIT_API_KEYS="sk_prod_abc123,sk_prod_xyz789"
fastapi run examples/auth_envvar.py
```

### 2. Docker Secrets (Most Secure)

```bash
# Create secrets file
mkdir -p secrets
echo -e "sk_prod_abc123\nsk_prod_xyz789" > secrets/api_keys.txt
chmod 400 secrets/api_keys.txt

# Set file path
export CHAPKIT_API_KEY_FILE="./secrets/api_keys.txt"
fastapi run examples/auth_docker_secrets.py
```

### 3. Direct Keys (Development Only)

```python
app = ServiceBuilder(info=info).with_auth(
    api_keys=["sk_dev_test123"]  # NOT for production!
).build()
```

## Quick Test Workflow

```bash
# 1. Start service
export CHAPKIT_API_KEYS="sk_dev_test123"
fastapi dev examples/auth_envvar.py

# 2. Health check (no auth)
curl http://localhost:8000/health

# 3. Try without auth (fails)
curl http://localhost:8000/api/v1/configs

# 4. With auth (works)
curl -H "X-API-Key: sk_dev_test123" http://localhost:8000/api/v1/configs

# 5. Create config
curl -X POST http://localhost:8000/api/v1/configs \
  -H "X-API-Key: sk_dev_test123" \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "data": {"key": "value"}}'
```

## API Reference

### Unauthenticated Endpoints

```bash
GET /health      # Health check
GET /docs               # Swagger UI
GET /redoc              # ReDoc
GET /openapi.json       # OpenAPI schema
GET /                   # Landing page
```

### Authenticated Endpoints (Require X-API-Key)

```bash
# Config
POST   /api/v1/configs
GET    /api/v1/configs
GET    /api/v1/configs/{id}
PUT    /api/v1/configs/{id}
DELETE /api/v1/configs/{id}

# Artifacts (if enabled)
POST   /api/v1/artifacts
GET    /api/v1/artifacts
GET    /api/v1/artifacts/{id}
DELETE /api/v1/artifacts/{id}

# Jobs (if enabled)
GET    /api/v1/jobs
GET    /api/v1/jobs/{id}
DELETE /api/v1/jobs/{id}

# ML (if enabled)
POST   /api/v1/ml/$train
POST   /api/v1/ml/$predict
```

## Code Snippets

### Python Service

```python
from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo

class AppConfig(BaseConfig):
    environment: str
    debug: bool = False

app = (
    ServiceBuilder(info=ServiceInfo(display_name="My API"))
    .with_health()
    .with_config(AppConfig)
    .with_auth()  # Reads from CHAPKIT_API_KEYS
    .build()
)
```

### Python Client

```python
import httpx
import os

headers = {"X-API-Key": os.getenv("API_KEY")}

async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://localhost:8000/api/v1/configs",
        headers=headers
    )
    print(response.json())
```

### cURL Examples

```bash
# List configs
curl -H "X-API-Key: sk_dev_test123" http://localhost:8000/api/v1/configs

# Create config
curl -X POST http://localhost:8000/api/v1/configs \
  -H "X-API-Key: sk_dev_test123" \
  -H "Content-Type: application/json" \
  -d '{"name": "prod", "data": {"env": "production"}}'

# Get specific config
curl -H "X-API-Key: sk_dev_test123" \
  http://localhost:8000/api/v1/configs/01K7XXXX...

# Update config
curl -X PUT http://localhost:8000/api/v1/configs/01K7XXXX... \
  -H "X-API-Key: sk_dev_test123" \
  -H "Content-Type: application/json" \
  -d '{"name": "prod_v2", "data": {"env": "production"}}'

# Delete config
curl -X DELETE http://localhost:8000/api/v1/configs/01K7XXXX... \
  -H "X-API-Key: sk_dev_test123"
```

## Key Format Convention

```
sk_{environment}_{random}

sk_prod_a1b2c3d4e5f6g7h8     # Production
sk_staging_x1y2z3a4b5c6d7e8  # Staging
sk_dev_test123               # Development
```

## Configuration Options

```python
.with_auth(
    api_keys=None,                      # List of keys (dev only)
    api_key_file=None,                  # File path (Docker secrets)
    env_var="CHAPKIT_API_KEYS",         # Environment variable name
    header_name="X-API-Key",            # HTTP header for API key
    unauthenticated_paths=None,         # Paths without auth
)
```

## Key Rotation

```bash
# Step 1: Add new key (both work)
export CHAPKIT_API_KEYS="sk_prod_old123,sk_prod_new456"

# Step 2: Update clients to use new key

# Step 3: Remove old key
export CHAPKIT_API_KEYS="sk_prod_new456"
```

## Docker Compose

```yaml
version: '3.8'

services:
  api:
    image: your-api-image
    ports:
      - "8000:8000"
    environment:
      CHAPKIT_API_KEYS: ${CHAPKIT_API_KEYS}
```

## Docker Secrets

```yaml
version: '3.8'

services:
  api:
    image: your-api-image
    ports:
      - "8000:8000"
    secrets:
      - api_keys
    environment:
      CHAPKIT_API_KEY_FILE: /run/secrets/api_keys

secrets:
  api_keys:
    file: ./secrets/api_keys.txt
```

## Error Responses (RFC 9457)

### Missing API Key (401)

```json
{
  "type": "urn:chapkit:error:unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "Missing authentication header: X-API-Key",
  "instance": "/api/v1/configs"
}
```

### Invalid API Key (401)

```json
{
  "type": "urn:chapkit:error:unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "Invalid API key",
  "instance": "/api/v1/configs"
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Missing authentication header" | Add `-H "X-API-Key: sk_..."` to request |
| "Invalid API key" | Check `$CHAPKIT_API_KEYS` matches key used |
| "No API keys configured" | Set `CHAPKIT_API_KEYS` environment variable |
| Health check returns 401 | Ensure `/health` in unauthenticated paths |

## Security Best Practices

✅ **DO:**
- Use environment variables in production
- Rotate keys quarterly
- Use different keys per environment
- Monitor authentication logs
- Keep keys in `.gitignore`

❌ **DON'T:**
- Commit keys to git
- Reuse keys across environments
- Use short keys (min 16 chars)
- Share keys via email/Slack

## Bash Workflow Script

```bash
#!/bin/bash
# auth_workflow.sh - Complete authenticated API workflow

API_KEY="sk_dev_test123"
BASE_URL="http://localhost:8000"

# Start service (in separate terminal)
# export CHAPKIT_API_KEYS="$API_KEY"
# fastapi dev examples/auth_envvar.py

echo "1. Health Check (no auth)"
curl -s $BASE_URL/health | jq

echo "\n2. Try without auth (should fail)"
curl -s $BASE_URL/api/v1/configs

echo "\n3. Create config (with auth)"
CONFIG_ID=$(curl -s -X POST $BASE_URL/api/v1/configs \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "data": {"env": "dev"}}' | jq -r '.id')

echo "Created config: $CONFIG_ID"

echo "\n4. List configs"
curl -s -H "X-API-Key: $API_KEY" $BASE_URL/api/v1/configs | jq

echo "\n5. Get specific config"
curl -s -H "X-API-Key: $API_KEY" $BASE_URL/api/v1/configs/$CONFIG_ID | jq

echo "\n6. Update config"
curl -s -X PUT $BASE_URL/api/v1/configs/$CONFIG_ID \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "test_updated", "data": {"env": "dev"}}' | jq

echo "\n7. Delete config"
curl -s -X DELETE $BASE_URL/api/v1/configs/$CONFIG_ID \
  -H "X-API-Key: $API_KEY"

echo "\n✅ Workflow complete"
```

## Links

- **Examples:** [auth_envvar.md](auth_envvar.md), [auth_docker_secrets.md](auth_docker_secrets.md), [auth_ml.md](auth_ml.md)
- **Source Code:** [../auth_basic.py](../auth_basic.py), [../auth_envvar.py](../auth_envvar.py)
- **Full Guide:** [../../docs/authentication.md](../../docs/authentication.md)
- **Project Docs:** [../../CLAUDE.md](../../CLAUDE.md)
