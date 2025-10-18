# auth_docker_secrets.py - Docker Secrets File Authentication cURL Guide

Secure API using Docker secrets file for authentication (most secure for containers).

## Quick Start

```bash
# 1. Create secrets directory and file
mkdir -p secrets
echo -e "sk_prod_a1b2c3d4e5f6g7h8\nsk_prod_x1y2z3a4b5c6d7e8" > secrets/api_keys.txt

# 2. Add to .gitignore (IMPORTANT!)
echo "secrets/api_keys.txt" >> .gitignore

# 3. Set environment variable pointing to file
export CHAPKIT_API_KEY_FILE="./secrets/api_keys.txt"

# 4. Start the service
fastapi run examples/auth_docker_secrets.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Authentication**: Docker secrets file (one key per line)
- **Most secure**: For Docker Compose, Docker Swarm, Kubernetes
- **Comments supported**: Lines starting with `#` are ignored
- **Multiple keys**: One per line, supports rotation
- **Production-ready**: Best practice for containerized deployments

## Secrets File Format

**secrets/api_keys.txt:**
```
# Production API keys for service authentication
# Format: sk_environment_random (one per line)

sk_prod_a1b2c3d4e5f6g7h8
sk_prod_x1y2z3a4b5c6d7e8

# Staging keys (commented out)
# sk_staging_abc123xyz789
```

**Rules:**
- One API key per line
- Lines starting with `#` are comments (ignored)
- Empty lines are ignored
- No commas or separators needed
- Trailing whitespace is trimmed

## Complete Workflow

### 1. Setup Secrets File

```bash
# Create directory
mkdir -p secrets

# Create secrets file
cat > secrets/api_keys.txt << 'EOF'
# API keys for secure service
sk_prod_a1b2c3d4e5f6g7h8
sk_prod_x1y2z3a4b5c6d7e8
EOF

# Make read-only (security best practice)
chmod 400 secrets/api_keys.txt

# Add to .gitignore
echo "secrets/api_keys.txt" >> .gitignore

# Verify file
cat secrets/api_keys.txt
```

### 2. Start Service with Secrets File

```bash
# Set file path via environment variable
export CHAPKIT_API_KEY_FILE="./secrets/api_keys.txt"

# Run service
fastapi run examples/auth_docker_secrets.py
```

### 3. Check Service Health (No Auth Required)

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

### 4. Test Authentication

```bash
# Without API key (fails)
curl http://127.0.0.1:8000/api/v1/configs
```

**Response (401):**
```json
{
  "type": "urn:chapkit:error:unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "Missing authentication header: X-API-Key",
  "instance": "/api/v1/configs"
}
```

```bash
# With valid API key (succeeds)
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  http://127.0.0.1:8000/api/v1/configs
```

**Response:**
```json
[]
```

### 5. Create Configuration

```bash
curl -X POST http://127.0.0.1:8000/api/v1/configs \
  -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "01K7BXYHJ7BR4E87VVTG8FNBMA",
    "name": "secure_production",
    "data": {
      "service_name": "api-gateway",
      "security_level": "high",
      "audit_enabled": true
    }
  }'
```

**Response:**
```json
{
  "id": "01K7BXYHJ7BR4E87VVTG8FNBMA",
  "name": "secure_production",
  "data": {
    "service_name": "api-gateway",
    "security_level": "high",
    "audit_enabled": true
  },
  "created_at": "2025-10-12T15:00:00Z",
  "updated_at": "2025-10-12T15:00:00Z"
}
```

### 6. List Configs with Both Keys

```bash
# Using first key
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  http://127.0.0.1:8000/api/v1/configs

# Using second key (both work!)
curl -H "X-API-Key: sk_prod_x1y2z3a4b5c6d7e8" \
  http://127.0.0.1:8000/api/v1/configs
```

## Docker Compose Deployment

### Method 1: Using Docker Secrets (Recommended)

**docker-compose.yml:**
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

**secrets/api_keys.txt:**
```
sk_prod_a1b2c3d4e5f6g7h8
sk_prod_x1y2z3a4b5c6d7e8
```

**.gitignore:**
```
secrets/api_keys.txt
```

**Deploy:**
```bash
docker-compose up -d
```

### Method 2: Using Environment Variable (Less Secure)

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    image: your-api-image
    ports:
      - "8000:8000"
    environment:
      CHAPKIT_API_KEY_FILE: /run/secrets/api_keys
    volumes:
      - ./secrets/api_keys.txt:/run/secrets/api_keys:ro
```

## Docker Swarm Deployment

```bash
# Create Docker secret
docker secret create chapkit_api_keys secrets/api_keys.txt

# Deploy service
docker service create \
  --name secure-api \
  --secret chapkit_api_keys \
  -e CHAPKIT_API_KEY_FILE=/run/secrets/chapkit_api_keys \
  -p 8000:8000 \
  your-api-image

# Verify deployment
docker service ps secure-api

# Test
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" http://localhost:8000/api/v1/configs
```

## Kubernetes Deployment

### 1. Create Kubernetes Secret

```bash
# Create secret from file
kubectl create secret generic chapkit-api-keys \
  --from-file=api_keys.txt=secrets/api_keys.txt

# Or create from literal values
kubectl create secret generic chapkit-api-keys \
  --from-literal=api_keys.txt='sk_prod_a1b2c3d4e5f6g7h8
sk_prod_x1y2z3a4b5c6d7e8'
```

### 2. Deploy Application

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secure-api
  template:
    metadata:
      labels:
        app: secure-api
    spec:
      containers:
      - name: api
        image: your-api-image
        ports:
        - containerPort: 8000
        env:
        - name: CHAPKIT_API_KEY_FILE
          value: /etc/secrets/api_keys.txt
        volumeMounts:
        - name: api-keys
          mountPath: /etc/secrets
          readOnly: true
      volumes:
      - name: api-keys
        secret:
          secretName: chapkit-api-keys
```

### 3. Deploy and Test

```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Expose service
kubectl expose deployment secure-api --type=LoadBalancer --port=8000

# Get service URL
kubectl get service secure-api

# Test
curl -H "X-API-Key: sk_prod_a1b2c3d4e5f6g7h8" \
  http://EXTERNAL-IP:8000/api/v1/configs
```

## Key Rotation with Docker Secrets

```bash
# Step 1: Update secrets file (add new key, keep old)
cat > secrets/api_keys.txt << 'EOF'
sk_prod_old_a1b2c3d4e5f6g7h8
sk_prod_new_x1y2z3a4b5c6d7e8
EOF

# Step 2: Update Docker secret
docker secret rm chapkit_api_keys
docker secret create chapkit_api_keys secrets/api_keys.txt

# Step 3: Force service update
docker service update --secret-rm chapkit_api_keys secure-api
docker service update --secret-add chapkit_api_keys secure-api

# Step 4: Update all clients to use new key

# Step 5: Remove old key from file
cat > secrets/api_keys.txt << 'EOF'
sk_prod_new_x1y2z3a4b5c6d7e8
EOF

# Step 6: Repeat update process
```

## Security Best Practices

### File Permissions

```bash
# Make secrets file read-only
chmod 400 secrets/api_keys.txt

# Verify permissions
ls -la secrets/api_keys.txt
# Should show: -r-------- (read-only for owner)
```

### .gitignore Setup

```bash
# Add to .gitignore
cat >> .gitignore << 'EOF'
# Secrets
secrets/api_keys.txt
secrets/*.txt
*.env
.env.*
!.env.example
EOF
```

### Example Secrets File

```bash
# Create example file for team (safe to commit)
cat > secrets/api_keys.txt.example << 'EOF'
# API Keys File Example
# Copy this file to api_keys.txt and replace with real keys

sk_prod_REPLACE_WITH_REAL_KEY_1
sk_prod_REPLACE_WITH_REAL_KEY_2

# For local development:
# sk_dev_test123
EOF

# Add to git
git add secrets/api_keys.txt.example
```

## Troubleshooting

### "FileNotFoundError: API key file not found"

**Problem:** File path is wrong or file doesn't exist

**Solution:**
```bash
# Verify file exists
ls -la secrets/api_keys.txt

# Check environment variable
echo $CHAPKIT_API_KEY_FILE

# Use absolute path if needed
export CHAPKIT_API_KEY_FILE="/full/path/to/secrets/api_keys.txt"
```

### "No API keys found in file"

**Problem:** File is empty or only contains comments/empty lines

**Solution:**
```bash
# Verify file contents
cat secrets/api_keys.txt

# Ensure at least one key exists (not commented)
echo "sk_dev_test123" >> secrets/api_keys.txt
```

### Permission Denied

**Problem:** File permissions too restrictive or running as different user

**Solution:**
```bash
# Make readable
chmod 600 secrets/api_keys.txt

# Or for container users
chmod 644 secrets/api_keys.txt
```

### Keys Not Reloading

**Problem:** Service doesn't automatically reload secrets file

**Solution:**
```bash
# Restart service to reload keys
# (Hot reload not yet implemented)
docker-compose restart api
```

## Comparison with Environment Variables

| Feature | Docker Secrets | Environment Variables |
|---------|---------------|----------------------|
| Security | ⭐⭐⭐⭐⭐ Most secure | ⭐⭐⭐⭐ Secure |
| Container Support | Docker, K8s native | All platforms |
| Key Rotation | File update + restart | Env update + restart |
| Multiple Keys | One per line | Comma-separated |
| Comments | Supported (#) | Not supported |
| Git Safety | File in .gitignore | .env in .gitignore |
| Best For | Production containers | All environments |

## Next Steps

- Try **[auth_envvar.md](auth_envvar.md)** for environment variable approach
- Try **[auth_ml.md](auth_ml.md)** for authenticated ML service
- Read **[../auth_docker_secrets.py](../auth_docker_secrets.py)** source code
- See **[../../docs/authentication.md](../../docs/authentication.md)** for Kubernetes guide
- Check **[../../examples/docker/](../../examples/docker/)** for Docker Compose examples
