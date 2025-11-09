# Configuration Management

Chapkit provides a type-safe configuration management system for storing and managing application settings, environment configurations, and ML model parameters with JSON storage and optional artifact linking.

## Quick Start

```python
from chapkit import BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo

class AppConfig(BaseConfig):
    """Application configuration schema."""
    debug: bool
    api_host: str
    api_port: int

app = (
    ServiceBuilder(info=ServiceInfo(display_name="My Service"))
    .with_health()
    .with_config(AppConfig)
    .build()
)
```

**Run:** `fastapi dev your_file.py`

Visit http://localhost:8000/docs to manage configurations via Swagger UI.

---

## Architecture

### Configuration Storage

Configurations are stored as key-value pairs with JSON data:

```
Config
  ├─ name: "production" (unique identifier)
  ├─ data: {...}         (validated against schema)
  ├─ id: ULID            (auto-generated)
  └─ created_at, updated_at, tags
```

### Type Safety with Pydantic

```python
class MLConfig(BaseConfig):
    model_name: str
    learning_rate: float = 0.001
    epochs: int = 100
    features: list[str]
```

**Benefits:**
- Compile-time type checking
- Runtime validation
- Automatic API documentation
- JSON schema generation
- Extra fields allowed by default

### Artifact Linking

Link configs to trained models or experiment results:

```
Config("production_model")
  └─> Trained Model Artifact (level 0)
       ├─> Predictions 1 (level 1)
       └─> Predictions 2 (level 1)
```

---

## Core Concepts

### BaseConfig

Base class for all configuration schemas with flexible schema support.

```python
from chapkit import BaseConfig

class DatabaseConfig(BaseConfig):
    """Database connection configuration."""
    host: str
    port: int = 5432
    username: str
    password: str
    database: str
    ssl_enabled: bool = True
```

**Features:**
- Inherits from `pydantic.BaseModel`
- `extra="allow"` - accepts arbitrary additional fields
- JSON serializable
- Validation on instantiation

### ConfigIn / ConfigOut

Input and output schemas for API operations.

```python
from chapkit import ConfigIn, ConfigOut

# Create config
config_in = ConfigIn[DatabaseConfig](
    name="production_db",
    data=DatabaseConfig(
        host="db.example.com",
        port=5432,
        username="app_user",
        password="secret",
        database="prod"
    )
)

# Response schema
config_out: ConfigOut[DatabaseConfig] = await manager.save(config_in)
```

### ConfigManager

Business logic layer for configuration operations.

```python
from chapkit import ConfigManager, ConfigRepository

manager = ConfigManager[AppConfig](repository, AppConfig)

# Create/update
config = await manager.save(ConfigIn(name="dev", data=app_config))

# Find by name
config = await manager.find_by_name("dev")

# List all
configs = await manager.find_all()

# Delete
await manager.delete_by_id(config_id)
```

---

## API Endpoints

### POST /api/v1/configs

Create new configuration.

**Request:**
```json
{
  "name": "production",
  "data": {
    "debug": false,
    "api_host": "0.0.0.0",
    "api_port": 8080,
    "max_connections": 2000
  }
}
```

**Response (201 Created):**
```json
{
  "id": "01K72P5N5KCRM6MD3BRE4P07N8",
  "name": "production",
  "data": {
    "debug": false,
    "api_host": "0.0.0.0",
    "api_port": 8080,
    "max_connections": 2000
  },
  "created_at": "2025-10-24T12:00:00Z",
  "updated_at": "2025-10-24T12:00:00Z",
  "tags": []
}
```

### GET /api/v1/configs

List all configurations with pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Page size (default: 50)

**Response:**
```json
{
  "items": [...],
  "total": 3,
  "page": 1,
  "size": 50,
  "pages": 1
}
```

### GET /api/v1/configs/{id}

Get configuration by ID.

### PUT /api/v1/configs/{id}

Update configuration.

**Request:**
```json
{
  "name": "production",
  "data": {
    "debug": false,
    "api_host": "0.0.0.0",
    "api_port": 9090,
    "max_connections": 3000
  }
}
```

### DELETE /api/v1/configs/{id}

Delete configuration.

**Note:** When deleted, all linked artifact trees are cascade deleted.

---

## Artifact Linking Operations

Enable artifact operations when building the service:

```python
app = (
    ServiceBuilder(info=info)
    .with_config(MLConfig, enable_artifact_operations=True)
    .with_artifacts(hierarchy=hierarchy)
    .build()
)
```

### POST /api/v1/configs/{id}/$link-artifact

Link a root artifact to a config.

**Request:**
```json
{
  "artifact_id": "01MODEL456..."
}
```

**Response:** 204 No Content

**Validation:**
- Artifact must exist
- Artifact must be a root (parent_id is NULL)
- Each artifact can only be linked to one config

### POST /api/v1/configs/{id}/$unlink-artifact

Unlink an artifact from a config.

**Request:**
```json
{
  "artifact_id": "01MODEL456..."
}
```

**Response:** 204 No Content

### GET /api/v1/configs/{id}/$artifacts

Get all root artifacts linked to a config.

**Response:**
```json
[
  {
    "id": "01MODEL456...",
    "parent_id": null,
    "level": 0,
    "data": {...},
    "created_at": "2025-10-24T10:00:00Z",
    "updated_at": "2025-10-24T10:00:00Z"
  }
]
```

---

## Configuration Patterns

### Environment Configurations

```python
class EnvironmentConfig(BaseConfig):
    """Environment-specific configuration."""
    debug: bool
    api_host: str
    api_port: int
    database_url: str
    log_level: str = "INFO"
    max_connections: int = 100

# Create configs for different environments
prod_config = ConfigIn(
    name="production",
    data=EnvironmentConfig(
        debug=False,
        api_host="0.0.0.0",
        api_port=8080,
        database_url="postgresql://...",
        log_level="WARNING",
        max_connections=2000
    )
)

dev_config = ConfigIn(
    name="development",
    data=EnvironmentConfig(
        debug=True,
        api_host="127.0.0.1",
        api_port=8000,
        database_url="sqlite:///./dev.db",
        log_level="DEBUG",
        max_connections=10
    )
)
```

### ML Model Configurations

```python
class MLModelConfig(BaseConfig):
    """Machine learning model configuration."""
    model_type: str
    learning_rate: float
    batch_size: int
    epochs: int
    features: list[str]
    hyperparameters: dict[str, float]

config = ConfigIn(
    name="weather_model_v2",
    data=MLModelConfig(
        model_type="RandomForest",
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        features=["temperature", "humidity", "pressure"],
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2
        }
    )
)
```

### Nested Configurations

```python
class DatabaseSettings(BaseModel):
    """Database connection settings."""
    host: str
    port: int
    ssl: bool = True

class CacheSettings(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    ttl_seconds: int = 3600

class AppConfig(BaseConfig):
    """Application configuration with nested settings."""
    app_name: str
    version: str
    database: DatabaseSettings
    cache: CacheSettings
    debug: bool = False

config = ConfigIn(
    name="app_config",
    data=AppConfig(
        app_name="My API",
        version="1.0.0",
        database=DatabaseSettings(
            host="db.example.com",
            port=5432,
            ssl=True
        ),
        cache=CacheSettings(
            enabled=True,
            ttl_seconds=7200
        ),
        debug=False
    )
)
```

### Extra Fields Support

```python
# BaseConfig allows extra fields
config = ConfigIn(
    name="flexible_config",
    data=AppConfig(
        required_field="value",
        dynamic_field="extra_value",  # Not in schema but allowed
        another_field=123
    )
)
```

---

## Database Seeding

Seed configurations on application startup:

```python
from fastapi import FastAPI
from servicekit import Database
from chapkit import ConfigIn, ConfigManager, ConfigRepository

SEED_CONFIGS = [
    ("production", EnvironmentConfig(debug=False, ...)),
    ("staging", EnvironmentConfig(debug=True, ...)),
    ("local", EnvironmentConfig(debug=True, ...)),
]

async def seed_configs(app: FastAPI) -> None:
    """Seed database with predefined configurations."""
    database: Database = app.state.database

    async with database.session() as session:
        repo = ConfigRepository(session)
        manager = ConfigManager[EnvironmentConfig](repo, EnvironmentConfig)

        # Clear existing configs (optional)
        await manager.delete_all()

        # Seed new configs
        await manager.save_all(
            ConfigIn(name=name, data=data)
            for name, data in SEED_CONFIGS
        )

app = (
    ServiceBuilder(info=info)
    .with_config(EnvironmentConfig)
    .on_startup(seed_configs)
    .build()
)
```

---

## Complete Workflow Example

### 1. Define Configuration Schema

```python
class WeatherModelConfig(BaseConfig):
    """Configuration for weather prediction model."""
    model_version: str
    training_features: list[str]
    prediction_horizon_days: int
    update_frequency: str
```

### 2. Build Service

```python
app = (
    ServiceBuilder(info=ServiceInfo(display_name="Weather Model Service"))
    .with_health()
    .with_config(WeatherModelConfig, enable_artifact_operations=True)
    .with_artifacts(hierarchy=ArtifactHierarchy(
        name="weather_models",
        level_labels={0: "trained_model", 1: "predictions"}
    ))
    .build()
)
```

### 3. Create Configuration

```bash
CONFIG_ID=$(curl -s -X POST http://localhost:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "weather_v1",
    "data": {
      "model_version": "1.0.0",
      "training_features": ["temperature", "humidity", "pressure"],
      "prediction_horizon_days": 7,
      "update_frequency": "daily"
    }
  }' | jq -r '.id')

echo "Config ID: $CONFIG_ID"
```

### 4. Train Model (Creates Artifact)

```bash
# Train model - creates artifact
TRAIN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "'$CONFIG_ID'",
    "data": {...}
  }')

MODEL_ARTIFACT_ID=$(echo $TRAIN_RESPONSE | jq -r '.artifact_id')
```

### 5. Link Model to Config

```bash
curl -X POST http://localhost:8000/api/v1/configs/$CONFIG_ID/\$link-artifact \
  -H "Content-Type: application/json" \
  -d '{"artifact_id": "'$MODEL_ARTIFACT_ID'"}'
```

### 6. Query Linked Artifacts

```bash
curl http://localhost:8000/api/v1/configs/$CONFIG_ID/\$artifacts | jq
```

### 7. Update Configuration

```bash
curl -X PUT http://localhost:8000/api/v1/configs/$CONFIG_ID \
  -H "Content-Type: application/json" \
  -d '{
    "name": "weather_v1",
    "data": {
      "model_version": "1.1.0",
      "training_features": ["temperature", "humidity", "pressure", "wind_speed"],
      "prediction_horizon_days": 14,
      "update_frequency": "twice_daily"
    }
  }' | jq
```

---

## Testing

### Unit Tests

```python
import pytest
from chapkit import BaseConfig, ConfigIn, ConfigManager, ConfigRepository

class TestConfig(BaseConfig):
    """Test configuration schema."""
    setting: str
    value: int

@pytest.mark.asyncio
async def test_config_crud(session):
    """Test config CRUD operations."""
    repo = ConfigRepository(session)
    manager = ConfigManager[TestConfig](repo, TestConfig)

    # Create
    config_in = ConfigIn(
        name="test",
        data=TestConfig(setting="test_setting", value=42)
    )
    config = await manager.save(config_in)

    assert config.name == "test"
    assert config.data.setting == "test_setting"
    assert config.data.value == 42

    # Find by name
    found = await manager.find_by_name("test")
    assert found is not None
    assert found.id == config.id

    # Update
    config_in.data.value = 100
    updated = await manager.save(config_in)
    assert updated.data.value == 100

    # Delete
    await manager.delete_by_id(config.id)
    assert await manager.find_by_id(config.id) is None
```

### Integration Tests with Artifact Linking

```python
@pytest.mark.asyncio
async def test_config_artifact_linking(session, artifact_manager, config_manager):
    """Test linking configs to artifacts."""
    # Create config
    config = await config_manager.save(ConfigIn(
        name="model_config",
        data=TestConfig(setting="ml", value=1)
    ))

    # Create root artifact
    artifact = await artifact_manager.save(ArtifactIn(
        data={"model": "trained"}
    ))

    # Link artifact to config
    await config_manager.link_artifact(config.id, artifact.id)

    # Verify link
    linked_artifacts = await config_manager.get_linked_artifacts(config.id)
    assert len(linked_artifacts) == 1
    assert linked_artifacts[0].id == artifact.id

    # Unlink
    await config_manager.unlink_artifact(artifact.id)
    linked_artifacts = await config_manager.get_linked_artifacts(config.id)
    assert len(linked_artifacts) == 0
```

### cURL Testing

```bash
# Create config
curl -X POST http://localhost:8000/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "data": {"debug": true, "port": 8000}}'

# List configs
curl http://localhost:8000/api/v1/configs | jq

# Get by ID
curl http://localhost:8000/api/v1/configs/01CONFIG123... | jq

# Update
curl -X PUT http://localhost:8000/api/v1/configs/01CONFIG123... \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "data": {"debug": false, "port": 9000}}'

# Delete
curl -X DELETE http://localhost:8000/api/v1/configs/01CONFIG123...
```

---

## Production Considerations

### Configuration Versioning

Use config names to track versions:

```python
# Version in name
configs = [
    ConfigIn(name="model_v1.0.0", data=config_data_v1),
    ConfigIn(name="model_v1.1.0", data=config_data_v11),
    ConfigIn(name="model_v2.0.0", data=config_data_v2),
]

# Or in data
class VersionedConfig(BaseConfig):
    version: str
    settings: dict[str, object]

config = ConfigIn(
    name="production_model",
    data=VersionedConfig(
        version="1.2.3",
        settings={...}
    )
)
```

### Environment Variables

Load configurations from environment:

```python
import os
from pydantic_settings import BaseSettings

class EnvConfig(BaseSettings):
    """Configuration from environment variables."""
    database_url: str
    api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

# Load from environment
env_config = EnvConfig()

# Store in database
config_in = ConfigIn(
    name="from_env",
    data=AppConfig(
        database_url=env_config.database_url,
        api_key=env_config.api_key,
        debug=env_config.debug
    )
)
```

### Secrets Management

**Never store secrets in configs**:

```python
# BAD: Secrets in database
class BadConfig(BaseConfig):
    api_key: str  # Don't store in database!
    password: str  # Don't store in database!

# GOOD: Reference to secrets
class GoodConfig(BaseConfig):
    secret_name: str  # Reference to secret manager
    credential_id: str  # Reference to vault

# Usage
config = ConfigIn(
    name="api_config",
    data=GoodConfig(
        secret_name="production_api_key",  # Load from AWS Secrets Manager
        credential_id="vault:db/prod"       # Load from HashiCorp Vault
    )
)
```

### Config Validation

Add custom validation:

```python
from pydantic import field_validator

class ValidatedConfig(BaseConfig):
    """Configuration with custom validation."""
    port: int
    workers: int
    timeout_seconds: int

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        """Validate worker count."""
        if v < 1 or v > 32:
            raise ValueError("Workers must be between 1 and 32")
        return v
```

### Backup Configurations

```bash
# Export all configs
curl http://localhost:8000/api/v1/configs?size=1000 | jq > configs_backup.json

# Restore
cat configs_backup.json | jq -c '.items[]' | while read config; do
  curl -X POST http://localhost:8000/api/v1/configs \
    -H "Content-Type: application/json" \
    -d "$config"
done
```

---

## Troubleshooting

### Validation Errors

**Problem:** Config creation fails with validation errors.

**Cause:** Data doesn't match schema.

**Solution:**
```python
# Check schema
print(YourConfig.model_json_schema())

# Validate data before saving
try:
    validated = YourConfig(debug=True, port="invalid")
except ValidationError as e:
    print(e.errors())
```

### Extra Fields Not Saved

**Problem:** Additional fields disappear after saving.

**Cause:** Only fields in schema are saved unless using `extra="allow"`.

**Solution:**
```python
# Ensure BaseConfig is used (has extra="allow")
class MyConfig(BaseConfig):  # Inherits extra="allow"
    required_field: str
    # Extra fields automatically allowed
```

### Artifact Link Fails

**Problem:** "Artifact is not a root artifact" error.

**Cause:** Trying to link a child artifact (has parent_id).

**Solution:**
```bash
# Check artifact
curl http://localhost:8000/api/v1/artifacts/$ARTIFACT_ID | jq '.parent_id'

# Should be null for root artifacts
# Only link root artifacts to configs
```

### Config Deletion Cascade

**Problem:** Deleting config also deletes artifacts.

**Cause:** Cascade delete removes entire artifact tree.

**Solution:**
```python
# Unlink artifacts before deleting config
artifacts = await manager.get_linked_artifacts(config_id)
for artifact in artifacts:
    await manager.unlink_artifact(artifact.id)

# Then delete config
await manager.delete_by_id(config_id)
```

---

## Complete Example

See `examples/config_basic/` for a complete working example with:
- Custom configuration schema
- Database seeding
- Environment configurations
- Custom service metadata
- Docker deployment

---

## Next Steps

- **Artifact Storage:** Link configs to trained models and results
- **ML Workflows:** Use configs for training and prediction
- **Database Migrations:** Add custom config tables
- **Monitoring:** Track config usage and changes
