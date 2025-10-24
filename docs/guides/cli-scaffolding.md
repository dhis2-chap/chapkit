# CLI Scaffolding

Chapkit provides a CLI tool for quickly scaffolding new ML service projects with all necessary configuration files, Docker setup, and optional monitoring stack.

## Installation

### From PyPI (Recommended)

Once chapkit is published to PyPI, you can use it with uvx (no installation needed):

```bash
# One-off project creation
uvx chapkit init my-ml-service
```

Or install it permanently:

```bash
# Install globally
uv tool install chapkit

# Use the installed tool
chapkit init my-ml-service
```

### From GitHub (Development/Pre-release)

Before PyPI release or to use the latest development version:

```bash
# One-off project creation from GitHub
uvx --from git+https://github.com/dhis2-chap/chapkit chapkit init my-ml-service

# Or install from GitHub
uv tool install git+https://github.com/dhis2-chap/chapkit
chapkit init my-ml-service
```

## Quick Start

Create and run a new project:

```bash
# Create project (using uvx for one-off usage)
uvx chapkit init my-ml-service
cd my-ml-service

# Install dependencies and run
uv sync
uv run python main.py
```

Visit http://localhost:8000/docs to interact with the ML API.

---

## CLI Commands

### `chapkit init`

Initialize a new chapkit ML service project.

**Usage:**

```bash
chapkit init PROJECT_NAME [OPTIONS]
```

**Arguments:**

- `PROJECT_NAME` - Name of the project to create (required)

**Options:**

- `--path PATH` - Target directory (default: current directory)
- `--monitoring` - Include Prometheus and Grafana monitoring stack
- `--help` - Show help message

**Examples:**

```bash
# Using uvx (one-off, no installation needed)
uvx chapkit init my-service

# Using installed tool
chapkit init my-service

# Create project in specific location
chapkit init my-service --path ~/projects

# Create project with monitoring stack
chapkit init my-service --monitoring

# From GitHub (before PyPI release)
uvx --from git+https://github.com/dhis2-chap/chapkit chapkit init my-service
```

---

## Generated Project Structure

### Basic Project

```
my-service/
├── main.py              # ML service application
├── pyproject.toml       # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── compose.yml          # Docker Compose configuration
├── .gitignore           # Python gitignore
└── README.md            # Project documentation
```

### With Monitoring

When using `--monitoring`, additional files are generated:

```
my-service/
├── main.py
├── pyproject.toml
├── Dockerfile
├── compose.yml          # Includes Prometheus & Grafana
├── .gitignore
├── README.md
└── monitoring/
    ├── prometheus/
    │   └── prometheus.yml
    └── grafana/
        ├── provisioning/
        │   ├── datasources/prometheus.yml
        │   └── dashboards/dashboard.yml
        └── dashboards/
            └── chapkit-service-metrics.json
```

---

## Generated Files

### main.py

The generated `main.py` includes:

- **Config Schema**: Pydantic model for ML parameters
- **Training Function**: `on_train` with simple model example
- **Prediction Function**: `on_predict` for inference
- **Service Info**: Metadata (name, version, author, status)
- **Artifact Hierarchy**: Storage structure for models and predictions
- **FastAPI App**: Built using `MLServiceBuilder`

**Example structure:**

```python
class MyServiceConfig(BaseConfig):
    # Add your model parameters here
    pass

async def on_train(config, data, geo=None):
    # Training logic
    model = {"means": data.select_dtypes(include=["number"]).mean().to_dict()}
    return model

async def on_predict(config, model, historic, future, geo=None):
    # Prediction logic
    return future

runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)
app = MLServiceBuilder(...).with_monitoring().build()
```

### pyproject.toml

Defines project metadata and dependencies:

```toml
[project]
name = "my-service"
version = "0.1.0"
description = "ML service for my-service"
requires-python = ">=3.13"
dependencies = ["chapkit"]

[dependency-groups]
dev = ["uvicorn[standard]>=0.30.0"]
```

### Dockerfile

Multi-stage Docker build with:

- **Builder stage**: UV-based dependency installation
- **Runtime stage**: Slim Python image with gunicorn/uvicorn
- Health checks and proper user setup
- Environment variables for configuration

### compose.yml

**Basic version:**

- Single service (API) on port 8000
- Health checks
- Configurable workers and logging

**Monitoring version:**

- API service (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000, admin/admin)
- Pre-configured dashboards and datasources

---

## Customization

### Update Model Logic

Edit the `on_train` and `on_predict` functions in `main.py`:

```python
async def on_train(config, data, geo=None):
    # Your training logic here
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(data[features], data[target])
    return model
```

### Add Configuration Parameters

Extend the config schema:

```python
class MyServiceConfig(BaseConfig):
    min_samples: int = 5
    learning_rate: float = 0.001
    features: list[str] = ["feature_1", "feature_2"]
```

### Add Dependencies

Use `uv` to add packages:

```bash
uv add scikit-learn pandas numpy
```

### Customize Service Metadata

Update the `MLServiceInfo`:

```python
info = MLServiceInfo(
    display_name="Production Model",
    version="2.0.0",
    summary="Production-ready ML service",
    description="Detailed description here",
    author="Your Team",
    author_assessed_status=AssessedStatus.green,
    contact_email="team@example.com",
)
```

---

## Development Workflow

### Local Development

```bash
# Install dependencies
uv sync

# Run development server
uv run python main.py

# Run tests (if added)
pytest

# Lint code
ruff check .
```

### Docker Development

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus** (if monitoring enabled): http://localhost:9090
- **Grafana** (if monitoring enabled): http://localhost:3000

---

## Next Steps

After scaffolding your project:

1. **Customize the model**: Update `on_train` and `on_predict` functions
2. **Add dependencies**: Use `uv add` to install required packages
3. **Update configuration**: Add model-specific parameters to config schema
4. **Test locally**: Run with `uv run python main.py`
5. **Dockerize**: Build and test with `docker compose up --build`
6. **Add tests**: Create tests for your training and prediction logic
7. **Deploy**: Use the generated Dockerfile for deployment

## Related Documentation

- [ML Workflows](ml-workflows.md) - Learn about model training and prediction
- [Configuration Management](configuration-management.md) - Working with configs
- [Artifact Storage](artifact-storage.md) - Managing models and predictions
- [Task Execution](task-execution.md) - Scheduling background jobs
