# CLI Scaffolding

Chapkit provides a CLI tool for quickly scaffolding new ML service projects with all necessary configuration files, Docker setup, and optional monitoring stack.

## Installation

### From PyPI (Recommended)

Use chapkit with uvx (no installation needed):

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

### From GitHub (Development)

To use the latest development version:

```bash
# One-off project creation from GitHub
uvx --from git+https://github.com/dhis2-chap/chapkit chapkit init my-ml-service

# Or install from GitHub
uv tool install git+https://github.com/dhis2-chap/chapkit
chapkit init my-ml-service
```

---

## Managing Installed Tool

If you installed chapkit with `uv tool install`, you can manage the installation:

### Check Installed Version

```bash
# List installed tools and versions
uv tool list

# Check version
chapkit --version
```

### Upgrade to Latest

```bash
# Upgrade to latest version
uv tool upgrade chapkit

# Upgrade to specific version
uv tool upgrade [email protected]
```

### Uninstall

```bash
# Remove installed tool
uv tool uninstall chapkit
```

**Note:** When using `uvx chapkit`, the latest version is used automatically unless you specify a version with `@`:

```bash
# Always uses latest
uvx chapkit init my-service

# Pin to specific version
uvx [email protected] init my-service
```

---

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
- `--with-monitoring` - Include Prometheus and Grafana monitoring stack
- `--template TYPE` - Template type: `ml` (default), `ml-shell`, or `task`
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
chapkit init my-service --with-monitoring

# Create project with ml-shell template (language-agnostic)
chapkit init my-service --template ml-shell

# Create project with task template (task execution)
chapkit init my-service --template task

# Combine options
chapkit init my-service --template ml-shell --with-monitoring

# From GitHub (development version)
uvx --from git+https://github.com/dhis2-chap/chapkit chapkit init my-service
```

---

## Template Types

### ML Template (Default)

The ML template is the simpler approach where you define training and prediction logic as Python functions directly in `main.py`:

**Pros:**
- Simpler to understand and get started
- All code in one file
- Direct access to Python ecosystem
- No external processes

**Cons:**
- Python-only workflows
- Less isolation between training and prediction

**Best for:** Python-centric ML workflows, prototyping, simpler models

### ML-Shell Template

The ML-shell template executes external scripts for training and prediction, enabling language-agnostic ML workflows:

**Pros:**
- Language-agnostic (Python, R, Julia, etc.)
- Better isolation and testing
- Can integrate existing scripts without modification
- File-based data interchange (CSV, YAML, pickle)

**Cons:**
- More files to manage
- Requires understanding of file I/O formats
- Slightly more complex

**Best for:** Multi-language environments, integrating existing scripts, team collaboration with different language preferences

### Task Template

The task template provides a general-purpose task execution system with both Python functions and shell commands:

**Pros:**
- Execute both Python functions and shell commands
- Dependency injection (Database, ArtifactManager, etc.)
- Dynamic task creation via API
- Job-based async execution
- Task results stored in artifacts

**Cons:**
- Not ML-specific (no train/predict operations)
- Requires understanding of task registry
- More complex than simple function calls

**Best for:** General-purpose automation, data processing pipelines, scheduled tasks, non-ML workflows

---

## Generated Project Structure

### ML Template (Default)

```
my-service/
├── main.py              # ML service with train/predict functions
├── pyproject.toml       # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── compose.yml          # Docker Compose configuration
├── data/                # Database directory
│   └── chapkit.db       # SQLite database (created at runtime)
├── .gitignore           # Python gitignore
└── README.md            # Project documentation
```

### ML-Shell Template

When using `--template ml-shell`, external scripts are generated:

```
my-service/
├── main.py              # ML service with command templates
├── scripts/             # External training/prediction scripts
│   ├── train_model.py   # Training script
│   └── predict_model.py # Prediction script
├── pyproject.toml       # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── compose.yml          # Docker Compose configuration
├── data/                # Database directory
│   └── chapkit.db       # SQLite database (created at runtime)
├── .gitignore           # Python gitignore
└── README.md            # Project documentation
```

### Task Template

When using `--template task`, a task execution service is generated:

```
my-service/
├── main.py              # Task execution service with Python functions
├── pyproject.toml       # Python dependencies
├── Dockerfile           # Multi-stage Docker build
├── compose.yml          # Docker Compose configuration
├── data/                # Database directory
│   └── chapkit.db       # SQLite database (created at runtime)
├── .gitignore           # Python gitignore
└── README.md            # Project documentation
```

### With Monitoring

When using `--with-monitoring`, additional files are generated:

```
my-service/
...
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

The generated `main.py` varies by template:

**ML Template (`ml`):**
- **Config Schema**: Pydantic model for ML parameters
- **Training Function**: `on_train` with simple model example
- **Prediction Function**: `on_predict` for inference
- **Service Info**: Metadata (name, version, author, status)
- **Artifact Hierarchy**: Storage structure for models and predictions
- **FastAPI App**: Built using `MLServiceBuilder`

**ML-Shell Template (`ml-shell`):**
- Similar to ML template but references external training/prediction scripts
- **Shell Commands**: Command templates for executing scripts
- **Scripts Directory**: Contains `train_model.py` and `predict_model.py`

**Task Template (`task`):**
- **Task Functions**: Python functions registered with `@TaskRegistry.register()`
- **Task Manager**: Handles task execution with dependency injection
- **Task Router**: API endpoints for task CRUD and execution
- **FastAPI App**: Built using `ServiceBuilder` (not ML-specific)

**Example structure (ML template):**

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

## Examples

The `examples/` directory contains working examples for each template type:

**ML Template Examples:**
- `ml_functional/` - ML template with Python functions (FunctionalModelRunner)
- `ml_class/` - Class-based ML runner approach
- `quickstart/` - Complete ML service with config, artifacts, and ML endpoints

**ML-Shell Template Example:**
- `ml_shell/` - Language-agnostic ML with external scripts (ShellModelRunner)

**Task Template Example:**
- `task_execution/` - General-purpose task execution with Python functions and shell commands

**Other Examples:**
- `ml_pipeline/` - Multi-stage ML pipeline with hierarchical artifacts
- `full_featured/` - Comprehensive example with monitoring and custom routers
- `config_artifact/` - Configuration with artifact linking
- `artifact/` - Read-only artifact API
- `custom_migrations/` - Database migrations with custom models

## Related Documentation

- [ML Workflows](ml-workflows.md) - Learn about model training and prediction
- [Configuration Management](configuration-management.md) - Working with configs
- [Artifact Storage](artifact-storage.md) - Managing models and predictions
- [Task Execution](task-execution.md) - Scheduling background jobs
