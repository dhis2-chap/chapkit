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
- `--with-validation` - Scaffold `on_validate_train` / `on_validate_predict` stubs so the `$validate` endpoint can emit domain-specific diagnostics. Off by default.
- `--template TYPE` - Template type: `fn-py` (default), `shell-py`, or `shell-r`
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

# Create project with shell-py template (Python scripts under scripts/)
chapkit init my-service --template shell-py

# Create project with shell-r template (R scripts on chapkit-r-inla)
chapkit init my-service --template shell-r

# Create project with $validate hook stubs
chapkit init my-service --with-validation

# Combine options
chapkit init my-service --template shell-py --with-monitoring --with-validation

# From GitHub (development version)
uvx --from git+https://github.com/dhis2-chap/chapkit chapkit init my-service
```

---

### `chapkit artifact list`

List artifacts stored in a chapkit database or running service.

**Alias:** `chapkit artifact ls`

**Usage:**

```bash
chapkit artifact list [OPTIONS]
```

**Options:**

- `--database, -d PATH` - Path to SQLite database file
- `--url, -u URL` - Base URL of running chapkit service
- `--type, -t TYPE` - Filter by artifact type (e.g., `ml_training_workspace`, `ml_prediction`)
- `--help` - Show help message

**Note:** Either `--database` or `--url` must be provided (mutually exclusive).

**Examples:**

```bash
# List from local database
chapkit artifact list --database ./data/chapkit.db

# List from running service
chapkit artifact list --url http://localhost:8000

# Filter by type
chapkit artifact list --database ./data/chapkit.db --type ml_training_workspace
```

**Output:**

The output shows artifacts with hierarchy indentation (2 spaces per level) for easier navigation:

```
ID                             TYPE                      SIZE       CONFIG         CREATED
----------------------------------------------------------------------------------------------------
01ABC123456789ABCDEFGHIJ       ml_training_workspace     1.2 MB     01CFG12345..   2024-01-15 10:30
  01DEF987654321FEDCBA98       ml_prediction             45.3 KB    01CFG12345..   2024-01-15 10:35
    01GHI111222333444555       ml_prediction_workspace   2.1 MB     01CFG12345..   2024-01-15 10:36
```

- **ID**: Artifact ULID (indented by level)
- **TYPE**: Artifact type from metadata
- **SIZE**: Human-readable size
- **CONFIG**: Config ID from artifact metadata (truncated)
- **CREATED**: Timestamp in YYYY-MM-DD HH:MM format

---

### `chapkit artifact download`

Download a ZIP artifact from a chapkit database or running service.

**Usage:**

```bash
chapkit artifact download ARTIFACT_ID [OPTIONS]
```

**Arguments:**

- `ARTIFACT_ID` - Artifact ID (ULID) to download (required)

**Options:**

- `--database, -d PATH` - Path to SQLite database file
- `--url, -u URL` - Base URL of running chapkit service
- `--output, -o PATH` - Output path (default: `./<artifact_id>.zip` or `./<artifact_id>/` with `--extract`)
- `--extract, -x` - Extract ZIP contents to a directory instead of saving as file
- `--force, -f` - Overwrite existing output file or directory
- `--help` - Show help message

**Note:** Either `--database` or `--url` must be provided (mutually exclusive).

**Examples:**

```bash
# Download as ZIP file (default)
chapkit artifact download 01ABC123... --database ./data/chapkit.db
# Creates: 01ABC123....zip

# Download with custom filename
chapkit artifact download 01ABC123... --database ./data/chapkit.db -o model.zip

# Extract to directory
chapkit artifact download 01ABC123... --database ./data/chapkit.db --extract
# Creates: 01ABC123.../

# Extract to custom directory
chapkit artifact download 01ABC123... --database ./data/chapkit.db --extract -o ./workspace

# Download from running service
chapkit artifact download 01ABC123... --url http://localhost:8000

# Force overwrite existing
chapkit artifact download 01ABC123... --database ./data/chapkit.db --force
```

---

### `chapkit test`

Run end-to-end tests against your ML service. This command only appears when inside a chapkit project directory.

See [Testing ML Services](testing-ml-services.md) for full documentation.

---

## Template Types

### `fn-py` (Default)

The default template defines training and prediction as Python functions directly in `main.py` (driven by `FunctionalModelRunner`):

**Pros:**
- Simpler to understand and get started
- All code in one file
- Direct access to the Python ecosystem
- No external subprocess overhead

**Cons:**
- Python-only workflows
- Couples train and predict in the same process

**Best for:** Python-centric ML workflows, prototyping, simpler models.

### `shell-py`

Train and predict via external Python scripts in `scripts/`, driven by `ShellModelRunner`. Scripts run as subprocesses with file-based interchange (CSV, YAML, pickle) over an isolated workspace.

**Pros:**
- Workspace isolation between train and predict
- Easy to bring an existing CLI script into chapkit unchanged
- Same image as `fn-py` (chapkit-py base)

**Cons:**
- More files to manage
- File I/O conventions to learn (`config.yml`, `model.pickle`, etc.)

**Best for:** Adopting an existing Python CLI workflow into chapkit, or wanting workspace isolation per train run.

### `shell-r`

Same `ShellModelRunner` shape as `shell-py`, but with R scripts in `scripts/` (`train.R`, `predict.R`). Defaults to the [`chapkit-r-inla`](https://github.com/dhis2-chap/chapkit-images) base image (R 4.5 + INLA + spatial / time-series stack preinstalled). The Dockerfile ships a commented `chapkit-r` alternative for projects that don't need INLA (smaller image, multi-arch).

**Pros:**
- R 4.5 + INLA / spatial / time-series stack ready out of the box
- File-based interchange (CSV, YAML, RDS) — same conventions as `shell-py`
- Closest match to `chapkit_ewars_model` and other published R models

**Cons:**
- `chapkit-r-inla` is amd64-only (Apple Silicon needs Rosetta)
- ~570 MB image footprint

**Best for:** R-language epidemiology / time-series models, especially anything using INLA.

---

## Generated Project Structure

### `fn-py` (Default)

```
my-service/
├── main.py              # ML service with train/predict functions
├── pyproject.toml       # Python dependencies
├── Dockerfile           # FROM chapkit-py + uv sync
├── compose.yml          # Docker Compose configuration
├── data/                # Database directory
│   └── chapkit.db       # SQLite database (created at runtime)
├── .gitignore           # Python gitignore
└── README.md            # Project documentation
```

### `shell-py`

When using `--template shell-py`, external Python scripts are generated under `scripts/`:

```
my-service/
├── main.py              # ML service with command templates
├── scripts/             # External training/prediction scripts (Python)
│   ├── train_model.py   # Training script
│   └── predict_model.py # Prediction script
├── pyproject.toml       # Python dependencies
├── Dockerfile           # FROM chapkit-py + uv sync
├── compose.yml          # Docker Compose configuration
├── data/                # Database directory
│   └── chapkit.db       # SQLite database (created at runtime)
├── .gitignore           # Python gitignore
└── README.md            # Project documentation
```

### `shell-r`

When using `--template shell-r`, external R scripts are generated under `scripts/`. The Dockerfile defaults to `chapkit-r-inla` (with a commented `chapkit-r` alternative for non-INLA projects):

```
my-service/
├── main.py              # ML service with Rscript-based command templates
├── scripts/             # External training/prediction scripts (R)
│   ├── train.R          # Training script
│   └── predict.R        # Prediction script
├── pyproject.toml       # Python deps for the service layer (chapkit only by default)
├── Dockerfile           # FROM chapkit-r-inla + uv sync
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

### With Validation Hooks

Applies to all three templates (`fn-py`, `shell-py`, `shell-r`).

When using `--with-validation` with an ML template, the generated `main.py`
gains two extra async functions and wires them into the runner:

- `on_validate_train(config, data, geo)` - runs on `POST /api/v1/ml/$validate` with `{"type": "train", ...}`
- `on_validate_predict(config, historic, future, geo)` - runs on `POST /api/v1/ml/$validate` with `{"type": "predict", ...}`

Both return `list[ValidationDiagnostic]`. Framework-level checks (config
exists, `prediction_periods` bounds, empty data, failed training artifact)
are run by chapkit before your hook; your hook is only invoked when no
framework diagnostic has `severity="error"`, so you do not need to
defensively re-check things chapkit has already covered. Use the hooks for
**domain** checks chapkit cannot know about - for example comparing
`config.n_lags` to `len(historic)`, or checking that required covariate
columns are present.

Omit `--with-validation` (default) if you do not need domain checks: the
`$validate` endpoint still works, it just returns only the framework-level
diagnostics. You can add hooks later by hand.

See [ML Workflows: `$validate`](ml-workflows.md#post-apiv1mlvalidate) for the
full endpoint reference and the `ValidationDiagnostic` schema.

---

## Generated Files

### main.py

The generated `main.py` varies by template:

**`fn-py` (default):**
- **Config Schema**: Pydantic model for ML parameters
- **Training Function**: `on_train` with simple model example
- **Prediction Function**: `on_predict` for inference
- **Service Info**: Metadata (name, version, author, status)
- **Artifact Hierarchy**: Storage structure for models and predictions
- **FastAPI App**: Built using `MLServiceBuilder` with a `FunctionalModelRunner`

**`shell-py`:**
- Same `MLServiceBuilder` shape as `fn-py` but uses `ShellModelRunner`
- **Shell Commands**: `train_command` / `predict_command` strings invoking `python scripts/train_model.py …`
- **Scripts Directory**: `scripts/train_model.py` and `scripts/predict_model.py`

**`shell-r`:**
- Same `ShellModelRunner` shape as `shell-py` but the commands invoke `Rscript scripts/train.R …`
- **Scripts Directory**: `scripts/train.R` and `scripts/predict.R`
- **Dockerfile**: defaults to `chapkit-r-inla` (with a commented `chapkit-r` alternative for non-INLA models)

**Example structure (`fn-py` template):**

```python
class MyServiceConfig(BaseConfig):
    # Add your model parameters here
    prediction_periods: int = 3

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
    prediction_periods: int = 3
```

### Add Dependencies

Use `uv` to add packages:

```bash
uv add scikit-learn pandas numpy
```

### Customize Service Metadata

Update the `MLServiceInfo`:

```python
from chapkit.api import ModelMetadata, PeriodType

info = MLServiceInfo(
    id="production-model",
    display_name="Production Model",
    version="2.0.0",
    description="Detailed description here",
    model_metadata=ModelMetadata(
        author="Your Team",
        author_assessed_status=AssessedStatus.green,
        contact_email="team@example.com",
    ),
    period_type=PeriodType.monthly,
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

### Docker Data Management

The generated `Dockerfile` and `compose.yml` are **starting points** designed to work out of the box. Customize them for your specific deployment needs.

The following describes the default configuration. If you change `DATABASE_URL` or other settings, your setup may differ.

**Named Volumes**

Docker Compose uses named volumes (not bind mounts) for data persistence:

```yaml
volumes:
  - ck_my_service_data:/workspace/data
```

This approach:

- Works consistently across macOS, Linux, and Windows
- Avoids UID permission issues on Linux
- Data persists across container restarts

**Accessing Data**

```bash
# List files in data directory
docker compose exec api ls /workspace/data

# Copy database out of container
docker compose cp api:/workspace/data/chapkit.db ./backup.db

# Copy database into container
docker compose cp ./mydata.db api:/workspace/data/chapkit.db

# Direct SQLite access
docker compose exec api sqlite3 /workspace/data/chapkit.db ".tables"
```

**Volume Management**

```bash
# List all Docker volumes
docker volume ls

# Inspect volume details
docker volume inspect ck_my_service_data

# Remove containers but keep data
docker compose down

# Remove containers AND data (warning: data loss)
docker compose down -v
```

**Using Bind Mounts**

If you need direct host filesystem access, modify `compose.yml` to use a bind mount:

```yaml
volumes:
  - ./data:/workspace/data  # Host path:container path
```

Note: On Linux, ensure the host directory has correct permissions for the container user (UID 10001).

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
7. **Deploy**: See [Deploying to chap-core](deploying-to-chap-core.md) for the end-to-end walkthrough from a scaffolded project to a model registered with chap-core and visible in the DHIS2 Modeling App.

## Examples

The `examples/` directory contains pattern-focused examples (use `chapkit init` for a fresh starter project):

**ML Workflow Patterns:**
- `ml_functional/` - `FunctionalModelRunner` (matches `--template fn-py`)
- `ml_class/` - Class-based `BaseModelRunner` subclass
- `ml_shell/` - `ShellModelRunner` with external Python scripts (matches `--template shell-py`)

**Other Patterns:**
- `ml_pipeline/` - Multi-stage ML pipeline with hierarchical artifacts
- `config_artifact/` - Configuration with artifact linking
- `config/` - Config CRUD walkthrough
- `artifact/` - Read-only artifact API
- `library_usage/` - Using chapkit as a library with custom models
- `dataframe_usage/` - Working with `chapkit.data.DataFrame`

## Related Documentation

- [Deploying to chap-core](deploying-to-chap-core.md) - End-to-end deploy from scaffold to DHIS2
- [ML Workflows](ml-workflows.md) - Learn about model training and prediction
- [Configuration Management](configuration-management.md) - Working with configs
- [Artifact Storage](artifact-storage.md) - Managing models and predictions
