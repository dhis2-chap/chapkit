# Chapkit

[![CI](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/chapkit)](https://pypi.org/project/chapkit/)
[![codecov](https://codecov.io/gh/dhis2-chap/chapkit/branch/main/graph/badge.svg)](https://codecov.io/gh/dhis2-chap/chapkit)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://dhis2-chap.github.io/chapkit/)

> ML service modules built on servicekit - config, artifact, and ML workflows

Chapkit provides domain-specific modules for building machine learning services on top of servicekit's core framework. Includes artifact storage, configuration management, and ML train/predict workflows.

## Features

- **Artifact Module**: Hierarchical storage for models, data, and experiment tracking with parent-child relationships
- **Config Module**: Key-value configuration with JSON data and Pydantic validation
- **ML Module**: Train/predict workflows with artifact-based model storage and timing metadata
- **Config-Artifact Linking**: Connect configurations to artifact hierarchies for experiment tracking

## Installation

Using uv (recommended):

```bash
uv add chapkit
```

Or using pip:

```bash
pip install chapkit
```

Chapkit automatically installs servicekit as a dependency.

### Optional Dependencies

For DataFrame conversions to/from pandas, polars, or xarray, install the extras you need:

```bash
uv add chapkit[pandas]    # pandas support
uv add chapkit[polars]    # polars support
uv add chapkit[xarray]    # xarray support (includes pandas)
uv add chapkit[dataframe] # all of the above
```

## CLI Usage

### `chapkit init` - Scaffold a new project

Quickly scaffold a new ML service project using `uvx`:

```bash
uvx chapkit init <project-name>
```

Example:
```bash
uvx chapkit init my-ml-service
```

Options:
- `--path <directory>` - Target directory (default: current directory)
- `--template <type>` - Template type: `fn-py` (default), `shell-py`, or `shell-r`
- `--with-validation` - Scaffold `on_validate_train` / `on_validate_predict` stubs so the `$validate` endpoint can emit domain-specific diagnostics. Off by default.

The scaffolded service exposes `/metrics` (Prometheus format) out of the box. To layer Prometheus + Grafana around it, see the [Monitoring guide](docs/guides/monitoring.md).

This creates a ready-to-run service with configuration, artifacts, and API endpoints pre-configured.

**Template Types:**
- **fn-py**: Define training/prediction as Python functions in `main.py` (simplest path, Python-only ML workflows)
- **shell-py**: Train/predict via external Python scripts in `scripts/` (driven by `ShellModelRunner`)
- **shell-r**: Train/predict via external R scripts in `scripts/`, defaults to the `chapkit-r-inla` base image

### `chapkit mlproject run` - Serve an existing MLproject

If you already have an MLflow-style `MLproject` directory (R, Python, or mixed), `chapkit mlproject run` stands it up as a chapkit service with no code generation:

```bash
chapkit mlproject run              # serve the MLproject in the current directory
chapkit mlproject run .            # same
chapkit mlproject run /path/to/mlproject
```

### `chapkit mlproject migrate` - Adopt an existing MLproject as a chapkit project

When you're ready to own the service code (commit it, containerise it, extend it with validation hooks), `chapkit mlproject migrate` generates `main.py`, a `Dockerfile` pointing at the right [chapkit-images](https://github.com/dhis2-chap/chapkit-images) base, a `pyproject.toml`, a `compose.yml`, and a `CHAPKIT.md`. Chaff (input data, ad-hoc runners, the `MLproject` file itself) is swept to `_old/`; your train/predict scripts stay where they are:

```bash
cd /path/to/your/mlproject
chapkit mlproject migrate --dry-run   # preview
chapkit mlproject migrate             # execute interactively
chapkit mlproject migrate --yes       # non-interactive (scripts / CI)
```

See the [MLproject Migrate guide](docs/guides/mlproject-migrate.md) for the classification table, base-image detection, and deferred features.

Or use the published `-cli` container images (no local chapkit install needed):

```bash
docker run --rm -p 8000:8000 -v "$(pwd):/work" ghcr.io/dhis2-chap/chapkit-py-cli:latest    # Python model
docker run --rm -p 8000:8000 -v "$(pwd):/work" ghcr.io/dhis2-chap/chapkit-r-cli:latest     # R model (no INLA)
docker run --rm -p 8000:8000 --platform=linux/amd64 \
    -v "$(pwd):/work" ghcr.io/dhis2-chap/chapkit-r-inla-cli:latest                          # R + INLA
```

The same `-cli` images can run `chapkit mlproject migrate` or any other chapkit CLI subcommand without a local install:

```bash
docker run --rm -v "$(pwd):/work" ghcr.io/dhis2-chap/chapkit-py-cli:latest \
    chapkit mlproject migrate . --yes
```

The unsuffixed images (`chapkit-py`, `chapkit-r`, `chapkit-r-inla`) are runtime-only bases without chapkit pre-installed — that's what `chapkit init` and `chapkit mlproject migrate` use as the `FROM` line in the Dockerfiles they generate, where `uv sync` then installs chapkit from the project's `pyproject.toml`. See the [MLproject Runner guide](docs/guides/mlproject-runner.md) for canonical parameter mapping, `user_options` -> dynamic config, env hints, and compose integration with chap-core.

## Quick Start

```python
from chapkit import ArtifactHierarchy, BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo

class MyConfig(BaseConfig):
    model_name: str
    threshold: float
    prediction_periods: int = 3

app = (
    ServiceBuilder(info=ServiceInfo(id="ml-service", display_name="ML Service"))
    .with_health()
    .with_config(MyConfig)
    .with_artifacts(hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "ml_training_workspace", 1: "ml_prediction"}))
    .with_jobs()
    .build()
)
```

## Modules

### Config

Key-value configuration storage with Pydantic schema validation:

```python
from chapkit import BaseConfig, ConfigManager

class AppConfig(BaseConfig):
    api_url: str
    timeout: int = 30
    prediction_periods: int = 3

# Automatic validation and CRUD endpoints
app.with_config(AppConfig)
```

### Artifacts

Hierarchical storage for models, data, and experiment tracking:

```python
from chapkit import ArtifactHierarchy, ArtifactManager, ArtifactIn

hierarchy = ArtifactHierarchy(
    name="ml_pipeline",
    level_labels={0: "experiment", 1: "model", 2: "evaluation"}
)

# Store pandas DataFrames, models, any Python object
artifact = await artifact_manager.save(
    ArtifactIn(data=trained_model, parent_id=experiment_id)
)
```

### ML

Train and predict workflows with automatic model storage:

```python
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner


async def train_model(config: MyConfig, data: DataFrame, geo=None) -> dict:
    """Train your model - returns trained model object."""
    df = data.to_pandas()
    # Your training logic here
    return {"trained": True}


async def predict(config: MyConfig, model: dict, historic: DataFrame, future: DataFrame, geo=None) -> DataFrame:
    """Make predictions - returns DataFrame with predictions."""
    future_df = future.to_pandas()
    future_df["sample_0"] = 0.0  # Your predictions here
    return DataFrame.from_pandas(future_df)


# Wrap functions in runner
runner = FunctionalModelRunner(on_train=train_model, on_predict=predict)
app.with_ml(runner=runner)
```

## Architecture

```
chapkit/
├── config/           # Configuration management with Pydantic validation
├── artifact/         # Hierarchical storage for models and data
├── ml/               # ML train/predict workflows
├── cli/              # CLI scaffolding tools
├── scheduler.py      # Job scheduling integration
└── api/              # ServiceBuilder with ML integration
    └── service_builder.py  # .with_config(), .with_artifacts(), .with_ml()
```

Chapkit extends servicekit's `BaseServiceBuilder` with ML-specific features and domain modules for configuration, artifacts, and ML workflows.

## Examples

See the `examples/` directory for complete working examples:

- `config/` - Config CRUD walkthrough
- `config_artifact/` - Config with artifact linking
- `artifact/` - Read-only artifact API with hierarchical storage
- `ml_functional/`, `ml_class/`, `ml_shell/` - ML workflow patterns (`FunctionalModelRunner`, class-based `BaseModelRunner`, `ShellModelRunner`)
- `ml_pipeline/` - Multi-stage ML pipeline with hierarchical artifacts
- `library_usage/` - Using chapkit as a library with custom models
- `dataframe_usage/` - Working with `chapkit.data.DataFrame`

For a fresh project, prefer `chapkit init` (see [`docs/guides/cli-scaffolding.md`](docs/guides/cli-scaffolding.md)) — the `examples/` directory targets specific patterns rather than a full starting point.

## Documentation

See `docs/guides/` for comprehensive guides:

- [R Quickstart](docs/guides/r-quickstart.md) - 10-minute path from R model to running chapkit service
- [ML Workflows](docs/guides/ml-workflows.md) - Train/predict patterns and model runners
- [Configuration Management](docs/guides/configuration-management.md) - Config schemas and validation
- [Artifact Storage](docs/guides/artifact-storage.md) - Hierarchical data storage for ML artifacts
- [CLI Scaffolding](docs/guides/cli-scaffolding.md) - Project scaffolding with `chapkit init`
- [Shell-runner Contract](docs/guides/shell-runner-contract.md) - Exact file-by-file lifecycle of train and predict workspaces
- [Monitoring](docs/guides/monitoring.md) - Adding Prometheus + Grafana around the scaffolded `/metrics` endpoint
- [MLproject Runner](docs/guides/mlproject-runner.md) - Serve existing MLproject directories with `chapkit mlproject run`
- [MLproject Migrate](docs/guides/mlproject-migrate.md) - Adopt an MLproject as a chapkit project with `chapkit mlproject migrate`
- [Database Migrations](docs/guides/database-migrations.md) - Custom models and Alembic migrations

Full documentation: https://dhis2-chap.github.io/chapkit/

## Testing

```bash
make test      # Run tests
make lint      # Run linter
make coverage  # Test coverage
```

## License

AGPL-3.0-or-later

## Related Projects

- **[servicekit](https://github.com/winterop-com/servicekit)** - Core framework foundation (FastAPI, SQLAlchemy, CRUD, auth, etc.) ([docs](https://winterop-com.github.io/servicekit))
- **[chapkit-images](https://github.com/dhis2-chap/chapkit-images)** - Dockerfiles and CI for the `chapkit-py`, `chapkit-r`, and `chapkit-r-inla` runtime images used by `chapkit mlproject run`.
