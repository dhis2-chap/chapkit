# Chapkit

[![CI](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dhis2-chap/chapkit/branch/main/graph/badge.svg)](https://codecov.io/gh/dhis2-chap/chapkit)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://dhis2-chap.github.io/chapkit/)

> ML service modules built on servicekit - config, artifact, task, and ML workflows

Chapkit provides domain-specific modules for building machine learning services on top of servicekit's core framework. Includes artifact storage, task execution, configuration management, and ML train/predict workflows.

## Features

- **Artifact Module**: Hierarchical storage for models, data, and experiment tracking with parent-child relationships
- **Task Module**: Reusable command templates for shell and Python task execution with parameter injection
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

## CLI Usage

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
- `--with-monitoring` - Include Prometheus and Grafana monitoring stack
- `--template <type>` - Template type: `ml` (default), `ml-shell`, or `task`

This creates a ready-to-run service with configuration, artifacts, and API endpoints pre-configured.

**Template Types:**
- **ml**: Define training/prediction as Python functions in `main.py` (simpler, best for Python-only ML workflows)
- **ml-shell**: Use external scripts for training/prediction (language-agnostic, supports Python/R/Julia/etc.)
- **task**: General-purpose task execution with Python functions and shell commands (not ML-specific)

## Quick Start

```python
from chapkit import ArtifactHierarchy, BaseConfig
from chapkit.api import ServiceBuilder, ServiceInfo

class MyConfig(BaseConfig):
    model_name: str
    threshold: float

app = (
    ServiceBuilder(info=ServiceInfo(display_name="ML Service"))
    .with_health()
    .with_config(MyConfig)
    .with_artifacts(hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "ml_training", 1: "ml_prediction"}))
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
from chapkit.ml import FunctionalModelRunner
import pandas as pd

async def train_model(config: MyConfig, data: pd.DataFrame, geo=None):
    """Train your model - returns trained model object."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(data[["feature1", "feature2"]], data["target"])
    return model

async def predict(config: MyConfig, model, historic: pd.DataFrame, future: pd.DataFrame, geo=None):
    """Make predictions - returns DataFrame with predictions."""
    predictions = model.predict(future[["feature1", "feature2"]])
    future["predictions"] = predictions
    return future

# Wrap functions in runner
runner = FunctionalModelRunner(on_train=train_model, on_predict=predict)
app.with_ml(runner=runner)
```

## Architecture

```
chapkit/
├── config/           # Configuration management with Pydantic validation
├── artifact/         # Hierarchical storage for models and data
├── task/             # Reusable task templates (Python functions, shell commands)
├── ml/               # ML train/predict workflows
├── cli/              # CLI scaffolding tools
├── scheduler.py      # Job scheduling integration
└── api/              # ServiceBuilder with ML integration
    └── service_builder.py  # .with_config(), .with_artifacts(), .with_ml()
```

Chapkit extends servicekit's `BaseServiceBuilder` with ML-specific features and domain modules for configuration, artifacts, tasks, and ML workflows.

## Examples

See the `examples/` directory for complete working examples:

- `quickstart/` - Complete ML service with config, artifacts, and ML endpoints
- `config_artifact/` - Config with artifact linking
- `ml_functional/`, `ml_class/`, `ml_shell/` - ML workflow patterns (ML template, class-based, ML-shell template)
- `ml_pipeline/` - Multi-stage ML pipeline with hierarchical artifacts
- `artifact/` - Read-only artifact API with hierarchical storage
- `task_execution/` - Task execution with Python functions and shell commands
- `full_featured/` - Comprehensive example with monitoring, custom routers, and hooks
- `library_usage/` - Using chapkit as a library with custom models
- `custom_migrations/` - Database migrations with custom models

## Documentation

See `docs/guides/` for comprehensive guides:

- [ML Workflows](docs/guides/ml-workflows.md) - Train/predict patterns and model runners
- [Configuration Management](docs/guides/configuration-management.md) - Config schemas and validation
- [Artifact Storage](docs/guides/artifact-storage.md) - Hierarchical data storage for ML artifacts
- [Task Execution](docs/guides/task-execution.md) - Python functions and shell command templates
- [CLI Scaffolding](docs/guides/cli-scaffolding.md) - Project scaffolding with `chapkit init`
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
