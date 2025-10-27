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
- `--monitoring` - Include Prometheus and Grafana monitoring stack

This creates a ready-to-run ML service with configuration, artifacts, and ML endpoints pre-configured.

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
    .with_artifacts(hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "model"}))
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
├── config/           # Configuration module
├── ml/               # ML train/predict workflows
└── api/              # ServiceBuilder with ML integration
    └── service_builder.py  # .with_config(), .with_ml()
```

Chapkit extends servicekit's `BaseServiceBuilder` with ML-specific features and uses servicekit's artifact and task modules.

## Examples

See the `examples/` directory:

- `quickstart.py` - Complete ML service
- `config_artifact_api.py` - Config with artifact linking
- `ml_basic.py`, `ml_class.py` - ML workflow patterns

## Documentation

See `docs/` for comprehensive guides:

- ML workflow guide

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
