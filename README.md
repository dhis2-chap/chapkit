# Chapkit

[![CI](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dhis2-chap/chapkit/branch/main/graph/badge.svg)](https://codecov.io/gh/dhis2-chap/chapkit)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://dhis2-chap.github.io/chapkit/)

> ML service modules built on servicekit - config management and ML workflows

Chapkit provides ML-specific modules for building machine learning services, built on top of servicekit's artifact and task infrastructure.

## Features

- **Config Module**: Key-value configuration with JSON data and Pydantic validation
- **ML Module**: Train/predict workflows with artifact-based model storage and timing metadata
- **Config-Artifact Linking**: Connect configurations to artifact hierarchies for experiment tracking

## Installation

```bash
pip install chapkit
```

Chapkit automatically installs servicekit as a dependency.

## Quick Start

```python
from servicekit.artifact import ArtifactHierarchy
from chapkit import BaseConfig
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

Chapkit uses servicekit's artifact system for hierarchical storage:

```python
from servicekit.artifact import ArtifactHierarchy, ArtifactManager, ArtifactIn

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

@FunctionalModelRunner.train
async def train_model(config: MyConfig) -> dict:
    model = train_sklearn_model(config)
    return {"model": model, "accuracy": 0.95}

@FunctionalModelRunner.predict
async def predict(model: dict, data: pd.DataFrame) -> dict:
    predictions = model["model"].predict(data)
    return {"predictions": predictions.tolist()}

app.with_ml(FunctionalModelRunner)
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
