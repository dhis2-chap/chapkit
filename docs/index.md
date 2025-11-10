# Chapkit

[![CI](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dhis2-chap/chapkit/branch/main/graph/badge.svg)](https://codecov.io/gh/dhis2-chap/chapkit)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Build production-ready ML services with train/predict workflows, artifact storage, config management, and job scheduling - all in a few lines of code.

## Quick Start: ML Service

```python
from chapkit import BaseConfig
from chapkit.api import MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import FunctionalModelRunner
import pandas as pd

class MyMLConfig(BaseConfig):
    """Configuration for your ML model."""

async def train_model(config, data, geo=None):
    """Train your model - returns trained model object."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(data[["feature1", "feature2"]], data["target"])
    return model

async def predict(config, model, historic, future, geo=None):
    """Make predictions using the trained model."""
    predictions = model.predict(future[["feature1", "feature2"]])
    future["predictions"] = predictions
    return future

# Build complete ML service with one builder
app = (
    MLServiceBuilder(
        info=MLServiceInfo(display_name="Disease Prediction Service"),
        config_schema=MyMLConfig,
        hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "ml_training", 1: "ml_prediction"}),
        runner=FunctionalModelRunner(on_train=train_model, on_predict=predict),
    )
    .with_monitoring()  # Optional: Add Prometheus metrics
    .build()
)
```

**What you get:**
- `POST /api/v1/ml/train` - Train models with versioning
- `POST /api/v1/ml/predict` - Make predictions
- `GET /api/v1/configs` - Manage model configurations
- `GET /api/v1/artifacts` - Browse trained models and predictions
- `GET /api/v1/jobs` - Monitor training/prediction jobs
- `GET /health` - Health checks
- `GET /metrics` - Prometheus metrics (with `.with_monitoring()`)

Run with: `fastapi dev your_file.py` → Service ready at `http://localhost:8000`

## Installation

```bash
uv add chapkit
```

Chapkit automatically installs servicekit as a dependency.

## Links

- [Repository](https://github.com/dhis2-chap/chapkit)
- [Issues](https://github.com/dhis2-chap/chapkit/issues)
- [Servicekit](https://github.com/winterop-com/servicekit) - Core framework foundation ([docs](https://winterop-com.github.io/servicekit))

## License

AGPL-3.0-or-later
