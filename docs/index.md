# Chapkit

[![CI](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml/badge.svg)](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dhis2-chap/chapkit/branch/main/graph/badge.svg)](https://codecov.io/gh/dhis2-chap/chapkit)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Build production-ready ML services with train/predict workflows, artifact storage, config management, and job scheduling - all in a few lines of code.

## Quick Start: ML Service

```python
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner


class MyMLConfig(BaseConfig):
    """Configuration for your ML model."""

    prediction_periods: int = 3


async def train_model(
    config: MyMLConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> dict:
    """Train your model - returns trained model object."""
    df = data.to_pandas()
    # Your training logic here - example using sklearn:
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(df[["feature1", "feature2"]], df["target"])
    return {"trained": True}


async def predict(
    config: MyMLConfig,
    model: dict,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Make predictions using the trained model."""
    future_df = future.to_pandas()
    # Your prediction logic here
    future_df["sample_0"] = 0.0  # Replace with actual predictions
    return DataFrame.from_pandas(future_df)


app = (
    MLServiceBuilder(
        info=MLServiceInfo(id="disease-prediction-service", display_name="Disease Prediction Service"),
        config_schema=MyMLConfig,
        hierarchy=ArtifactHierarchy(
            name="ml",
            level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
        ),
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
