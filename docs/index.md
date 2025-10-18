# Chapkit

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/dhis2-chap/chapkit/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/badge/coverage-83%25-brightgreen)](https://codecov.io/gh/dhis2-chap/chapkit)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

ML and data service modules built on servicekit - config, artifacts, tasks, and ML workflows.

## Quick Start

```python
from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit import ArtifactHierarchy, BaseConfig

class MyConfig(BaseConfig):
    model_name: str
    threshold: float

app = (
    ServiceBuilder(info=ServiceInfo(display_name="ML Service"))
    .with_health()
    .with_config(MyConfig)
    .with_artifacts(hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "model"}))
    .with_tasks()
    .build()
)
```

Run with: `fastapi dev your_file.py`

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
