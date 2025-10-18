# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Documentation Standards

**IMPORTANT: All code must follow these documentation requirements:**

- **Every Python file**: One-line module docstring at top
- **Every class**: One-line docstring
- **Every method/function**: One-line docstring
- **Format**: Use triple quotes `"""docstring"""`
- **Style**: Keep concise - one line preferred

**Example:**
```python
"""Module for handling user authentication."""

class AuthManager:
    """Manages user authentication and authorization."""

    def verify_token(self, token: str) -> bool:
        """Verify JWT token validity."""
        ...
```

## Git Workflow

**Branch + PR workflow is highly recommended. Ask user before creating branches/PRs.**

**Branch naming:**
- `feat/*` - New features (aligns with `feat:` commits)
- `fix/*` - Bug fixes (aligns with `fix:` commits)
- `refactor/*` - Code refactoring (aligns with `refactor:` commits)
- `docs/*` - Documentation changes (aligns with `docs:` commits)
- `test/*` - Test additions/corrections (aligns with `test:` commits)
- `chore/*` - Dependencies, tooling, maintenance (aligns with `chore:` commits)

**Process:**
1. **Ask user** if they want a branch + PR for the change
2. Create branch from `main`: `git checkout -b feat/my-feature`
3. Make changes and commit: `git commit -m "feat: add new feature"`
4. Push: `git push -u origin feat/my-feature`
5. Create PR: `gh pr create --title "..." --body "..."`
6. Wait for manual review and merge

**Commit message prefixes:** `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`

**Commit message requirements:**
- NEVER include "Co-Authored-By: Claude" or similar AI attribution
- Keep messages concise and descriptive
- Focus on what changed and why

**PR requirements:**
- All tests must pass (`make test`)
- All linting must pass (`make lint`)
- Code coverage should not decrease
- Descriptive PR title and body

## Project Overview

`chapkit` is a collection of domain-specific modules for ML and data services, built on top of the [servicekit](https://github.com/winterop-com/servicekit) framework. It provides config management, artifact storage, task execution, and ML workflows.

**Primary Modules:** Config (key-value with JSON), Artifact (hierarchical trees), Task (script execution with output capture), ML (train/predict operations with artifact-based model storage)

## Architecture

```
chapkit/
├── config/              # Configuration management
│   ├── models.py        # Config, ConfigArtifact ORM models
│   ├── schemas.py       # BaseConfig, ConfigIn, ConfigOut
│   ├── repository.py    # ConfigRepository with artifact linking
│   ├── manager.py       # ConfigManager
│   └── router.py        # ConfigRouter with artifact operations
├── artifact/            # Hierarchical artifact storage
│   ├── models.py        # Artifact ORM model
│   ├── schemas.py       # ArtifactIn, ArtifactOut, ArtifactTreeNode, ArtifactHierarchy
│   ├── repository.py    # ArtifactRepository
│   ├── manager.py       # ArtifactManager
│   └── router.py        # ArtifactRouter
├── task/                # Task execution system
│   ├── models.py        # Task ORM model
│   ├── schemas.py       # TaskIn, TaskOut
│   ├── repository.py    # TaskRepository
│   ├── manager.py       # TaskManager with Python/shell execution
│   ├── router.py        # TaskRouter
│   └── registry.py      # TaskRegistry for Python functions
├── ml/                  # ML train/predict workflows
│   ├── manager.py       # MLManager
│   ├── router.py        # MLRouter
│   ├── runner.py        # FunctionalModelRunner, ClassModelRunner
│   └── schemas.py       # TrainRequest, PredictRequest, etc.
└── api/                 # Application orchestration
    ├── service_builder.py   # ServiceBuilder, MLServiceBuilder
    └── dependencies.py      # get_config_manager, get_artifact_manager, etc.
```

**Dependencies:**
- **servicekit** - Core framework (Database, Repository, Manager, Router, FastAPI utilities)
- Modules import from servicekit for base classes
- ServiceBuilder orchestrates all modules into a complete FastAPI application

**Layer Rules:**
- Modules import from servicekit but not from each other (except artifact/config linking)
- api/ imports from both servicekit and chapkit modules

## Quick Start

```python
from chapkit import BaseConfig, ArtifactHierarchy
from chapkit.api import ServiceBuilder, ServiceInfo

class MyConfig(BaseConfig):
    host: str
    port: int

app = (
    ServiceBuilder(info=ServiceInfo(display_name="My Service"))
    .with_health()
    .with_config(MyConfig)
    .with_artifacts(hierarchy=ArtifactHierarchy(name="ml", level_labels={0: "model"}))
    .build()
)
```

**Run:** `fastapi dev your_file.py`

## ServiceBuilder API

**ServiceBuilder** (in `chapkit.api`): Extends servicekit's BaseServiceBuilder with chapkit modules

**Key methods:**
- `.with_health()` - Health check endpoint (from servicekit)
- `.with_system()` - System info endpoint (from servicekit)
- `.with_config(schema)` - Config CRUD endpoints at `/api/v1/configs`
- `.with_artifacts(hierarchy)` - Artifact CRUD at `/api/v1/artifacts`
- `.with_jobs()` - Job scheduler at `/api/v1/jobs` (from servicekit)
- `.with_tasks(validate_on_startup=True)` - Task execution at `/api/v1/tasks`
- `.with_ml(runner)` - ML train/predict at `/api/v1/ml`
- `.build()` - Returns FastAPI app

**MLServiceBuilder**: Specialized builder that bundles health, config, artifacts, jobs, ml

## Task Execution System

Chapkit provides a task execution system supporting both shell commands and Python functions with type-based dependency injection.

**Task Types:**
- **Shell tasks**: Execute commands via asyncio subprocess, capture stdout/stderr/exit_code
- **Python tasks**: Execute registered functions via TaskRegistry, capture result/error with traceback

**Python Task Registration:**
```python
from chapkit import TaskRegistry
from sqlalchemy.ext.asyncio import AsyncSession

@TaskRegistry.register("my_task")
async def my_task(name: str, session: AsyncSession) -> dict:
    """Task with user parameters and dependency injection."""
    # name comes from task.parameters (user-provided)
    # session is injected by framework (type-based)
    return {"status": "success", "name": name}
```

**Type-Based Dependency Injection:**

Framework types are automatically injected based on function parameter type hints:
- `AsyncSession` - SQLAlchemy async database session
- `Database` - servicekit Database instance
- `ArtifactManager` - chapkit Artifact management service
- `JobScheduler` - servicekit Job scheduling service

**Key Features:**
- Enable/disable controls for tasks
- Automatic orphaned task validation (enabled by default, auto-disables tasks with missing functions on startup)
- Support both sync and async Python functions
- Mix user parameters with framework injections
- Optional type support (`AsyncSession | None`)
- Artifact-based execution results for both shell and Python tasks

See `examples/python_task_execution_api.py` for working examples.

## Common Endpoints

**Config Service:** CRUD operations, artifact linking (`/$link-artifact`, `/$unlink-artifact`, `/$artifacts`)
**Artifact Service:** CRUD + tree operations (`/$tree`, `/$expand`)
**Task Service:** CRUD, execute (`/$execute`), enable/disable controls
**ML Service:** Train (`/$train`) and predict (`/$predict`) operations

**Operation prefix:** `$` indicates operations (computed/derived data) vs resource access

## Database & Migrations

Chapkit uses servicekit's database infrastructure (SqliteDatabase, migrations via Alembic).

**Commands:**
```bash
make migrate MSG='description'  # Generate migration
make upgrade                    # Apply migrations (auto-applied on init)
```

**Workflow:**
1. Modify ORM models in `src/chapkit/{module}/models.py`
2. Generate: `make migrate MSG='description'`
3. Review in `alembic/versions/`
4. Restart app (auto-applies)
5. Commit migration file

## Naming Conventions

**IMPORTANT: Always use full descriptive names, never abbreviations**

- ✅ `self.repository` (not `self.repo`)
- ✅ `config_repository` (not `config_repo`)
- ✅ `artifact_repository` (not `artifact_repo`)

This applies to:
- Class attributes
- Local variables
- Function parameters
- Any other code references

**Rationale:** Full names improve code readability and maintainability.

## Code Quality

**Standards:**
- Python 3.13+, line length 120, type annotations required
- Double quotes, async/await, conventional commits
- Class order: public → protected → private
- `__all__` declarations only in `__init__.py` files

**Documentation Requirements:**
- Every Python file: one-line module docstring at top
- Every class: one-line docstring
- Every method/function: one-line docstring
- Use triple quotes `"""docstring"""`
- Keep concise - one line preferred

**Testing:**
```bash
make test      # Fast tests
make coverage  # With coverage
make lint      # Linting
```

**Always run `make lint` and `make test` after changes**

## Common Patterns

**Repository naming:**
- `find_*`: Single entity or None
- `find_all_*`: Sequence
- `exists_*`: Boolean
- `count`: Integer

**Manager vs Repository:**
- Repository: Low-level ORM data access (from servicekit.repository.BaseRepository)
- Manager: Pydantic validation + business logic (from servicekit.manager.BaseManager)

## Dependency Management

**Always use `uv`:**
```bash
uv add <package>          # Runtime dependency
uv add --dev <package>    # Dev dependency
uv add <package>@latest   # Update specific
uv lock --upgrade         # Update all
```

**Never manually edit `pyproject.toml`**

## Key Dependencies

- servicekit - Core framework foundation
- sqlalchemy[asyncio] >= 2.0
- aiosqlite >= 0.21
- pydantic >= 2.11
- fastapi, ulid-py

## Additional Resources

- Full examples: `examples/` directory
- servicekit docs: https://winterop-com.github.io/servicekit
- Chapkit docs: https://dhis2-chap.github.io/chapkit
