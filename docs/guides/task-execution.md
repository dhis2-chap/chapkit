# Task Execution

Chapkit provides a registry-based task execution system for running Python functions synchronously with dependency injection and tag-based organization.

## Quick Start

### Basic Task Registration

```python
from chapkit.task import TaskRegistry, TaskExecutor, TaskRouter
from chapkit.api import ServiceBuilder, ServiceInfo
from servicekit import Database
from servicekit.api.dependencies import get_database
from fastapi import Depends

# Clear registry on module reload (for development)
TaskRegistry.clear()

# Register Python task with tags
@TaskRegistry.register("greet_user", tags=["demo", "simple"])
async def greet_user(name: str = "World") -> dict[str, str]:
    """Simple task that returns a greeting."""
    return {"message": f"Hello, {name}!"}

# Task with dependency injection
@TaskRegistry.register("process_data", tags=["demo", "injection"])
async def process_data(database: Database) -> dict[str, object]:
    """Dependencies are automatically injected."""
    return {"status": "processed", "database_url": str(database.url)}

# Build service
info = ServiceInfo(id="task-service", display_name="Task Service")

def get_task_executor(database: Database = Depends(get_database)) -> TaskExecutor:
    """Provide task executor for dependency injection."""
    return TaskExecutor(database)

task_router = TaskRouter.create(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
    executor_factory=get_task_executor,
)

app = (
    ServiceBuilder(info=info)
    .with_health()
    .include_router(task_router.router)
    .build()
)
```

**Run:** `fastapi dev your_file.py`

---

## Architecture

### Task Registration

**Python Functions:**
- Registered with `@TaskRegistry.register(name, tags=[])`
- URL-safe names (alphanumeric, underscore, hyphen only)
- Automatic dependency injection
- Return dict with results
- Async or sync functions supported

### Execution Flow

```
1. Task Registered (in-memory)
   @TaskRegistry.register("task_name", tags=["tag1"])

2. Task Discovery
   GET /api/v1/tasks
   GET /api/v1/tasks/task_name

3. Task Execution
   POST /api/v1/tasks/task_name/$execute
   ├─> Dependencies injected
   ├─> Function executed synchronously
   └─> Result returned in response (200 OK)
```

---

## Core Concepts

### TaskRegistry

Global in-memory registry for Python task functions.

```python
from chapkit.task import TaskRegistry

# Register with decorator
@TaskRegistry.register("my_task", tags=["processing", "etl"])
async def my_task(param: str) -> dict[str, object]:
    """Task docstring."""
    return {"result": param.upper()}

# Or register imperatively
def another_task(x: int) -> dict[str, int]:
    """Another task."""
    return {"doubled": x * 2}

TaskRegistry.register_function("double_it", another_task, tags=["math"])

# Check registration
assert TaskRegistry.has("my_task")

# Get task metadata
info = TaskRegistry.get_info("my_task")
print(info.signature)  # (param: str) -> dict[str, object]
print(info.tags)       # ["processing", "etl"]

# List all tasks
all_tasks = TaskRegistry.list_all()  # ["my_task", "double_it"]

# Filter by tags (requires ALL tags)
math_tasks = TaskRegistry.list_by_tags(["math"])  # ["double_it"]
```

**Rules:**
- Task names must be URL-safe: `^[a-zA-Z0-9_-]+$`
- Task names must be unique
- Functions should return dict or None
- Both async and sync functions supported
- Parameters can have defaults

### Tags

Tasks can be tagged for organization:

```python
@TaskRegistry.register("extract_data", tags=["data", "etl", "extract"])
async def extract_data() -> dict:
    """Extract data from source."""
    return {"records": 100}

@TaskRegistry.register("transform_data", tags=["data", "etl", "transform"])
async def transform_data() -> dict:
    """Transform extracted data."""
    return {"transformed": True}

# Filter tasks that have ALL specified tags
etl_tasks = TaskRegistry.list_by_tags(["data", "etl"])
# Returns: ["extract_data", "transform_data"]

extract_tasks = TaskRegistry.list_by_tags(["etl", "extract"])
# Returns: ["extract_data"]
```

### Dependency Injection

Tasks can request framework dependencies as function parameters:

```python
from servicekit import Database
from sqlalchemy.ext.asyncio import AsyncSession

@TaskRegistry.register("with_dependencies")
async def with_dependencies(
    database: Database,
    session: AsyncSession,
    custom_param: str = "default"
) -> dict[str, object]:
    """Dependencies automatically injected at runtime."""
    # Framework types are injected, user params come from request
    return {"database_url": str(database.url), "custom_param": custom_param}
```

**Available Injectable Types:**
- `AsyncSession` - Database session (always available)
- `Database` - Database instance (always available)
- `ChapkitScheduler` - Job scheduler (available when `.with_jobs()` is configured)
- `ArtifactManager` - Artifact manager (available when `.with_artifacts()` is configured and passed to TaskExecutor)

**Note:** Parameters are categorized automatically:
- Framework types (in INJECTABLE_TYPES) are injected when available
- All other parameters must be provided in execution request

**Simple setup (default):**
```python
def get_task_executor(database: Database = Depends(get_database)) -> TaskExecutor:
    return TaskExecutor(database)
```

**Advanced setup (with scheduler + artifacts):**
```python
from servicekit.api.dependencies import get_scheduler
from chapkit.artifact import ArtifactHierarchy, ArtifactManager, ArtifactRepository

TASK_HIERARCHY = ArtifactHierarchy(name="task_results", level_labels={0: "task_run"})

async def get_task_executor(
    database: Database = Depends(get_database),
    scheduler = Depends(get_scheduler),
) -> TaskExecutor:
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo, hierarchy=TASK_HIERARCHY)
        return TaskExecutor(database, scheduler, artifact_manager)

# Also add to service builder:
app = (
    ServiceBuilder(info=info)
    .with_jobs(max_concurrency=5)
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .include_router(task_router.router)
    .build()
)
```

### TaskInfo Schema

When you retrieve task metadata, you get a TaskInfo object:

```python
class TaskInfo(BaseModel):
    name: str                        # URL-safe task name
    docstring: str | None            # Function docstring
    signature: str                   # Function signature string
    parameters: list[ParameterInfo]  # Parameter metadata
    tags: list[str]                  # Task tags

class ParameterInfo(BaseModel):
    name: str                 # Parameter name
    annotation: str | None    # Type annotation as string
    default: str | None       # Default value as string
    required: bool            # Whether parameter is required
```

---

## API Endpoints

### GET /api/v1/tasks

List all registered tasks with metadata.

**Response:**
```json
[
  {
    "name": "greet_user",
    "docstring": "Simple task that returns a greeting.",
    "signature": "(name: str = 'World') -> dict[str, str]",
    "parameters": [
      {
        "name": "name",
        "annotation": "<class 'str'>",
        "default": "'World'",
        "required": false
      }
    ],
    "tags": ["demo", "simple"]
  }
]
```

**Example:**
```bash
# List all tasks
curl http://localhost:8000/api/v1/tasks
```

### GET /api/v1/tasks/{name}

Get task metadata by URL-safe name.

**Response:**
```json
{
  "name": "greet_user",
  "docstring": "Simple task that returns a greeting.",
  "signature": "(name: str = 'World') -> dict[str, str]",
  "parameters": [
    {
      "name": "name",
      "annotation": "<class 'str'>",
      "default": "'World'",
      "required": false
    }
  ],
  "tags": ["demo", "simple"]
}
```

**Errors:**
- 404 Not Found: Task not registered

### POST /api/v1/tasks/{name}/$execute

Execute task by name with runtime parameters.

**Request:**
```json
{
  "params": {
    "name": "Alice"
  }
}
```

**Response (200 OK):**
```json
{
  "task_name": "greet_user",
  "params": {"name": "Alice"},
  "result": {"message": "Hello, Alice!"},
  "error": null
}
```

**Response on Error (200 OK):**
```json
{
  "task_name": "greet_user",
  "params": {"name": "Alice"},
  "result": null,
  "error": {
    "type": "ValueError",
    "message": "Invalid parameter",
    "traceback": "Traceback (most recent call last)..."
  }
}
```

**Errors:**
- 404 Not Found: Task not registered

**Examples:**
```bash
# Execute without parameters
curl -X POST http://localhost:8000/api/v1/tasks/greet_user/\$execute \
  -H "Content-Type: application/json" \
  -d '{}'

# Execute with parameters
curl -X POST http://localhost:8000/api/v1/tasks/greet_user/\$execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"name": "Bob"}}'

# Execute task with multiple parameters
curl -X POST http://localhost:8000/api/v1/tasks/multiply_numbers/\$execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"a": 6, "b": 7}}'
```

---

## Task Patterns

### Simple Task

```python
@TaskRegistry.register("hello", tags=["demo"])
async def hello() -> dict[str, str]:
    """Simple hello world task."""
    return {"message": "Hello!"}
```

### Task with Parameters

```python
@TaskRegistry.register("add", tags=["math"])
async def add(a: int, b: int) -> dict[str, int]:
    """Add two numbers."""
    return {"result": a + b}

# Execute:
# POST /api/v1/tasks/add/$execute
# {"params": {"a": 5, "b": 3}}
```

### Task with Optional Parameters

```python
@TaskRegistry.register("greet", tags=["demo"])
async def greet(name: str = "World", greeting: str = "Hello") -> dict[str, str]:
    """Greet someone."""
    return {"message": f"{greeting}, {name}!"}

# Execute with defaults:
# POST /api/v1/tasks/greet/$execute
# {}

# Execute with custom values:
# POST /api/v1/tasks/greet/$execute
# {"params": {"name": "Alice", "greeting": "Hi"}}
```

### Task with Dependency Injection

```python
@TaskRegistry.register("store_result", tags=["storage"])
async def store_result(
    artifact_manager: ArtifactManager,
    data: dict
) -> dict[str, object]:
    """Store result in artifact."""
    artifact = await artifact_manager.save(ArtifactIn(data=data))
    return {"artifact_id": str(artifact.id)}

# Execute:
# POST /api/v1/tasks/store_result/$execute
# {"params": {"data": {"key": "value"}}}
# Note: artifact_manager is injected, only data needs to be provided
```

### Database Query Task

```python
@TaskRegistry.register("count_users", tags=["database", "reporting"])
async def count_users(database: Database) -> dict[str, int]:
    """Count users in database."""
    async with database.session() as session:
        from sqlalchemy import select, func
        from myapp.models import User

        stmt = select(func.count(User.id))
        result = await session.execute(stmt)
        count = result.scalar()

    return {"user_count": count}
```

### File Processing Task

```python
@TaskRegistry.register("process_csv", tags=["data", "processing"])
async def process_csv(filepath: str) -> dict[str, object]:
    """Process CSV file."""
    import pandas as pd

    df = pd.read_csv(filepath)
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "summary": df.describe().to_dict()
    }

    return summary
```

### Synchronous Task

```python
@TaskRegistry.register("multiply", tags=["math"])
def multiply(a: int, b: int) -> dict[str, int]:
    """Synchronous task that multiplies numbers."""
    # Sync functions are automatically wrapped for async execution
    return {"result": a * b}
```

### Shell Command Task

```python
from chapkit.task import run_shell

@TaskRegistry.register("backup_database", tags=["admin", "backup"])
async def backup_database(database_url: str, s3_bucket: str) -> dict[str, object]:
    """Backup database to S3 using shell commands."""
    # Dump database
    dump_result = await run_shell(
        f"pg_dump {database_url} | gzip > /tmp/backup.sql.gz",
        timeout=300.0
    )

    if dump_result["returncode"] != 0:
        return {
            "status": "failed",
            "step": "dump",
            "error": dump_result["stderr"]
        }

    # Upload to S3
    upload_result = await run_shell(
        f"aws s3 cp /tmp/backup.sql.gz s3://{s3_bucket}/backup.sql.gz",
        timeout=60.0
    )

    if upload_result["returncode"] != 0:
        return {
            "status": "failed",
            "step": "upload",
            "error": upload_result["stderr"]
        }

    return {
        "status": "success",
        "size": len(dump_result["stdout"])
    }

# Simple shell command task
@TaskRegistry.register("run_command", tags=["demo", "subprocess"])
async def run_command(command: str) -> dict[str, object]:
    """Run a shell command."""
    return await run_shell(command)

# Shell command with custom working directory and timeout
@TaskRegistry.register("list_files", tags=["filesystem"])
async def list_files(directory: str = ".") -> dict[str, object]:
    """List files in directory."""
    result = await run_shell("ls -la", cwd=directory, timeout=5.0)
    return {
        "directory": directory,
        "output": result["stdout"],
        "success": result["returncode"] == 0
    }
```

**run_shell() options:**
- `command: str` - Shell command to execute
- `timeout: float | None` - Optional timeout in seconds
- `cwd: str | Path | None` - Optional working directory
- `env: dict[str, str] | None` - Optional environment variables

**Returns dict with:**
- `command: str` - The command that was executed
- `stdout: str` - Standard output (decoded)
- `stderr: str` - Standard error (decoded)
- `returncode: int` - Exit code (0 = success, -1 = timeout)

**Note:** `run_shell()` never raises exceptions for non-zero exit codes. Always check `returncode` in the result.

---

## Advanced Patterns

### When to Use Simple vs Advanced Setup

**Use Simple Setup (default) when:**
- Tasks execute quickly (< 5 seconds)
- Results can be returned directly in HTTP response
- No need for background job scheduling
- No need for persistent artifact storage

**Use Advanced Setup when:**
- Tasks need to spawn background jobs for long-running operations
- Tasks need to store results in artifact hierarchy for audit/retrieval
- Tasks need job scheduling capabilities (retry, scheduling, etc.)
- Building a system with complex task orchestration

**Use ML Module instead when:**
- Building train/predict workflows
- Need versioned model storage
- Need experiment tracking
- Need standardized ML pipeline

### Task with Background Job Scheduling

When a task needs to spawn long-running background work:

```python
from chapkit.scheduler import ChapkitScheduler

@TaskRegistry.register("spawn_background_job", tags=["admin", "background"])
async def spawn_background_job(
    scheduler: ChapkitScheduler,
    processing_time: int = 60
) -> dict[str, object]:
    """Task that spawns a background job for long-running work."""

    async def background_work():
        """The actual long-running work."""
        import asyncio
        await asyncio.sleep(processing_time)
        return {"status": "completed", "processing_time": processing_time}

    # Spawn background job
    job_id = await scheduler.spawn(
        background_work(),
        description=f"Background processing ({processing_time}s)"
    )

    return {
        "message": "Background job started",
        "job_id": str(job_id),
        "check_status": f"/api/v1/jobs/{job_id}"
    }

# Execute:
# POST /api/v1/tasks/spawn_background_job/$execute
# {"params": {"processing_time": 120}}
#
# Response (immediate):
# {
#   "task_name": "spawn_background_job",
#   "result": {
#     "message": "Background job started",
#     "job_id": "01234567-89ab-cdef-0123-456789abcdef",
#     "check_status": "/api/v1/jobs/01234567-89ab-cdef-0123-456789abcdef"
#   }
# }
#
# Then check job status:
# GET /api/v1/jobs/01234567-89ab-cdef-0123-456789abcdef
```

### Task with Artifact Storage

When a task needs to store results in the artifact hierarchy:

```python
from chapkit.artifact import ArtifactManager, ArtifactIn

@TaskRegistry.register("store_analysis_results", tags=["analytics", "storage"])
async def store_analysis_results(
    artifact_manager: ArtifactManager,
    dataset_name: str,
    analysis_type: str
) -> dict[str, object]:
    """Task that stores analysis results as artifacts."""

    # Perform analysis
    results = {
        "dataset": dataset_name,
        "type": analysis_type,
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }

    # Store in artifact hierarchy
    artifact = await artifact_manager.save(
        ArtifactIn(
            data=results,
            metadata={
                "dataset": dataset_name,
                "analysis_type": analysis_type
            }
        )
    )

    return {
        "status": "stored",
        "artifact_id": str(artifact.id),
        "retrieve_url": f"/api/v1/artifacts/{artifact.id}"
    }

# Execute:
# POST /api/v1/tasks/store_analysis_results/$execute
# {"params": {"dataset_name": "customer_churn", "analysis_type": "classification"}}
#
# Response:
# {
#   "task_name": "store_analysis_results",
#   "result": {
#     "status": "stored",
#     "artifact_id": "01234567-89ab-cdef-0123-456789abcdef",
#     "retrieve_url": "/api/v1/artifacts/01234567-89ab-cdef-0123-456789abcdef"
#   }
# }
```

### Task with Both Scheduler and Artifacts

Combining job scheduling with artifact storage for complex workflows:

```python
@TaskRegistry.register("orchestrate_pipeline", tags=["pipeline", "orchestration"])
async def orchestrate_pipeline(
    scheduler: ChapkitScheduler,
    artifact_manager: ArtifactManager,
    pipeline_config: dict
) -> dict[str, object]:
    """Task that orchestrates multi-step pipeline with job tracking and artifact storage."""

    # Store pipeline configuration as artifact
    config_artifact = await artifact_manager.save(
        ArtifactIn(
            data=pipeline_config,
            metadata={"type": "pipeline_config"}
        )
    )

    # Define pipeline steps as background jobs
    async def step_1():
        result = {"step": 1, "status": "completed", "data": [1, 2, 3]}
        # Store step result
        await artifact_manager.save(ArtifactIn(
            data=result,
            metadata={"step": 1, "pipeline_config_id": str(config_artifact.id)}
        ))
        return result

    async def step_2():
        result = {"step": 2, "status": "completed", "data": [4, 5, 6]}
        await artifact_manager.save(ArtifactIn(
            data=result,
            metadata={"step": 2, "pipeline_config_id": str(config_artifact.id)}
        ))
        return result

    # Spawn jobs for each step
    job_1 = await scheduler.spawn(step_1(), description="Pipeline Step 1")
    job_2 = await scheduler.spawn(step_2(), description="Pipeline Step 2")

    return {
        "message": "Pipeline started",
        "config_artifact_id": str(config_artifact.id),
        "jobs": {
            "step_1": str(job_1),
            "step_2": str(job_2)
        },
        "monitor": {
            "jobs": "/api/v1/jobs",
            "artifacts": f"/api/v1/artifacts/{config_artifact.id}/$expand"
        }
    }
```

### Advanced Setup Configuration

Complete service setup with scheduler and artifact injection:

```python
from fastapi import Depends, FastAPI
from servicekit import Database
from servicekit.api.dependencies import get_database, get_scheduler

from chapkit.api import ServiceBuilder, ServiceInfo
from chapkit.artifact import ArtifactHierarchy, ArtifactManager, ArtifactRepository
from chapkit.scheduler import ChapkitScheduler
from chapkit.task import TaskExecutor, TaskRouter

# Define artifact hierarchy for task results
TASK_HIERARCHY = ArtifactHierarchy(
    name="task_results",
    level_labels={0: "task_run"}
)

# Advanced executor factory with scheduler and artifacts
async def get_task_executor(
    database: Database = Depends(get_database),
    scheduler = Depends(get_scheduler),
) -> TaskExecutor:
    """Provide task executor with scheduler and artifact manager."""
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo, hierarchy=TASK_HIERARCHY)
        if isinstance(scheduler, ChapkitScheduler):
            return TaskExecutor(database, scheduler, artifact_manager)
        return TaskExecutor(database)

# Create task router
task_router = TaskRouter.create(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
    executor_factory=get_task_executor,
)

# Build service with jobs and artifacts
app: FastAPI = (
    ServiceBuilder(info=ServiceInfo(id="advanced-task-service", display_name="Advanced Task Service"))
    .with_landing_page()
    .with_logging()
    .with_health()
    .with_system()
    .with_jobs(max_concurrency=5)          # Enable job scheduler
    .with_artifacts(hierarchy=TASK_HIERARCHY)  # Enable artifact storage
    .include_router(task_router.router)
    .build()
)
```

### Decision Guide: Tasks vs ML Module

**Use Task Execution when:**
- Running general-purpose Python functions
- Need flexible dependency injection
- Building custom workflows
- Tasks are ephemeral or self-contained
- Mix of data processing, admin, ETL, etc.

**Use ML Module when:**
- Specifically doing ML train/predict
- Need standardized ML workflow
- Need model versioning
- Need experiment tracking
- Want automatic model artifact management

**Example: When to use each**
```python
# Use Task Execution for this:
@TaskRegistry.register("process_user_data")
async def process_user_data(database: Database, user_id: str):
    """Custom business logic."""
    # ... custom processing
    return {"processed": True}

# Use ML Module for this:
from chapkit.ml import MLManager
ml_manager = MLManager(...)
await ml_manager.train(TrainRequest(
    model_name="customer_churn",
    parameters={"n_estimators": 100}
))
```

---

## Complete Workflow Example

```bash
# Start service
fastapi dev main.py

# List all tasks
curl http://localhost:8000/api/v1/tasks | jq

# Get task metadata
curl http://localhost:8000/api/v1/tasks/greet_user | jq

# Execute task and get result immediately
curl -s -X POST http://localhost:8000/api/v1/tasks/greet_user/\$execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"name": "Alice"}}' | jq

# Expected response:
# {
#   "task_name": "greet_user",
#   "params": {"name": "Alice"},
#   "result": {"message": "Hello, Alice!"},
#   "error": null
# }
```

---

## Result Storage

Task execution results are **ephemeral** - they are returned directly in the HTTP response and not persisted.

**For task history/persistence needs:** Use the ML module instead, which provides artifact-based storage for train/predict workflows.

**Response structure for successful execution:**
```json
{
    "task_name": "greet_user",
    "params": {"name": "Alice"},
    "result": {"message": "Hello, Alice!"},
    "error": null
}
```

**Response structure for failed execution:**
```json
{
    "task_name": "failing_task",
    "params": {},
    "result": null,
    "error": {
        "type": "ValueError",
        "message": "Something went wrong",
        "traceback": "Traceback (most recent call last)..."
    }
}
```

---

## Testing

### Unit Tests

```python
import pytest
from servicekit import Database, SqliteDatabaseBuilder
from chapkit.task import TaskRegistry, TaskExecutor

# Clear registry before tests
TaskRegistry.clear()

@TaskRegistry.register("test_task", tags=["test"])
async def test_task(value: str) -> dict[str, str]:
    """Test task."""
    return {"result": value.upper()}

@pytest.fixture
async def database() -> Database:
    """Create in-memory database for testing."""
    db = SqliteDatabaseBuilder().in_memory().build()
    await db.init()
    return db

@pytest.fixture
def task_executor(database: Database) -> TaskExecutor:
    """Create task executor."""
    return TaskExecutor(database)

@pytest.mark.asyncio
async def test_task_execution(task_executor: TaskExecutor):
    """Test task execution returns result directly."""
    # Execute task
    result = await task_executor.execute("test_task", {"value": "hello"})

    # Verify result
    assert result["result"] == "HELLO"

@pytest.mark.asyncio
async def test_task_registry():
    """Test task registry."""
    # Verify registration
    assert TaskRegistry.has("test_task")

    # Get metadata
    info = TaskRegistry.get_info("test_task")
    assert info.name == "test_task"
    assert info.tags == ["test"]

@pytest.mark.asyncio
async def test_dependency_injection(database: Database, task_executor: TaskExecutor):
    """Test dependency injection."""
    @TaskRegistry.register("test_injection")
    async def task_with_db(db: Database, param: str) -> dict[str, str]:
        """Task with injected database."""
        return {"param": param, "db_injected": db is not None}

    result = await task_executor.execute("test_injection", {"param": "test"})
    assert result["param"] == "test"
    assert result["db_injected"] is True
```

### Integration Tests

```python
from fastapi.testclient import TestClient

def test_task_workflow(client: TestClient):
    """Test complete task workflow."""
    # List tasks
    response = client.get("/api/v1/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) > 0

    # Get task metadata
    task_name = tasks[0]["name"]
    response = client.get(f"/api/v1/tasks/{task_name}")
    assert response.status_code == 200
    task_info = response.json()
    assert task_info["name"] == task_name

    # Execute task - gets result immediately
    exec_response = client.post(
        f"/api/v1/tasks/{task_name}/$execute",
        json={"params": {"name": "Test"}}
    )
    assert exec_response.status_code == 200
    data = exec_response.json()

    # Verify response structure
    assert data["task_name"] == task_name
    assert data["params"] == {"name": "Test"}
    assert data["result"] is not None
    assert data["error"] is None

def test_task_error_handling(client: TestClient):
    """Test task error handling."""
    # Execute task that will fail
    response = client.post(
        "/api/v1/tasks/failing_task/$execute",
        json={"params": {}}
    )
    assert response.status_code == 200
    data = response.json()

    # Error captured in response
    assert data["result"] is None
    assert data["error"] is not None
    assert "type" in data["error"]
    assert "message" in data["error"]
    assert "traceback" in data["error"]
```

---

## Production Considerations

### Error Handling

Tasks should handle errors gracefully:

```python
@TaskRegistry.register("safe_task", tags=["production"])
async def safe_task(risky_param: str) -> dict[str, object]:
    """Task with error handling."""
    try:
        result = process_risky_operation(risky_param)
        return {"status": "success", "result": result}
    except Exception as e:
        # Error will be captured in response automatically
        # but you can also return error status for app-level handling
        return {"status": "error", "error": str(e)}
```

**Note:** Even if a task raises an exception, the TaskExecutor will catch it and return the error in the HTTP response with full traceback. Tasks execute synchronously and return results/errors directly.

### Execution Timeout

Since tasks execute synchronously, consider the HTTP request timeout:

```python
@TaskRegistry.register("quick_task", tags=["production"])
async def quick_task(data: dict) -> dict[str, object]:
    """Task should complete quickly (< 30s recommended)."""
    # Process data quickly
    result = fast_processing(data)
    return {"result": result}

# For longer operations, use advanced setup with scheduler
@TaskRegistry.register("long_operation", tags=["production"])
async def long_operation(scheduler: ChapkitScheduler, data: dict) -> dict[str, object]:
    """Spawn background job for long-running work."""
    async def background_work():
        return slow_processing(data)

    job_id = await scheduler.spawn(background_work(), description="Long operation")
    return {
        "status": "started",
        "job_id": str(job_id),
        "check_at": f"/api/v1/jobs/{job_id}"
    }
```

### Concurrency Control (Advanced Setup Only)

When using advanced setup with `.with_jobs()`, you can limit concurrent background jobs:

```python
app = (
    ServiceBuilder(info=ServiceInfo(id="task-service", display_name="Task Service"))
    .with_jobs(max_concurrency=5)  # Max 5 concurrent background jobs
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .build()
)
```

**Note:** This only limits background jobs spawned by tasks, not the synchronous task execution itself. FastAPI handles concurrent HTTP requests based on its own worker configuration.

### Long-Running Tasks (Advanced Setup)

For tasks that need progress tracking, use advanced setup with artifacts:

```python
from chapkit.artifact import ArtifactManager, ArtifactIn

@TaskRegistry.register("long_task", tags=["processing", "batch"])
async def long_task(artifact_manager: ArtifactManager) -> dict[str, object]:
    """Task with progress tracking via artifacts."""
    total_steps = 10
    results = []

    for i in range(total_steps):
        # Do work
        step_result = await process_step(i)
        results.append(step_result)

        # Store intermediate progress
        await artifact_manager.save(ArtifactIn(
            data={"step": i, "total": total_steps, "result": step_result}
        ))

    return {
        "status": "complete",
        "steps_completed": total_steps,
        "results": results
    }
```

**Or spawn a background job:**

```python
@TaskRegistry.register("long_batch", tags=["processing", "batch"])
async def long_batch(
    scheduler: ChapkitScheduler,
    artifact_manager: ArtifactManager
) -> dict[str, object]:
    """Spawn background job for truly long operations."""

    async def batch_work():
        results = []
        for i in range(100):
            result = await process_step(i)
            results.append(result)
        # Store final result
        artifact = await artifact_manager.save(ArtifactIn(data={"results": results}))
        return {"artifact_id": str(artifact.id)}

    job_id = await scheduler.spawn(batch_work(), description="Batch processing")
    return {"job_id": str(job_id), "check_at": f"/api/v1/jobs/{job_id}"}
```

### Task Organization with Tags

Use tags for effective task organization:

```python
# By functionality
@TaskRegistry.register("extract_data", tags=["etl", "extract"])
@TaskRegistry.register("transform_data", tags=["etl", "transform"])
@TaskRegistry.register("load_data", tags=["etl", "load"])

# By environment
@TaskRegistry.register("dev_setup", tags=["dev", "setup"])
@TaskRegistry.register("prod_setup", tags=["prod", "setup"])

# By priority
@TaskRegistry.register("urgent_task", tags=["high-priority"])
@TaskRegistry.register("batch_task", tags=["low-priority", "batch"])
```

### Hot Reload During Development

Clear the registry when your module reloads:

```python
# At top of your main.py
TaskRegistry.clear()

# Then register tasks
@TaskRegistry.register("my_task")
async def my_task():
    ...
```

This prevents duplicate registration errors during development.

---

## Complete Example

See `examples/task_execution/main.py` for a complete working example with:
- Multiple task types (simple, with parameters, with injection)
- Tag-based organization
- Dependency injection
- Artifact integration
- Service configuration

## Next Steps

- **Job Scheduler:** Learn about job monitoring and concurrency control
- **Artifact Storage:** Understand artifact hierarchies and result storage
- **Service Builder:** Configure services with multiple features
- **Monitoring:** Track task execution metrics
