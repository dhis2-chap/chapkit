# Task Execution

Chapkit provides a registry-based task execution system for running Python functions asynchronously with dependency injection, tag-based organization, and artifact storage integration.

## Quick Start

### Basic Task Registration

```python
from chapkit.task import TaskRegistry, TaskExecutor, TaskRouter
from chapkit.api import ServiceBuilder, ServiceInfo
from servicekit import Database
from chapkit.artifact import ArtifactManager
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
async def process_data(database: Database, artifact_manager: ArtifactManager) -> dict[str, object]:
    """Dependencies are automatically injected."""
    return {"status": "complete", "database_url": str(database.url)}

# Build service
info = ServiceInfo(display_name="Task Service")

async def get_task_executor(
    scheduler = Depends(get_scheduler),
    database = Depends(get_database),
) -> TaskExecutor:
    """Provide task executor for dependency injection."""
    async with database.session() as session:
        artifact_repo = ArtifactRepository(session)
        artifact_manager = ArtifactManager(artifact_repo)
        return TaskExecutor(scheduler, database, artifact_manager)

task_router = TaskRouter.create(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
    executor_factory=get_task_executor,
)

app = (
    ServiceBuilder(info=info)
    .with_health()
    .with_jobs(max_concurrency=5)
    .include_router(task_router)
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
   GET /api/v1/tasks?tags=tag1,tag2
   GET /api/v1/tasks/task_name

3. Task Execution
   POST /api/v1/tasks/task_name/$execute
   ├─> Job created in scheduler
   ├─> Dependencies injected
   └─> Execution started

4. Results Storage
   ├─> Artifact created with task results
   ├─> Job result contains artifact_id
   └─> Retrieve via /api/v1/artifacts/{artifact_id}
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

Tasks can be tagged for organization and filtering:

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
from chapkit.artifact import ArtifactManager
from chapkit.scheduler import ChapkitJobScheduler
from sqlalchemy.ext.asyncio import AsyncSession

@TaskRegistry.register("with_dependencies")
async def with_dependencies(
    database: Database,
    artifact_manager: ArtifactManager,
    scheduler: ChapkitJobScheduler,
    session: AsyncSession,
    custom_param: str = "default"
) -> dict[str, object]:
    """Dependencies automatically injected at runtime."""
    # Framework types are injected, user params come from request
    artifact = await artifact_manager.save(ArtifactIn(data={"result": custom_param}))
    return {"artifact_id": str(artifact.id)}
```

**Available Injectable Types:**
- `AsyncSession` - Database session
- `Database` - Database instance
- `ArtifactManager` - Artifact manager
- `ChapkitJobScheduler` - Job scheduler

**Note:** Parameters are categorized automatically:
- Framework types (in INJECTABLE_TYPES) are injected
- All other parameters must be provided in execution request

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

**Query Parameters:**
- `tags` (optional): Comma-separated tags to filter by (requires ALL tags)

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

**Examples:**
```bash
# List all tasks
curl http://localhost:8000/api/v1/tasks

# Filter by tags (AND operation)
curl http://localhost:8000/api/v1/tasks?tags=demo,simple
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

**Response (202 Accepted):**
```json
{
  "job_id": "01JOB456...",
  "message": "Task 'greet_user' submitted for execution. Job ID: 01JOB456..."
}
```

**Errors:**
- 404 Not Found: Task not registered
- 400 Bad Request: Missing required parameters
- 500 Internal Server Error: Execution failed

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

---

## Complete Workflow Example

```bash
# Start service
fastapi dev main.py

# List all tasks
curl http://localhost:8000/api/v1/tasks | jq

# Filter tasks by tags
curl 'http://localhost:8000/api/v1/tasks?tags=demo,simple' | jq

# Get task metadata
curl http://localhost:8000/api/v1/tasks/greet_user | jq

# Execute task
JOB_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/tasks/greet_user/\$execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"name": "Alice"}}')

JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# Monitor job status
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq

# Wait for completion and get artifact ID
ARTIFACT_ID=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.result')
echo "Artifact ID: $ARTIFACT_ID"

# Get task results from artifact
curl http://localhost:8000/api/v1/artifacts/$ARTIFACT_ID | jq

# Expected artifact data structure:
# {
#   "id": "01ABC...",
#   "data": {
#     "task_name": "greet_user",
#     "params": {"name": "Alice"},
#     "result": {"message": "Hello, Alice!"},
#     "error": null
#   }
# }
```

---

## Result Storage

Task execution results are automatically stored in artifacts:

```python
# Artifact data structure for successful execution:
{
    "task_name": "greet_user",
    "params": {"name": "Alice"},
    "result": {"message": "Hello, Alice!"},
    "error": null
}

# Artifact data structure for failed execution:
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

The artifact ID is returned as the job result field.

---

## Testing

### Unit Tests

```python
import pytest
from chapkit.task import TaskRegistry

# Clear registry before tests
TaskRegistry.clear()

@TaskRegistry.register("test_task", tags=["test"])
async def test_task(value: str) -> dict[str, str]:
    """Test task."""
    return {"result": value.upper()}

@pytest.mark.asyncio
async def test_task_execution(task_executor):
    """Test task execution."""
    # Execute task
    job_id = await task_executor.execute("test_task", {"value": "hello"})

    # Verify job was created
    assert job_id is not None

@pytest.mark.asyncio
async def test_task_registry():
    """Test task registry."""
    # Verify registration
    assert TaskRegistry.has("test_task")

    # Get metadata
    info = TaskRegistry.get_info("test_task")
    assert info.name == "test_task"
    assert info.tags == ["test"]
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

    # Execute task
    exec_response = client.post(f"/api/v1/tasks/{task_name}/$execute", json={
        "params": {"name": "Test"}
    })
    assert exec_response.status_code == 202
    job_id = exec_response.json()["job_id"]

    # Wait for completion
    import time
    time.sleep(1)

    # Check job
    job_response = client.get(f"/api/v1/jobs/{job_id}")
    assert job_response.json()["status"] == "completed"
```

---

## Production Considerations

### Concurrency Control

Limit concurrent task execution:

```python
app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_jobs(max_concurrency=5)  # Max 5 concurrent tasks
    .build()
)
```

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
        # Error will be captured in artifact automatically
        # but you can also return error status
        return {"status": "error", "error": str(e)}
```

Note: Even if a task raises an exception, the TaskExecutor will catch it and store the error in the artifact with full traceback.

### Long-Running Tasks

For long-running tasks, consider breaking them into smaller steps:

```python
@TaskRegistry.register("long_task", tags=["processing", "batch"])
async def long_task(artifact_manager: ArtifactManager) -> dict[str, object]:
    """Long-running task with progress tracking."""
    total_steps = 10
    results = []

    for i in range(total_steps):
        # Do work
        step_result = await process_step(i)
        results.append(step_result)

        # Store intermediate progress in artifact
        progress_artifact = await artifact_manager.save(ArtifactIn(
            data={"step": i, "total": total_steps, "result": step_result}
        ))

    return {
        "status": "complete",
        "steps_completed": total_steps,
        "results": results
    }
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

# Filter in API:
# GET /api/v1/tasks?tags=etl,extract  # Only extract tasks
# GET /api/v1/tasks?tags=prod         # All prod tasks
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
