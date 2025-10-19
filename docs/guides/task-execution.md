# Task Execution

Servicekit provides a task execution system for running Python functions and shell commands asynchronously with dependency injection, orphan detection, and artifact storage integration.

## Quick Start

### Python Task Functions

```python
from chapkit.task import TaskRegistry, TaskManager, TaskIn, TaskRepository, TaskRouter
from servicekit.api import BaseServiceBuilder, ServiceInfo
from servicekit import Database
from chapkit.artifact import ArtifactManager
from fastapi import Depends

# Register Python task
@TaskRegistry.register("greet_user")
async def greet_user(name: str = "World") -> dict[str, str]:
    """Simple task that returns a greeting."""
    return {"message": f"Hello, {name}!"}

# Task with dependency injection
@TaskRegistry.register("process_data")
async def process_data(database: Database, artifact_manager: ArtifactManager) -> dict[str, object]:
    """Dependencies are automatically injected."""
    # Use injected database and artifact_manager
    return {"status": "complete", "database_url": str(database.url)}

# Build service
app = (
    BaseServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_health()
    .with_database("sqlite+aiosqlite:///./data.db")
    .with_jobs(max_concurrency=5)
    .build()
)

# Add task router
task_router = TaskRouter.create(
    prefix="/api/v1/tasks",
    tags=["Tasks"],
    manager_factory=get_task_manager,  # Dependency factory
    entity_in_type=TaskIn,
    entity_out_type=TaskOut,
)
app.include_router(task_router)
```

### Shell Commands

```python
import sys

# Execute shell commands
await manager.save(TaskIn(
    command=f"{sys.executable} --version",
    enabled=True
))

await manager.save(TaskIn(
    command="echo 'Hello from shell!'",
    enabled=True
))
```

**Run:** `fastapi dev your_file.py`

---

## Architecture

### Task Types

**Python Functions:**
- Registered with `@TaskRegistry.register(name)`
- Automatic dependency injection
- Return dict with results
- Async or sync functions

**Shell Commands:**
- Any executable command
- Stdout/stderr captured
- Exit code handling
- Environment variable support

### Execution Flow

```
1. Task Submitted
   POST /api/v1/tasks

2. Task Stored
   ├─> Database record created
   ├─> Enabled = True/False
   └─> Metadata stored

3. Task Execution ($execute endpoint)
   ├─> Job created in scheduler
   ├─> Dependencies injected (Python tasks)
   ├─> Process spawned (shell tasks)
   └─> Results returned

4. Results Storage
   ├─> Return value stored
   ├─> Stdout/stderr captured
   └─> Optional artifact linking
```

---

## Core Concepts

### TaskRegistry

Global registry for Python task functions.

```python
from chapkit.task import TaskRegistry

@TaskRegistry.register("my_task")
async def my_task(param: str) -> dict[str, object]:
    """Task docstring."""
    return {"result": param.upper()}

# Verify registration
assert TaskRegistry.is_registered("my_task")

# Get task function
task_fn = TaskRegistry.get("my_task")
```

**Rules:**
- Task names must be unique
- Functions must return dict or None
- Both async and sync functions supported
- Parameters can have defaults

### Dependency Injection

Tasks can request dependencies as function parameters:

```python
@TaskRegistry.register("with_dependencies")
async def with_dependencies(
    database: Database,
    artifact_manager: ArtifactManager,
    scheduler: JobScheduler,
    custom_param: str = "default"
) -> dict[str, object]:
    """Dependencies automatically injected at runtime."""
    # Use injected dependencies
    artifact = await artifact_manager.save(ArtifactIn(data={"result": custom_param}))
    return {"artifact_id": str(artifact.id)}
```

**Available Injectables:**
- `database`: Database instance
- `artifact_manager`: ArtifactManager instance
- `scheduler`: JobScheduler instance

**Note:** Task parameters with defaults are treated as task arguments, not injected dependencies.

### Task Model

```python
from chapkit.task import TaskIn

task = TaskIn(
    command="greet_user",      # Function name or shell command
    enabled=True,               # Whether task can be executed
    params={"name": "Alice"},  # Parameters for Python tasks
)
```

**Fields:**
- `id`: ULID (auto-generated)
- `command`: Task function name or shell command
- `enabled`: Boolean flag for execution control
- `params`: Optional dict of parameters (Python tasks only)
- `created_at`, `updated_at`: Timestamps

### TaskManager

Business logic layer for task operations.

```python
from chapkit.task import TaskManager

manager = TaskManager(repository, scheduler, database, artifact_manager)

# Create task
task = await manager.save(TaskIn(command="greet_user", enabled=True))

# Execute task
result = await manager.execute_task(task.id, params={"name": "Bob"})

# List all tasks
tasks = await manager.find_all()

# Find orphaned tasks (no job assigned)
orphans = await manager.find_orphaned_tasks()
```

---

## API Endpoints

### POST /api/v1/tasks

Create new task.

**Request:**
```json
{
  "command": "greet_user",
  "enabled": true,
  "params": {
    "name": "Alice"
  }
}
```

**Response (201 Created):**
```json
{
  "id": "01TASK123...",
  "command": "greet_user",
  "enabled": true,
  "params": {
    "name": "Alice"
  },
  "created_at": "2025-10-18T12:00:00Z",
  "updated_at": "2025-10-18T12:00:00Z"
}
```

### GET /api/v1/tasks

List all tasks with pagination.

### GET /api/v1/tasks/{id}

Get task by ID.

### PUT /api/v1/tasks/{id}

Update task.

**Request:**
```json
{
  "enabled": false,
  "params": {
    "name": "Updated"
  }
}
```

### DELETE /api/v1/tasks/{id}

Delete task.

### POST /api/v1/tasks/{id}/$execute

Execute task asynchronously.

**Request:**
```json
{
  "params": {
    "name": "Bob"
  }
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "01JOB456...",
  "message": "Task execution started"
}
```

### GET /api/v1/tasks/$orphaned

Find orphaned tasks (tasks without assigned jobs).

**Response:**
```json
{
  "orphaned_tasks": [
    {
      "id": "01TASK789...",
      "command": "greet_user",
      "enabled": true
    }
  ],
  "count": 1
}
```

---

## Python Task Patterns

### Simple Task

```python
@TaskRegistry.register("hello")
async def hello() -> dict[str, str]:
    """Simple hello world task."""
    return {"message": "Hello!"}
```

### Task with Parameters

```python
@TaskRegistry.register("add")
async def add(a: int, b: int) -> dict[str, int]:
    """Add two numbers."""
    return {"result": a + b}

# Execute with params
await manager.execute_task(task_id, params={"a": 5, "b": 3})
```

### Task with Dependency Injection

```python
@TaskRegistry.register("store_result")
async def store_result(
    artifact_manager: ArtifactManager,
    data: dict
) -> dict[str, object]:
    """Store result in artifact."""
    artifact = await artifact_manager.save(ArtifactIn(data=data))
    return {"artifact_id": str(artifact.id)}

# Execute with params
await manager.execute_task(task_id, params={"data": {"key": "value"}})
```

### Database Query Task

```python
@TaskRegistry.register("count_users")
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
@TaskRegistry.register("process_csv")
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

---

## Shell Command Patterns

### Simple Commands

```python
# Python version
await manager.save(TaskIn(
    command=f"{sys.executable} --version",
    enabled=True
))

# Echo command
await manager.save(TaskIn(
    command="echo 'Processing complete'",
    enabled=True
))

# List files
await manager.save(TaskIn(
    command="ls -la /data",
    enabled=True
))
```

### Script Execution

```python
# Execute Python script
await manager.save(TaskIn(
    command=f"{sys.executable} scripts/process_data.py --input data.csv --output results.json",
    enabled=True
))

# Execute shell script
await manager.save(TaskIn(
    command="bash scripts/backup.sh /data /backups",
    enabled=True
))
```

### Data Pipeline

```python
# Multi-step data processing
commands = [
    "wget https://example.com/data.csv -O /tmp/data.csv",
    f"{sys.executable} scripts/clean_data.py /tmp/data.csv /tmp/clean.csv",
    f"{sys.executable} scripts/analyze.py /tmp/clean.csv /tmp/results.json"
]

for cmd in commands:
    task = await manager.save(TaskIn(command=cmd, enabled=True))
    await manager.execute_task(task.id)
```

---

## Complete Workflow Example

```bash
# Start service
fastapi dev examples/task_execution_api.py

# Create Python task
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "greet_user",
    "enabled": true,
    "params": {"name": "Alice"}
  }' | jq -r '.id')

# Execute task
EXEC_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute \
  -H "Content-Type: application/json" \
  -d '{"params": {"name": "Bob"}}')

JOB_ID=$(echo $EXEC_RESPONSE | jq -r '.job_id')

# Monitor job status
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq

# Stream job updates (SSE)
curl -N http://localhost:8000/api/v1/jobs/$JOB_ID/\$stream

# Check for orphaned tasks
curl http://localhost:8000/api/v1/tasks/\$orphaned | jq
```

---

## Testing

### Unit Tests

```python
import pytest
from chapkit.task import TaskRegistry, TaskManager, TaskIn

@TaskRegistry.register("test_task")
async def test_task(value: str) -> dict[str, str]:
    """Test task."""
    return {"result": value.upper()}

@pytest.mark.asyncio
async def test_task_execution(task_manager):
    """Test task execution."""
    # Create task
    task = await task_manager.save(TaskIn(
        command="test_task",
        enabled=True,
        params={"value": "hello"}
    ))

    # Execute
    result = await task_manager.execute_task(task.id)

    # Verify
    assert result["result"] == "HELLO"
```

### Integration Tests

```python
from fastapi.testclient import TestClient

def test_task_workflow(client: TestClient):
    """Test complete task workflow."""
    # Create task
    response = client.post("/api/v1/tasks", json={
        "command": "greet_user",
        "enabled": True
    })
    assert response.status_code == 201
    task_id = response.json()["id"]

    # Execute task
    exec_response = client.post(f"/api/v1/tasks/{task_id}/$execute", json={
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
    BaseServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_jobs(max_concurrency=5)  # Max 5 concurrent tasks
    .build()
)
```

### Error Handling

Tasks should handle errors gracefully:

```python
@TaskRegistry.register("safe_task")
async def safe_task(risky_param: str) -> dict[str, object]:
    """Task with error handling."""
    try:
        # Risky operation
        result = process_risky_operation(risky_param)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### Long-Running Tasks

For long-running tasks, provide progress updates:

```python
@TaskRegistry.register("long_task")
async def long_task() -> dict[str, object]:
    """Long-running task."""
    import time

    steps = 10
    for i in range(steps):
        # Do work
        time.sleep(1)
        # Log progress (captured in job logs)
        print(f"Progress: {i+1}/{steps}")

    return {"status": "complete"}
```

### Orphan Detection

Regularly check for orphaned tasks:

```bash
# Cron job to detect orphans
*/15 * * * * curl http://localhost:8000/api/v1/tasks/\$orphaned
```

### Shell Command Security

**Important:** Validate shell commands to prevent injection:

```python
import shlex

def validate_command(command: str) -> bool:
    """Validate shell command for safety."""
    # Parse command safely
    try:
        args = shlex.split(command)
        # Check against whitelist
        allowed_executables = ["python", "bash", "ls", "echo"]
        if args[0] not in allowed_executables:
            return False
        return True
    except ValueError:
        return False

# Use in task creation
if validate_command(task_command):
    task = await manager.save(TaskIn(command=task_command, enabled=True))
```

---

## Complete Example

See `examples/task_execution_api.py` for a complete working example with Python tasks, shell commands, and dependency injection.

## Next Steps

- **Job Scheduler:** Learn about job monitoring and concurrency control
- **Artifact Storage:** Store task results in artifacts
- **Authentication:** Secure task endpoints with API keys
- **Monitoring:** Track task execution with Prometheus metrics
