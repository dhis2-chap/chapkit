# Task Execution

Chapkit provides a task execution system for running shell commands and Python functions asynchronously with artifact-based result storage. Tasks are reusable templates that can be executed multiple times, with each execution creating a Job and storing results in an Artifact.

## Quick Start

### Create and Execute a Task

```bash
# Start the example service
fastapi dev examples/task_execution_api.py

# Create a task template
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"command": "echo \"Hello World\""}' | jq -r '.id')

echo "Task ID: $TASK_ID"

# Execute the task
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Wait for completion and get results
sleep 1
ARTIFACT_ID=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.artifact_id')

# View execution results
curl http://localhost:8000/api/v1/artifacts/$ARTIFACT_ID | jq '.data'
```

Output:
```json
{
  "task": {
    "id": "01JCSEED...",
    "command": "echo \"Hello World\"",
    "created_at": "2025-10-14T...",
    "updated_at": "2025-10-14T..."
  },
  "stdout": "Hello World\n",
  "stderr": "",
  "exit_code": 0
}
```

---

## Architecture

The Task execution system uses a clean separation of concerns:

### Task Templates

**Tasks** are reusable command templates stored in the database:
- Contain only the command to execute
- No status or output fields (stateless)
- Can be executed multiple times
- Immutable history (past executions unaffected by template changes)

### Job Execution

When a task is executed:
1. Task snapshot is captured (ID, command, timestamps)
2. Job is submitted to the scheduler
3. Command runs asynchronously via `asyncio.subprocess`
4. stdout, stderr, and exit code are captured

### Artifact Storage

Execution results are stored as Artifacts:
- **task**: Full snapshot of the task template at execution time
- **stdout**: Command standard output
- **stderr**: Command standard error
- **exit_code**: Process exit code

The Job record links to the result artifact via `Job.artifact_id`.

**Benefits:**
- Tasks remain reusable templates
- Complete execution history preserved
- Modifying task doesn't affect past results
- Deleting task preserves execution artifacts

---

## Python Task Execution

In addition to shell commands, Chapkit supports executing registered Python functions as tasks. This provides type-safe, IDE-friendly task execution with parameter validation.

### TaskRegistry

Python functions must be registered before they can be executed as tasks. This prevents arbitrary code execution and ensures all callable functions are explicitly defined.

**Registration Methods:**

**1. Decorator Registration:**
```python
from chapkit import TaskRegistry

@TaskRegistry.register("calculate_sum")
async def calculate_sum(a: int, b: int) -> dict:
    """Calculate sum of two numbers asynchronously."""
    await asyncio.sleep(0.1)  # Simulate async work
    return {"result": a + b, "operation": "sum"}

@TaskRegistry.register("process_data")
def process_data(input_text: str, uppercase: bool = False) -> dict:
    """Process text data synchronously."""
    result = input_text.upper() if uppercase else input_text.lower()
    return {"processed": result, "original": input_text}
```

**2. Imperative Registration:**
```python
def my_function(param: str) -> dict:
    return {"result": f"Processed {param}"}

TaskRegistry.register_function("my_task", my_function)
```

### Creating Python Tasks

Python tasks use `task_type="python"` and accept a `parameters` dict:

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 10, "b": 32}
  }'
```

**Field Mapping:**
- `command` - Name of registered function (not the function body)
- `task_type` - Must be "python"
- `parameters` - Dict passed as kwargs to the function

### Python Task Artifacts

Python task results have a different structure than shell tasks:

**Successful Execution:**
```json
{
  "task": {
    "id": "01TASK...",
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 10, "b": 32},
    "created_at": "2025-10-17T...",
    "updated_at": "2025-10-17T..."
  },
  "result": {
    "result": 42,
    "operation": "sum"
  },
  "error": null
}
```

**Failed Execution:**
```json
{
  "task": {...},
  "result": null,
  "error": {
    "type": "ValueError",
    "message": "Invalid parameter value",
    "traceback": "Traceback (most recent call last):\n..."
  }
}
```

**Comparison with Shell Tasks:**

| Feature | Shell Tasks | Python Tasks |
|---------|-------------|--------------|
| Output fields | `stdout`, `stderr`, `exit_code` | `result`, `error` |
| Success indicator | `exit_code == 0` | `error == null` |
| Error info | `stderr` text | Full exception with traceback |
| Return value | Command output text | Any JSON-serializable Python object |

### Sync vs Async Functions

TaskRegistry supports both synchronous and asynchronous functions:

```python
# Async function - awaited directly
@TaskRegistry.register("async_task")
async def async_task(param: str) -> dict:
    await asyncio.sleep(1)
    return {"result": param}

# Sync function - executed in thread pool
@TaskRegistry.register("sync_task")
def sync_task(param: str) -> dict:
    import time
    time.sleep(1)  # Blocking operation
    return {"result": param}
```

Synchronous functions are executed in a thread pool via `asyncio.to_thread()` to prevent blocking the event loop.

### Dependency Injection

Python task functions support **type-based dependency injection** for framework services. The framework automatically injects dependencies based on parameter type hints, while user parameters come from `task.parameters`.

#### Injectable Types Reference

| Type | Description | Use Case |
|------|-------------|----------|
| `AsyncSession` | SQLAlchemy async database session | Database queries, ORM operations |
| `Database` | chapkit Database instance | Creating sessions, database operations |
| `ArtifactManager` | Artifact management service | Saving/loading artifacts during execution |
| `JobScheduler` | Job scheduling service | Submitting child jobs, job management |

**Location**: Defined in `src/chapkit/modules/task/manager.py` as `INJECTABLE_TYPES`

#### Basic Injection

Functions request framework services via type hints:

```python
from sqlalchemy.ext.asyncio import AsyncSession
from chapkit import TaskRegistry

@TaskRegistry.register("query_task_count")
async def query_task_count(session: AsyncSession) -> dict:
    """Task that queries database using injected session."""
    from sqlalchemy import select, func
    from chapkit.task.models import Task

    # Use injected session
    stmt = select(func.count()).select_from(Task)
    result = await session.execute(stmt)
    count = result.scalar() or 0

    return {
        "total_tasks": count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
```

**Execution** - No parameters needed:
```json
{
  "command": "query_task_count",
  "task_type": "python",
  "parameters": {}
}
```

#### Mixed Parameters

Combine user parameters with injected dependencies:

```python
@TaskRegistry.register("process_with_db")
async def process_with_db(
    input_text: str,        # From task.parameters
    count: int,             # From task.parameters
    session: AsyncSession,  # Injected by framework
) -> dict:
    """Mix user params and framework injection."""
    # Perform database operations using session
    # Process user-provided input_text and count
    return {"processed": input_text, "count": count}
```

**Execution**:
```json
{
  "command": "process_with_db",
  "task_type": "python",
  "parameters": {
    "input_text": "Hello",
    "count": 42
  }
}
```

**Parameter Sources**:
- User parameters: Primitives (`str`, `int`, `dict`) and generic types (`pd.DataFrame`)
- Framework parameters: Injectable types from the table above

#### Optional Injection

Use Optional types for optional dependencies:

```python
@TaskRegistry.register("optional_db_task")
async def optional_db_task(
    data: dict,                        # From task.parameters (required)
    session: AsyncSession | None = None,  # Injected if available (optional)
) -> dict:
    """Task with optional session injection."""
    if session:
        # Use database if session available
        pass
    return {"processed": data}
```

#### Flexible Naming

Parameter names don't matter - only types:

```python
# All of these work - framework matches by type
async def task_a(session: AsyncSession) -> dict: ...
async def task_b(db_session: AsyncSession) -> dict: ...
async def task_c(conn: AsyncSession) -> dict: ...
```

This allows natural, readable parameter names in your functions.

#### Multiple Injections

Inject multiple framework services:

```python
from chapkit import Database, ArtifactManager

@TaskRegistry.register("complex_task")
async def complex_task(
    input_data: dict,                    # From task.parameters
    database: Database,                  # Injected
    artifact_manager: ArtifactManager,   # Injected
    session: AsyncSession,               # Injected
) -> dict:
    """Task using multiple framework services."""
    # Use all injected services
    return {"result": "processed"}
```

#### Error Handling

Missing required user parameters raise clear errors:

```python
@TaskRegistry.register("needs_param")
async def needs_param(name: str, session: AsyncSession) -> dict:
    return {"name": name}

# Executing without 'name' parameter:
{
  "command": "needs_param",
  "task_type": "python",
  "parameters": {}  # Missing 'name'
}

# Error captured in artifact:
{
  "error": {
    "type": "ValueError",
    "message": "Missing required parameter 'name' for task function.
                Parameter is not injectable and not provided in task.parameters."
  }
}
```

#### Best Practices

**DO:**
- Use type hints for all parameters
- Request only needed framework services
- Use descriptive parameter names
- Combine user parameters with injections naturally

**DON'T:**
- Mix user and framework parameter types (primitives vs injectable types are clear)
- Forget type hints (injection requires them)
- Assume services are always available (use Optional for optional deps)

#### Example: Database Query Task

Complete example combining injection with user parameters:

```python
@TaskRegistry.register("search_tasks")
async def search_tasks(
    command_pattern: str,           # User parameter
    enabled_only: bool = True,      # User parameter with default
    session: AsyncSession,           # Injected
) -> dict:
    """Search for tasks matching a pattern."""
    from sqlalchemy import select
    from chapkit.task.models import Task

    # Build query using injected session
    stmt = select(Task).where(Task.command.like(f"%{command_pattern}%"))

    if enabled_only:
        stmt = stmt.where(Task.enabled == True)

    result = await session.execute(stmt)
    tasks = result.scalars().all()

    return {
        "matches": len(tasks),
        "tasks": [{"id": str(t.id), "command": t.command} for t in tasks],
    }
```

**Usage**:
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "search_tasks",
    "task_type": "python",
    "parameters": {
      "command_pattern": "echo",
      "enabled_only": true
    }
  }'
```

### Complete Example

See `examples/python_task_execution_api.py` for a complete working example with:
- Multiple registered functions (async and sync)
- Error handling demonstrations
- Mixed shell and Python tasks
- Seeded example tasks

---

## Task Lifecycle

```
1. CREATE TEMPLATE          2. EXECUTE              3. RESULTS STORED
   POST /tasks             POST /tasks/:id/$execute   Artifact created
   └─> Task stored         └─> Job submitted          └─> Job.artifact_id
       (reusable)              (async execution)          (immutable result)
```

### State Transitions

**Task:** Stateless template (no execution state)

**Job:** Tracks execution state
- `pending` → `running` → `completed` (success)
                       → `failed` (error)
                       → `canceled` (user canceled)

**Artifact:** Immutable result record containing task snapshot + outputs

---

## ServiceBuilder Setup

### Minimal Configuration

```python
from chapkit import ArtifactHierarchy
from chapkit.api import ServiceBuilder, ServiceInfo

# Simple hierarchy for task execution artifacts
TASK_HIERARCHY = ArtifactHierarchy(
    name="task_executions",
    level_labels={0: "execution"},
)

app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_health()
    .with_artifacts(hierarchy=TASK_HIERARCHY)  # Required first
    .with_jobs(max_concurrency=5)              # Required second
    .with_tasks()                               # Enables task execution
    .build()
)
```

**Requirements:**
1. `.with_artifacts()` must be called before `.with_tasks()`
2. `.with_jobs()` must be called before `.with_tasks()`
3. Without these, `execute_task()` will raise `ValueError`

### With Database Configuration

```python
app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_database("tasks.db")  # Persist tasks and artifacts
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=3)
    .with_tasks()
    .build()
)
```

---

## API Reference

### POST /api/v1/tasks

Create a new task template (shell or Python).

**Request (Shell Task):**
```json
{
  "command": "echo 'Hello World'"
}
```

**Request (Python Task):**
```json
{
  "command": "calculate_sum",
  "task_type": "python",
  "parameters": {"a": 10, "b": 32}
}
```

**Fields:**
- `command` (required) - Shell command or registered Python function name
- `task_type` (optional) - "shell" (default) or "python"
- `parameters` (optional) - Dict of parameters for Python tasks (ignored for shell tasks)
- `enabled` (optional) - Boolean to enable/disable task execution (default: true)

**Response (201):**
```json
{
  "id": "01JCSEED0000000000000TASK1",
  "command": "calculate_sum",
  "task_type": "python",
  "parameters": {"a": 10, "b": 32},
  "enabled": true,
  "created_at": "2025-10-14T10:30:00Z",
  "updated_at": "2025-10-14T10:30:00Z"
}
```

### GET /api/v1/tasks

List all task templates with optional pagination and filtering.

```bash
# List all tasks
curl http://localhost:8000/api/v1/tasks

# Filter by enabled status
curl http://localhost:8000/api/v1/tasks?enabled=true   # Only enabled tasks
curl http://localhost:8000/api/v1/tasks?enabled=false  # Only disabled tasks

# With pagination
curl http://localhost:8000/api/v1/tasks?page=1&size=20

# Combine filters
curl http://localhost:8000/api/v1/tasks?enabled=true&page=1&size=10
```

**Response:**
```json
[
  {
    "id": "01JCSEED0000000000000TASK1",
    "command": "ls -la /tmp",
    "task_type": "shell",
    "parameters": null,
    "enabled": true,
    "created_at": "2025-10-14T10:30:00Z",
    "updated_at": "2025-10-14T10:30:00Z"
  },
  {
    "id": "01JCSEED0000000000000TASK2",
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 10, "b": 32},
    "enabled": false,
    "created_at": "2025-10-14T10:30:00Z",
    "updated_at": "2025-10-14T10:30:00Z"
  }
]
```

### GET /api/v1/tasks/{task_id}

Retrieve a specific task template by ID.

```bash
curl http://localhost:8000/api/v1/tasks/01JCSEED0000000000000TASK1
```

### PUT /api/v1/tasks/{task_id}

Update a task template.

**Request:**
```json
{
  "command": "echo 'Updated command'",
  "task_type": "shell"
}
```

**Note:** Updating a task does not affect previous execution artifacts. You can change task_type and parameters when updating.

### DELETE /api/v1/tasks/{task_id}

Delete a task template.

```bash
curl -X DELETE http://localhost:8000/api/v1/tasks/01JCSEED0000000000000TASK1
```

Returns `204 No Content` on success.

**Note:** Deleting a task preserves all execution artifacts.

### POST /api/v1/tasks/{task_id}/$execute

Execute a task asynchronously.

```bash
curl -X POST http://localhost:8000/api/v1/tasks/01JCSEED0000000000000TASK1/\$execute
```

**Response (202 Accepted):**
```json
{
  "job_id": "01JQR7X...",
  "message": "Task submitted for execution. Job ID: 01JQR7X..."
}
```

**Errors:**
- `400 Bad Request` - Task not found, invalid ID, or task is disabled
- `409 Conflict` - Scheduler or artifact manager not configured

### Task Enable/Disable

Tasks can be enabled or disabled to control execution. Disabled tasks cannot be executed but remain in the database for reference.

**Creating a Disabled Task:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "echo test",
    "enabled": false
  }'
```

**Disabling an Existing Task:**
```bash
curl -X PUT http://localhost:8000/api/v1/tasks/{task_id} \
  -H "Content-Type: application/json" \
  -d '{
    "command": "echo test",
    "enabled": false
  }'
```

**Attempting to Execute a Disabled Task:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks/{disabled_task_id}/\$execute
```

**Response (400):**
```json
{
  "detail": "Cannot execute disabled task {task_id}"
}
```

**Use Cases:**
- Temporarily pause task execution without deletion
- Preserve task history while preventing new executions
- Automatically disable orphaned Python tasks (see Orphaned Tasks section)
- Soft-delete pattern for auditing and compliance

---

## Artifact Integration

### Result Structure

Each execution creates an artifact with this structure:

```json
{
  "id": "01ARTIFACT...",
  "data": {
    "task": {
      "id": "01TASK...",
      "command": "echo 'test'",
      "created_at": "2025-10-14T10:30:00Z",
      "updated_at": "2025-10-14T10:30:00Z"
    },
    "stdout": "test\n",
    "stderr": "",
    "exit_code": 0
  },
  "created_at": "2025-10-14T10:31:00Z",
  "updated_at": "2025-10-14T10:31:00Z"
}
```

### Task Snapshot Preservation

The artifact contains a **complete snapshot** of the task at execution time:

```bash
# Create task
curl -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": "echo original"}' > task.json
TASK_ID=$(jq -r '.id' task.json)

# Execute task
curl -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute > exec1.json
JOB1=$(jq -r '.job_id' exec1.json)

# Modify task
curl -X PUT http://localhost:8000/api/v1/tasks/$TASK_ID \
  -d '{"command": "echo modified"}'

# Execute again
curl -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute > exec2.json
JOB2=$(jq -r '.job_id' exec2.json)

# First execution has original command
curl http://localhost:8000/api/v1/jobs/$JOB1 | jq '.artifact_id' | \
  xargs -I {} curl http://localhost:8000/api/v1/artifacts/{} | \
  jq '.data.task.command'
# Output: "echo original"

# Second execution has modified command
curl http://localhost:8000/api/v1/jobs/$JOB2 | jq '.artifact_id' | \
  xargs -I {} curl http://localhost:8000/api/v1/artifacts/{} | \
  jq '.data.task.command'
# Output: "echo modified"
```

### Finding Task Executions

```bash
# Get all artifacts (includes task execution results)
curl http://localhost:8000/api/v1/artifacts

# Filter by task ID in application code
artifacts=$(curl -s http://localhost:8000/api/v1/artifacts)
echo "$artifacts" | jq --arg task_id "$TASK_ID" \
  '[.[] | select(.data.task.id == $task_id)]'
```

---

## Examples

### Shell Task Examples

**Simple Commands:**

```bash
# Directory listing
curl -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": "ls -la /tmp"}' | jq -r '.id'

# Date command
curl -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": "date"}' | jq -r '.id'

# Echo with output
curl -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": "echo \"Task execution works!\""}' | jq -r '.id'
```

**Python One-liners (Shell Tasks):**

```bash
# Python one-liner as shell command
curl -X POST http://localhost:8000/api/v1/tasks -d '{
  "command": "python3 -c \"import sys; print(sys.version); print(2+2)\""
}'

# Python script with multiple operations
curl -X POST http://localhost:8000/api/v1/tasks -d '{
  "command": "python3 -c \"import json; print(json.dumps({\\\"result\\\": 42}))\""
}'
```

### Python Task Examples

**Async Function Execution:**

```bash
# Assuming you have registered this function:
# @TaskRegistry.register("calculate_sum")
# async def calculate_sum(a: int, b: int) -> dict:
#     await asyncio.sleep(0.1)
#     return {"result": a + b, "operation": "sum"}

# Create Python task
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 15, "b": 27}
  }' | jq -r '.id')

# Execute task
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute | jq -r '.job_id')

# Wait and get result
sleep 1
ARTIFACT_ID=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.artifact_id')

# View result
curl -s http://localhost:8000/api/v1/artifacts/$ARTIFACT_ID | jq '.data.result'
# Output: {"result": 42, "operation": "sum"}
```

**Sync Function with Parameters:**

```bash
# Assuming you have registered:
# @TaskRegistry.register("process_data")
# def process_data(input_text: str, uppercase: bool = False) -> dict:
#     result = input_text.upper() if uppercase else input_text.lower()
#     return {"processed": result, "original": input_text}

curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "process_data",
    "task_type": "python",
    "parameters": {
      "input_text": "Hello World",
      "uppercase": true
    }
  }'
```

**Error Handling:**

```bash
# Assuming you have registered:
# @TaskRegistry.register("failing_task")
# async def failing_task(should_fail: bool = True) -> dict:
#     if should_fail:
#         raise ValueError("This task was designed to fail")
#     return {"success": True}

TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -d '{
    "command": "failing_task",
    "task_type": "python",
    "parameters": {"should_fail": true}
  }' | jq -r '.id')

# Execute and check artifact
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute | jq -r '.job_id')
sleep 1

# View error details
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.artifact_id' | \
  xargs -I {} curl -s http://localhost:8000/api/v1/artifacts/{} | jq '.data.error'

# Output:
# {
#   "type": "ValueError",
#   "message": "This task was designed to fail",
#   "traceback": "Traceback (most recent call last):\n..."
# }
```

**Complete Working Example:**

See `examples/python_task_execution_api.py` for a full service with:
- Multiple registered functions (async and sync)
- Error handling demonstrations
- Mixed shell and Python tasks
- Integration with ServiceBuilder

### Multi-line Commands

```bash
# Using printf for multi-line output
curl -X POST http://localhost:8000/api/v1/tasks -d '{
  "command": "printf \"Line 1\\nLine 2\\nLine 3\""
}'

# Using bash -c for complex commands
curl -X POST http://localhost:8000/api/v1/tasks -d '{
  "command": "bash -c \"for i in {1..5}; do echo Step $i; done\""
}'
```

### Failing Commands

```bash
# Command that fails (non-existent path)
curl -X POST http://localhost:8000/api/v1/tasks -d '{
  "command": "ls /nonexistent/directory"
}'

# Execute and check results
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": "ls /nonexistent/directory"}' | jq -r '.id')

JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute | jq -r '.job_id')

# Wait and check artifact
sleep 1
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.artifact_id' | \
  xargs -I {} curl http://localhost:8000/api/v1/artifacts/{} | jq '.data'

# Output shows:
# - exit_code: non-zero (e.g., 1 or 2)
# - stderr: error message about missing directory
# - Job status: "completed" (job succeeded even though command failed)
```

**Note:** Job status is `completed` even if command fails. Check `exit_code` in artifact to determine command success.

### Capturing stderr

```bash
# Write to stderr
curl -X POST http://localhost:8000/api/v1/tasks -d '{
  "command": ">&2 echo \"error message\""
}'

# Execute and check stderr in artifact
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": ">&2 echo \"error message\""}' | jq -r '.id')
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute | jq -r '.job_id')

sleep 1
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.artifact_id' | \
  xargs -I {} curl http://localhost:8000/api/v1/artifacts/{} | jq '.data.stderr'
```

### Concurrent Execution

```bash
# Create task
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -d '{"command": "sleep 2 && echo done"}' | jq -r '.id')

# Execute multiple times concurrently
for i in {1..5}; do
  curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute &
done
wait

# List all jobs to see concurrent executions
curl http://localhost:8000/api/v1/jobs | jq
```

---

## Error Handling

### Task Not Found (400)

```bash
curl -X POST http://localhost:8000/api/v1/tasks/01K72P5N5KCRM6MD3BRE4P0999/\$execute
```

**Response:**
```json
{
  "detail": "Task 01K72P5N5KCRM6MD3BRE4P0999 not found"
}
```

### Invalid ULID (400)

```bash
curl -X POST http://localhost:8000/api/v1/tasks/invalid-id/\$execute
```

**Response:**
```json
{
  "type": "urn:chapkit:error:invalid-ulid",
  "title": "Invalid ULID",
  "status": 400,
  "detail": "Invalid ULID format: invalid-id"
}
```

### Missing Dependencies (409)

If `.with_artifacts()` or `.with_jobs()` not configured:

```json
{
  "detail": "Task execution requires artifacts. Use ServiceBuilder.with_artifacts() before with_tasks()."
}
```

### Validation Errors (422)

```bash
# Missing command field
curl -X POST http://localhost:8000/api/v1/tasks -d '{}'
```

**Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "command"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Testing

### Manual Testing

**Terminal 1: Start service**
```bash
fastapi dev examples/task_execution_api.py
```

**Terminal 2: Test workflow**
```bash
# Create task
TASK_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"command": "echo \"test\""}' | jq -r '.id')

echo "Task ID: $TASK_ID"

# Execute task
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/\$execute | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Monitor job status
curl -N http://localhost:8000/api/v1/jobs/$JOB_ID/\$stream

# Get results
ARTIFACT_ID=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.artifact_id')
curl http://localhost:8000/api/v1/artifacts/$ARTIFACT_ID | jq '.data'
```

### Automated Testing

```python
import httpx
import time
from collections.abc import Generator

def wait_for_job_completion(
    client: httpx.Client,
    job_id: str,
    timeout: float = 5.0
) -> dict:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = client.get(f"/api/v1/jobs/{job_id}")
        job = response.json()

        if job["status"] in ["completed", "failed", "canceled"]:
            return job

        time.sleep(0.1)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_task_execution():
    """Test creating and executing a task."""
    with httpx.Client(base_url="http://localhost:8000") as client:
        # Create task
        response = client.post(
            "/api/v1/tasks",
            json={"command": "echo 'test'"}
        )
        assert response.status_code == 201
        task = response.json()
        task_id = task["id"]

        # Execute task
        response = client.post(f"/api/v1/tasks/{task_id}/$execute")
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        # Wait for completion
        job = wait_for_job_completion(client, job_id)
        assert job["status"] == "completed"
        assert job["artifact_id"] is not None

        # Check artifact
        response = client.get(f"/api/v1/artifacts/{job['artifact_id']}")
        artifact = response.json()

        assert artifact["data"]["task"]["id"] == task_id
        assert "test" in artifact["data"]["stdout"]
        assert artifact["data"]["exit_code"] == 0
```

### Pytest with TestClient

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient with lifespan context."""
    from examples.task_execution_api import app

    with TestClient(app) as test_client:
        yield test_client


def test_create_task(client: TestClient) -> None:
    """Test creating a new task template."""
    response = client.post(
        "/api/v1/tasks",
        json={"command": "echo 'test'"}
    )
    assert response.status_code == 201
    task = response.json()

    assert "id" in task
    assert task["command"] == "echo 'test'"
    assert "created_at" in task


def test_execute_task(client: TestClient) -> None:
    """Test executing a task and retrieving results."""
    # Create task
    response = client.post(
        "/api/v1/tasks",
        json={"command": "echo 'Hello World'"}
    )
    task_id = response.json()["id"]

    # Execute
    response = client.post(f"/api/v1/tasks/{task_id}/$execute")
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # Wait for completion
    import time
    time.sleep(1)

    # Get results
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    artifact = client.get(f"/api/v1/artifacts/{job['artifact_id']}").json()

    assert "Hello World" in artifact["data"]["stdout"]
    assert artifact["data"]["exit_code"] == 0
```

---

## Production Deployment

### Concurrency Control

Limit concurrent task executions to prevent resource exhaustion:

```python
app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)  # Max 5 tasks running simultaneously
    .with_tasks()
    .build()
)
```

**Recommendations:**
- **CPU-bound tasks**: Set to number of CPU cores (e.g., 4-8)
- **I/O-bound tasks**: Higher limits OK (10-20)
- **Memory-intensive**: Lower limits to prevent OOM (2-5)
- **Long-running**: Consider lower limits (3-5)

### Database Configuration

Use persistent database for production:

```python
app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_database("/data/tasks.db")  # Persistent storage
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks()
    .build()
)
```

**Best Practices:**
- Mount `/data` volume in Docker/Kubernetes
- Regular backups of task templates and artifacts
- Monitor database size (artifacts can grow)

### Task Retention

Implement cleanup for old execution artifacts:

```python
from datetime import datetime, timedelta
from chapkit.api import ServiceBuilder

async def cleanup_old_artifacts(app):
    """Remove artifacts older than 30 days."""
    artifact_manager = app.state.artifact_manager

    cutoff_date = datetime.utcnow() - timedelta(days=30)

    # Implementation depends on your retention policy
    # Consider using artifact metadata or timestamps
    pass

app = (
    ServiceBuilder(info=info)
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks()
    .on_startup(cleanup_old_artifacts)  # Run on startup
    .build()
)
```

### Monitoring

Track task execution metrics:

```python
from prometheus_client import Counter, Histogram

task_executions = Counter(
    'task_executions_total',
    'Total task executions',
    ['status']
)

task_duration = Histogram(
    'task_duration_seconds',
    'Task execution duration'
)

# Combine with monitoring feature
app = (
    ServiceBuilder(info=info)
    .with_monitoring()  # Prometheus metrics at /metrics
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks()
    .build()
)
```

### Security Considerations

**Command Injection Prevention:**

Tasks execute arbitrary shell commands. Implement access controls using CRUD permissions:

```python
from servicekit.api.crud import CrudPermissions
from chapkit.api import ServiceBuilder, ServiceInfo

# Read-only task API (tasks created only via code)
task_permissions = CrudPermissions(
    allow_create=False,    # Disable runtime task creation
    allow_read=True,       # Allow reading tasks
    allow_update=False,    # Disable runtime updates
    allow_delete=False,    # Disable deletion
)

app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_database("tasks.db")
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks(permissions=task_permissions)  # Apply permissions
    .build()
)
```

**Read-Only API Pattern:**

With read-only permissions, all tasks are pre-seeded at startup:

```python
from chapkit import TaskIn, TaskManager

async def seed_tasks(app):
    """Pre-seed task templates on startup."""
    task_manager = app.state.task_manager

    # Define tasks programmatically
    tasks = [
        TaskIn(command="echo 'System health check'", enabled=True),
        TaskIn(command="python3 /app/backup.py", enabled=True),
        TaskIn(command="process_data", task_type="python",
               parameters={"batch_size": 100}, enabled=True),
    ]

    for task in tasks:
        await task_manager.save(task)

app = (
    ServiceBuilder(info=info)
    .with_database("tasks.db")
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks(permissions=CrudPermissions(
        allow_create=False,
        allow_read=True,
        allow_update=False,
        allow_delete=False,
    ))
    .on_startup(seed_tasks)
    .build()
)
```

**Benefits:**
- Tasks defined in code (version controlled)
- No runtime command injection risk
- API users can only execute pre-defined tasks
- Tasks can be audited before deployment
- Enables GitOps workflow for task management

**Recommendations:**
- Use read-only API for production (pre-seed tasks at startup)
- Apply authentication (`.with_auth()`) for execution endpoint
- Validate commands in seeding logic
- Run service with limited OS user permissions
- Use container security (no privileged mode)
- Monitor execution logs for suspicious activity
- Use `validate_and_disable_orphaned_tasks` to prevent broken Python tasks

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Copy application
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 taskuser && \
    chown -R taskuser:taskuser /app

USER taskuser

# Run service
CMD ["fastapi", "run", "examples/task_execution_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  task-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - task-data:/data
    environment:
      - DATABASE_URL=/data/tasks.db
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G

volumes:
  task-data:
```

---

## Troubleshooting

### Job Status is "completed" but Command Failed

**Problem:** Job shows `status: completed` but command actually failed.

**Cause:** Job execution succeeded (subprocess ran), but command returned non-zero exit code.

**Solution:** Check `exit_code` in artifact:

```bash
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.artifact_id' | \
  xargs -I {} curl http://localhost:8000/api/v1/artifacts/{} | \
  jq '.data.exit_code'

# exit_code == 0: command succeeded
# exit_code != 0: command failed
```

### "Task execution requires artifacts" Error

**Problem:** `ValueError: Task execution requires artifacts`

**Solution:** Call `.with_artifacts()` before `.with_tasks()`:

```python
# Wrong order
.with_tasks()
.with_artifacts(hierarchy=TASK_HIERARCHY)  # Too late!

# Correct order
.with_artifacts(hierarchy=TASK_HIERARCHY)
.with_tasks()
```

### "Task execution requires a scheduler" Error

**Problem:** `ValueError: Task execution requires a scheduler`

**Solution:** Call `.with_jobs()` before `.with_tasks()`:

```python
# Wrong order
.with_tasks()
.with_jobs()  # Too late!

# Correct order
.with_jobs(max_concurrency=5)
.with_tasks()
```

### Jobs Not Executing (Stuck in "pending")

**Problem:** Jobs remain in `pending` state indefinitely.

**Causes:**
1. Reached `max_concurrency` limit
2. Scheduler not started properly
3. Long-running jobs blocking queue

**Solution:**
```bash
# Check running jobs
curl http://localhost:8000/api/v1/jobs?status_filter=running | jq 'length'

# If at max_concurrency, wait for completion or increase limit
# Restart service to reset scheduler if needed
```

### Artifact Not Created

**Problem:** `Job.artifact_id` is `null` after completion.

**Cause:** Job failed during execution (before artifact creation).

**Solution:** Check job error:

```bash
curl http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.error'
```

### Command Not Found in Container

**Problem:** Task works locally but fails in Docker with "command not found".

**Cause:** Command not installed in container image.

**Solution:** Install required tools in Dockerfile:

```dockerfile
# Install common utilities
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    python3 \
    && rm -rf /var/lib/apt/lists/*
```

### Orphaned Python Tasks

**Problem:** Python task references a function that was removed or renamed from the registry.

**Cause:** Function was removed or renamed but task template still references the old name.

**Automatic Disabling (Recommended):**

Chapkit provides a startup validation utility that automatically disables orphaned Python tasks:

```python
from chapkit import validate_and_disable_orphaned_tasks
from chapkit.api import ServiceBuilder, ServiceInfo

app = (
    ServiceBuilder(info=ServiceInfo(display_name="Task Service"))
    .with_database("tasks.db")
    .with_artifacts(hierarchy=TASK_HIERARCHY)
    .with_jobs(max_concurrency=5)
    .with_tasks()
    .on_startup(validate_and_disable_orphaned_tasks)
    .build()
)
```

**Behavior:**
- Checks all Python tasks against `TaskRegistry` on startup
- Automatically disables tasks referencing unregistered functions
- Logs warnings for each orphaned task with task IDs and function names
- Preserves task history (soft-delete via `enabled=False`)
- Returns count of disabled tasks

**Example Log Output:**
```
WARNING Found orphaned Python tasks - disabling them
  count: 2
  task_ids: ['01TASK1...', '01TASK2...']
  commands: ['old_function', 'removed_function']
INFO Disabling orphaned task 01TASK1...: function 'old_function' not found in registry
INFO Disabling orphaned task 01TASK2...: function 'removed_function' not found in registry
WARNING Disabled 2 orphaned Python task(s)
```

**Filtering Disabled Tasks:**
```bash
# List all disabled tasks
curl http://localhost:8000/api/v1/tasks?enabled=false

# List only enabled tasks
curl http://localhost:8000/api/v1/tasks?enabled=true
```

**Re-enabling Tasks:**
If you re-register the function, you can re-enable the task:

```python
# Re-register the function
@TaskRegistry.register("old_function")
def old_function(**params) -> dict:
    return {"result": "restored"}
```

```bash
# Re-enable the task
curl -X PUT http://localhost:8000/api/v1/tasks/{task_id} \
  -H "Content-Type: application/json" \
  -d '{
    "command": "old_function",
    "task_type": "python",
    "enabled": true
  }'
```

**Alternative Solutions:**

**Option 1: Keep deprecated functions with errors**
```python
@TaskRegistry.register("old_function")
def old_function(**params) -> dict:
    """Deprecated - use new_function instead."""
    raise NotImplementedError("This function has been removed. Use new_function instead.")
```

**Option 2: Manual deletion**
```bash
# Find orphaned tasks
curl http://localhost:8000/api/v1/tasks?enabled=false | \
  jq '.[] | select(.task_type == "python")'

# Delete specific task
curl -X DELETE http://localhost:8000/api/v1/tasks/{task_id}
```

**Best Practices:**
- Always use `validate_and_disable_orphaned_tasks` on startup (production ready)
- Monitor logs for orphaned task warnings
- Consider versioning function names (e.g., `process_data_v1`, `process_data_v2`)
- Document which tasks depend on which functions
- Periodically review disabled tasks for cleanup

---

## Next Steps

- **Job Monitoring:** Use `.with_jobs()` SSE streaming for real-time task progress
- **ML Workflows:** Combine with `.with_ml()` for ML training tasks
- **Authentication:** Secure with `.with_auth()` for production
- **Monitoring:** Track execution metrics with `.with_monitoring()`

For more examples:
- `examples/task_execution_api.py` - Shell task execution service
- `examples/python_task_execution_api.py` - Python task execution with TaskRegistry
- `tests/test_example_task_execution_api.py` - Shell task test suite
- `tests/test_example_python_task_execution_api.py` - Python task test suite
- `docs/guides/job-scheduler.md` - Job scheduler and SSE streaming
