# python_task_execution_api.py - Python Task Execution cURL Guide

Task execution service demonstrating Python function registration, task parameters, enable/disable controls, and orphaned task validation.

## Quick Start

```bash
# Start the service
fastapi dev examples/python_task_execution_api.py

# Service available at: http://127.0.0.1:8000
```

## Features

- **Python Functions**: Register sync and async functions as executable tasks
- **Parameters**: Pass JSON parameters as kwargs to Python functions
- **Task Types**: Both "shell" and "python" tasks supported
- **Enable/Disable**: Control task execution with enabled flag
- **Orphaned Tasks**: Automatic validation and disabling on startup
- **Task Registry**: Type-safe function registration with @TaskRegistry.register()

## Complete Workflow

### 1. Check Service Health

```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy"
  }
}
```

### 2. List All Tasks

```bash
curl http://127.0.0.1:8000/api/v1/tasks
```

**Response:**
```json
[
  {
    "id": "01JCSEED0000000000000PYTH1",
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 10, "b": 32},
    "enabled": true,
    "created_at": "2025-10-17T10:00:00Z",
    "updated_at": "2025-10-17T10:00:00Z"
  },
  {
    "id": "01JCSEED0000000000000PYTH2",
    "command": "process_data",
    "task_type": "python",
    "parameters": {"input_text": "Hello World", "uppercase": true},
    "enabled": true,
    "created_at": "2025-10-17T10:00:00Z",
    "updated_at": "2025-10-17T10:00:00Z"
  }
]
```

### 3. Filter Tasks by Status

```bash
# Only enabled tasks
curl "http://127.0.0.1:8000/api/v1/tasks?enabled=true"

# Only disabled tasks
curl "http://127.0.0.1:8000/api/v1/tasks?enabled=false"
```

### 4. Get Specific Task

```bash
curl http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH1
```

**Response:**
```json
{
  "id": "01JCSEED0000000000000PYTH1",
  "command": "calculate_sum",
  "task_type": "python",
  "parameters": {"a": 10, "b": 32},
  "enabled": true,
  "created_at": "2025-10-17T10:00:00Z",
  "updated_at": "2025-10-17T10:00:00Z"
}
```

### 5. Execute Python Task (Async Function)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH1/\$execute
```

**Response:**
```json
{
  "job_id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "message": "Task submitted for execution. Job ID: 01K79YAHJ7BR4E87VVTG8FNBMA"
}
```

### 6. Poll Job Status

```bash
# Poll every 1-2 seconds until status is "completed"
curl http://127.0.0.1:8000/api/v1/jobs/01K79YAHJ7BR4E87VVTG8FNBMA
```

**Response (pending):**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "status": "pending",
  "artifact_id": null,
  "created_at": "2025-10-17T10:01:00Z",
  "updated_at": "2025-10-17T10:01:00Z"
}
```

**Response (completed):**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMA",
  "status": "completed",
  "artifact_id": "01K79YAHJ7BR4E87VVTG8FNBMB",
  "created_at": "2025-10-17T10:01:00Z",
  "updated_at": "2025-10-17T10:01:02Z"
}
```

### 7. Get Python Task Results

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/01K79YAHJ7BR4E87VVTG8FNBMB
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMB",
  "parent_id": null,
  "level": 0,
  "data": {
    "task": {
      "id": "01JCSEED0000000000000PYTH1",
      "command": "calculate_sum",
      "task_type": "python",
      "parameters": {"a": 10, "b": 32},
      "created_at": "2025-10-17T10:00:00Z",
      "updated_at": "2025-10-17T10:00:00Z"
    },
    "result": {
      "result": 42,
      "operation": "sum"
    },
    "error": null
  },
  "created_at": "2025-10-17T10:01:02Z",
  "updated_at": "2025-10-17T10:01:02Z"
}
```

### 8. Execute Sync Python Task

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH2/\$execute
```

Wait for job completion, then get results:

```bash
curl http://127.0.0.1:8000/api/v1/artifacts/ARTIFACT_ID
```

**Response:**
```json
{
  "data": {
    "task": {...},
    "result": {
      "original": "Hello World",
      "processed": "HELLO WORLD",
      "length": 11,
      "timestamp": "2025-10-17T10:05:00Z"
    },
    "error": null
  }
}
```

### 9. Execute Shell Task (For Comparison)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH5/\$execute
```

**Shell Task Artifact (Different Structure):**
```json
{
  "data": {
    "task": {
      "id": "01JCSEED0000000000000PYTH5",
      "command": "echo \"This is a shell task\"",
      "created_at": "2025-10-17T10:00:00Z",
      "updated_at": "2025-10-17T10:00:00Z"
    },
    "stdout": "This is a shell task\n",
    "stderr": "",
    "exit_code": 0
  }
}
```

### 10. Error Handling - Execute Failing Task

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH4/\$execute
```

**Artifact with Error:**
```json
{
  "data": {
    "task": {...},
    "result": null,
    "error": {
      "type": "ValueError",
      "message": "This task was designed to fail",
      "traceback": "Traceback (most recent call last):\n  File \"...\", line 48, in failing_task\n    raise ValueError(\"This task was designed to fail\")\nValueError: This task was designed to fail"
    }
  }
}
```

### 11. Try to Execute Disabled Task

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH6/\$execute
```

**Response (400 Bad Request):**
```json
{
  "detail": "Cannot execute disabled task 01JCSEED0000000000000PYTH6"
}
```

## Creating Tasks

### Create Python Task with Parameters

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 25, "b": 17},
    "enabled": true
  }'
```

**Response:**
```json
{
  "id": "01K79YAHJ7BR4E87VVTG8FNBMC",
  "command": "calculate_sum",
  "task_type": "python",
  "parameters": {"a": 25, "b": 17},
  "enabled": true,
  "created_at": "2025-10-17T10:10:00Z",
  "updated_at": "2025-10-17T10:10:00Z"
}
```

### Create Shell Task

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "date",
    "task_type": "shell",
    "enabled": true
  }'
```

### Create Disabled Task

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "command": "process_data",
    "task_type": "python",
    "parameters": {"input_text": "test", "uppercase": false},
    "enabled": false
  }'
```

## Updating Tasks

### Enable a Disabled Task

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH6 \
  -H "Content-Type: application/json" \
  -d '{
    "command": "process_data",
    "task_type": "python",
    "parameters": {"input_text": "Disabled", "uppercase": false},
    "enabled": true
  }'
```

### Disable an Enabled Task

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH1 \
  -H "Content-Type: application/json" \
  -d '{
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 10, "b": 32},
    "enabled": false
  }'
```

### Update Task Parameters

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/tasks/01JCSEED0000000000000PYTH1 \
  -H "Content-Type: application/json" \
  -d '{
    "command": "calculate_sum",
    "task_type": "python",
    "parameters": {"a": 100, "b": 200},
    "enabled": true
  }'
```

## Deleting Tasks

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/tasks/01K79YAHJ7BR4E87VVTG8FNBMC
```

**Response:** `204 No Content`

**Note:** Deleting a task preserves all execution artifacts in the database.

## Advanced Workflows

### Stream Job Progress (SSE)

```bash
curl -N http://127.0.0.1:8000/api/v1/jobs/01K79YAHJ7BR4E87VVTG8FNBMA/\$stream
```

**Response (Server-Sent Events):**
```
data: {"id": "01K79YAHJ7BR4E87VVTG8FNBMA", "status": "pending", ...}

data: {"id": "01K79YAHJ7BR4E87VVTG8FNBMA", "status": "running", ...}

data: {"id": "01K79YAHJ7BR4E87VVTG8FNBMA", "status": "completed", ...}
```

### List All Jobs

```bash
# All jobs
curl http://127.0.0.1:8000/api/v1/jobs

# Filter by status
curl "http://127.0.0.1:8000/api/v1/jobs?status_filter=completed"
curl "http://127.0.0.1:8000/api/v1/jobs?status_filter=failed"
curl "http://127.0.0.1:8000/api/v1/jobs?status_filter=pending"
```

### Pagination

```bash
# Paginate tasks
curl "http://127.0.0.1:8000/api/v1/tasks?page=1&size=10"

# Paginate artifacts
curl "http://127.0.0.1:8000/api/v1/artifacts?page=1&size=20"
```

### Find Task Executions

```bash
# Get all artifacts (includes task execution results)
curl http://127.0.0.1:8000/api/v1/artifacts

# Filter by task ID in application code or jq
curl -s http://127.0.0.1:8000/api/v1/artifacts | \
  jq '[.[] | select(.data.task.id == "01JCSEED0000000000000PYTH1")]'
```

## Task Registry Examples

These Python functions are pre-registered in the example:

### 1. calculate_sum (Async)
- **Parameters**: `a: int, b: int`
- **Returns**: `{"result": <sum>, "operation": "sum"}`
- **Type**: Async function

### 2. process_data (Sync)
- **Parameters**: `input_text: str, uppercase: bool = False`
- **Returns**: `{"original": str, "processed": str, "length": int, "timestamp": str}`
- **Type**: Sync function (runs in thread pool)

### 3. slow_computation (Sync)
- **Parameters**: `seconds: int = 2`
- **Returns**: `{"completed": true, "duration_seconds": int}`
- **Type**: Sync function with blocking sleep

### 4. failing_task (Async)
- **Parameters**: `should_fail: bool = True`
- **Returns**: `{"success": true}` or raises ValueError
- **Type**: Async function for error handling demo

## Orphaned Task Behavior

The service automatically validates Python tasks on startup:

1. **Checks**: All Python tasks against TaskRegistry
2. **Detects**: Tasks referencing unregistered functions
3. **Disables**: Orphaned tasks automatically (enabled = false)
4. **Logs**: Warnings with task IDs and function names

**Example Log Output:**
```
WARNING Found orphaned Python tasks - disabling them
  count: 1
  task_ids: ['01JCSEED0000000000000PYTH7']
  commands: ['nonexistent_function']
INFO Disabling orphaned task 01JCSEED0000000000000PYTH7: function 'nonexistent_function' not found in registry
WARNING Disabled 1 orphaned Python task(s)
```

**Check Disabled Tasks:**
```bash
curl "http://127.0.0.1:8000/api/v1/tasks?enabled=false"
```

## Python vs Shell Tasks Comparison

| Feature | Shell Tasks | Python Tasks |
|---------|-------------|--------------|
| **task_type** | "shell" | "python" |
| **command** | Shell command string | Registered function name |
| **parameters** | Not used | JSON dict passed as kwargs |
| **Success output** | stdout, stderr, exit_code | result (any JSON-serializable) |
| **Error output** | stderr text | Full exception with traceback |
| **Success check** | exit_code == 0 | error == null |
| **Execution** | asyncio.subprocess | Direct function call |
| **Registration** | Not required | Required via TaskRegistry |

## Tips

1. **Parameters**: Always passed as kwargs - function signature must match parameter keys
2. **Sync Functions**: Automatically run in thread pool via asyncio.to_thread()
3. **Error Handling**: Python exceptions captured with full traceback
4. **Task Snapshot**: Artifacts preserve task state at execution time (immutable)
5. **Orphaned Tasks**: Re-register function and re-enable task to fix
6. **Disabled Tasks**: Cannot execute but remain visible for auditing

## Troubleshooting

### "Python function 'xxx' not found in registry"

**Problem:** Function not registered or service restarted without registration

**Solution:**
```python
# Re-register the function
@TaskRegistry.register("xxx")
def xxx(**params) -> dict:
    return {"result": "ok"}
```

### "Cannot execute disabled task"

**Problem:** Task has `enabled: false`

**Solution:**
```bash
# Re-enable the task
curl -X PUT http://127.0.0.1:8000/api/v1/tasks/TASK_ID \
  -H "Content-Type: application/json" \
  -d '{...task data..., "enabled": true}'
```

### TypeError on function execution

**Problem:** Parameters don't match function signature

**Solution:** Ensure parameter keys match function argument names exactly:
```python
# Function expects 'a' and 'b'
def calculate_sum(a: int, b: int) -> dict: ...

# Parameters must use same names
{"a": 10, "b": 32}  # Correct
{"x": 10, "y": 32}  # Wrong - TypeError
```

### Job stays "pending"

**Problem:**
1. Reached max_concurrency limit (default: 3)
2. Long-running jobs blocking queue

**Solution:**
```bash
# Check running jobs
curl "http://127.0.0.1:8000/api/v1/jobs?status_filter=running"

# Wait for jobs to complete or increase max_concurrency in code
```

## Next Steps

- Try **[readonly_task_api.py](../readonly_task_api.py)** for read-only security pattern
- Read **[task-execution.md](../../docs/guides/task-execution.md)** for complete API reference
- Check **[../python_task_execution_api.py](../python_task_execution_api.py)** source code
- See **[../../CLAUDE.md](../../CLAUDE.md)** for architecture overview
