# Task System Simplification

**Status:** Draft
**Date:** 2025-11-07

## Problem

Current workflow is convoluted:
1. Register function in TaskRegistry (in-memory)
2. Create task record in database (persistent)
3. Execute by ULID

**Issues:**
- Duplicate tracking (registry + database)
- Manual database record creation required
- No validation at creation time
- Parameter confusion (stored vs runtime)
- Three ID layers: name → task ULID → job ULID

## Solution

**Registry-based execution:**
1. Register: `@TaskRegistry.register("task_name")`
2. Execute: `POST /api/v1/tasks/task_name/$execute {"params": {...}}`

TaskRegistry is single source of truth. No database layer for task definitions.

## Tags Support

Tasks can be tagged for organization and filtering:

```python
@TaskRegistry.register("process_data", tags=["data", "etl"])
@TaskRegistry.register("clean_data", tags=["data", "preprocessing"])
@TaskRegistry.register("send_email", tags=["notifications"])
```

**Filter by tags:**
```
GET /api/v1/tasks?tags=data
GET /api/v1/tasks?tags=data,etl  (tasks with ALL specified tags)
```

## API Changes

**New endpoints:**
- `GET /api/v1/tasks` - List all registered functions (supports `?tags=` filter)
- `GET /api/v1/tasks/{name}` - Get function metadata
- `POST /api/v1/tasks/{name}/$execute` - Execute with runtime params

**Removed endpoints:**
- `POST /api/v1/tasks` - Create task
- `PUT /api/v1/tasks/{id}` - Update task
- `DELETE /api/v1/tasks/{id}` - Delete task

**Naming rules:**
- URL-safe names only: `^[a-zA-Z0-9_-]+$`
- Examples: `greet_user`, `process-data`, `task123`

## Implementation

**Delete:**
- `src/chapkit/task/models.py`
- `src/chapkit/task/repository.py`
- `src/chapkit/task/manager.py`
- `src/chapkit/task/validation.py`

**Create:**
- `src/chapkit/task/executor.py`

**Modify:**
- `src/chapkit/task/registry.py` - Add tags support, `get_info()`, `list_all_info()`, `list_by_tags()`
- `src/chapkit/task/router.py` - Replace CRUD with 3 endpoints, add tag filtering
- `src/chapkit/task/schemas.py` - Remove TaskIn/TaskOut, add TaskInfo with tags field
- `src/chapkit/task/__init__.py` - Update exports
- `src/chapkit/cli/templates/main_task.py.jinja2` - Remove seed functions
- `examples/task_execution/main.py` - Remove seed functions
- `docs/guides/task-execution.md` - Complete rewrite
- `src/chapkit/cli/templates/postman_collection_task.json.jinja2` - Remove CRUD
- `tests/test_task_*.py` - Update tests

**Database:**
- Create migration to drop task table

**Version:**
- Bump to 0.7.0 (breaking change)

## Breaking Changes

**Removed:**
- Task database model
- CRUD endpoints (POST, PUT, DELETE /api/v1/tasks)
- Shell command support
- Task ULIDs (use function names)
- Stored parameters

**Migration:**

Before:
```python
# Register + create + execute
@TaskRegistry.register("my_task")
async def my_task(param: str) -> dict: ...

task = await manager.save(TaskIn(command="my_task", parameters={"param": "value"}))
job_id = await manager.execute_task(task.id)
```

After:
```python
# Register + execute
@TaskRegistry.register("my_task")
async def my_task(param: str) -> dict: ...

POST /api/v1/tasks/my_task/$execute {"params": {"param": "value"}}
```

## Notes

- Dependency injection preserved
- Artifact storage preserved
- Shell commands: use `subprocess` in Python wrapper
- Task disabling: unregister or add condition in function
- Hot reload: `TaskRegistry.clear()` before re-registration

## Open Questions

1. **Task enable/disable support?**
   - Current: removed (unregister function instead)
   - Alternative: decorator flag `@TaskRegistry.register("task", enabled=False)`

2. **Task discovery in multi-module apps?**
   - Import all task modules explicitly
   - Or: auto-discover tasks in specific directory

3. **Task name namespaces?**
   - Example: `user.create`, `user.delete`
   - Would allow grouping related tasks
