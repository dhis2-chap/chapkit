# Artifact Example

Demonstrates artifact hierarchies with config linking, a read-only artifact API, and non-JSON payloads.

What it shows:

- Defining an `ArtifactHierarchy` with labeled levels (`train` / `predict` / `result`)
- Seeding artifact trees with parent-child relationships on startup
- Linking configs to root artifacts for experiment tracking (`$artifacts`, `$config`)
- Storing arbitrary Python objects (pickled) as artifact data, and the `$metadata` operation
- Disabling create/update/delete to expose a read-only API
- Custom health checks via `.with_health(checks=...)`

Run:

```bash
uv run python main.py
```

Then browse http://127.0.0.1:8000/docs. Try `GET /api/v1/artifacts/{id}/$tree` on a seeded root artifact.
