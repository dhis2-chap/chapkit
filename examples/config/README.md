# Config Example

Demonstrates config management with startup seeding and extended service info.

What it shows:

- Defining a typed config schema by subclassing `BaseConfig`
- Seeding configs on startup via `.on_startup(...)` with `ConfigManager` / `ConfigRepository`
- Extending `ServiceInfo` with custom metadata (author, contact email, config schema)
- The config CRUD surface at `/api/v1/configs`

Run:

```bash
uv run python main.py
```

Then browse http://127.0.0.1:8000/docs and list the seeded configs at `GET /api/v1/configs`.
