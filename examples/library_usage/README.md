# Library Usage Example

Demonstrates using chapkit as a library: adding your own ORM entity alongside chapkit's modules.

What it shows:

- Defining a custom `Entity` ORM model (`User`) with its own table
- Writing a custom `BaseRepository` and `BaseManager` subclass with domain queries
- Generating REST endpoints for the custom model with `CrudRouter.create(...)`
- Combining the custom router with chapkit's config module in one `ServiceBuilder` app
- Seeding both configs and custom entities on startup

Run:

```bash
uv run python main.py
```

Then browse http://127.0.0.1:8000/docs — both `/api/v1/configs` and `/api/v1/users` are available.
