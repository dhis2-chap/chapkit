# Class-Based ML Runner Example

Demonstrates an OOP-style ML workflow by subclassing `BaseModelRunner`, with scikit-learn preprocessing.

What it shows:

- Subclassing `BaseModelRunner` instead of using `FunctionalModelRunner`
- Lifecycle hooks (`on_init`, `on_cleanup`) and shared state between train and predict
- Feature preprocessing with `StandardScaler`, toggled by config
- Input validation via `on_validate_train` (surfaced by `POST /api/v1/ml/$validate`)
- Prometheus metrics via `.with_monitoring()` (exposed at `/metrics`)

Run:

```bash
uv run python main.py
```

Then browse http://127.0.0.1:8000/docs. Create a config, then `POST /api/v1/ml/$train` and
`POST /api/v1/ml/$predict`; poll `/api/v1/jobs/{id}` for completion.
