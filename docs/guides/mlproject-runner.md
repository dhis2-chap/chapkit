# MLproject Runner

`chapkit run` turns any directory containing an MLflow-style `MLproject` file into a running chapkit service — no code generation, no changes to the MLproject repo. It is aimed at users coming to chapkit with an existing `train.r` / `predict.py` (or any shell-command-driven model) who want a chapkit HTTP API around their scripts in seconds.

## Quick Start

Point it at a directory. All three forms work:

```bash
chapkit run              # uses current directory
chapkit run .            # same
chapkit run /path/to/my_mlproject
```

This parses the `MLproject` file, translates the entry-point commands to chapkit's workspace conventions, builds a FastAPI service with `/api/v1/ml/$train` and `/$predict` endpoints, and serves on `127.0.0.1:8000` by default. Override host/port with `--host` and `--port`.

A minimal R MLproject like `dhis2-chap/minimalist_example_r`:

```yaml
name: minimalist_r
renv_env: renv.lock
entry_points:
  train:
    parameters:
      train_data: path
      model: str
    command: "Rscript train.r {train_data} {model}"
  predict:
    parameters:
      historic_data: path
      future_data: path
      model: str
      out_file: path
    command: "Rscript predict.r {model} {historic_data} {future_data} {out_file}"
```

...becomes a service that accepts `POST /api/v1/ml/$train` (with your CSV as a `DataFrame` payload), runs `Rscript train.r data.csv model` in an isolated workspace, and stores the resulting workspace as an artifact. Predict works the same way, re-entering the training workspace.

---

## Canonical Parameter Mapping

`chapkit run` recognises the canonical MLproject parameter names used across chap-core-compatible models. Each is substituted with a fixed filename that matches chapkit's `ShellModelRunner` workspace layout:

| MLproject parameter | Substitutes to  | Notes                                                                 |
| ------------------- | --------------- | --------------------------------------------------------------------- |
| `train_data`        | `data.csv`      | Training data CSV written by chapkit                                  |
| `historic_data`     | `historic.csv`  | Historic data CSV during predict                                      |
| `future_data`       | `future.csv`    | Future-period data CSV during predict                                 |
| `out_file`          | `predictions.csv` | Where your script writes predictions                                 |
| `model`             | `model`         | Literal path; your script saves to and loads from it. The file (or directory) persists across train → predict via workspace copy. |
| `model_config`      | `config.yml`    | Config YAML chapkit writes from `user_options` + `prediction_periods` |
| `polygons`          | `geo.json`      | Optional GeoJSON for spatial models                                   |

These names are the lingua franca used by chap-core (`chap_core/runners/command_line_runner.py`) so any MLproject that already runs under chap-core is expected to run under `chapkit run` without change.

### Overriding the Map

If your MLproject uses a non-canonical placeholder (e.g. `{dataset}`), provide an override at launch:

```bash
chapkit run . --param dataset=data.csv
```

The `--param NAME=FILENAME` flag is repeatable. Overrides win over the canonical map, which lets you re-point `{model}` or any other parameter if your scripts expect a different filename.

---

## Dynamic Config from `user_options`

MLproject `user_options` become typed fields on a chapkit `BaseConfig` subclass, generated at startup with `pydantic.create_model`. Given:

```yaml
name: ewars_template
user_options:
  n_lags:
    type: integer
    default: 3
    description: Number of lags to include in the model.
  precision:
    type: number
    default: 0.01
    description: Prior on the precision of fixed effects.
```

`chapkit run` builds an `ewars_templateConfig` with `n_lags: int = 3`, `precision: float = 0.01`, and the standard `prediction_periods: int = 3` injected automatically. Scripts read these values from `config.yml`, which chapkit writes to the workspace root before invoking your train/predict command.

Supported `type` values: `integer`/`int`, `number`/`float`, `string`/`str`, `boolean`/`bool`, `path` (treated as string). Unknown types fall back to `str`. Options without a `default` become required fields.

---

## Environment Hints

If your MLproject declares a runtime environment (`docker_env`, `renv_env`, `python_env`, `uv_env`, `conda_env`), `chapkit run` **warns** about it on startup but does **not** auto-activate it:

```
WARNING: chapkit run does not auto-activate environments.
  - uv_env: pyproject.toml
Activate the right runtime (R/renv, conda, Docker image, etc.) before launching
chapkit run, or invoke chapkit run from inside it.
```

Because `chapkit run` shells out `python ...` / `Rscript ...` via `ShellModelRunner`, the subprocess inherits whatever is on `PATH` at launch time. Invoking the chapkit entry point directly (e.g. `./.venv/bin/chapkit run .`) does **not** put `.venv/bin` on subprocess `PATH`, so `import pandas` (or similar) will fail.

Recommended invocations:

- **Python MLproject:** `uv run chapkit run .` (preferred) or `source .venv/bin/activate && chapkit run .`
- **R MLproject:** run `chapkit run .` from inside an R-capable container or an `renv`-activated shell.

The published container images (below) set `PATH` correctly so this is a non-issue there.

---

## Running in a Container

Chapkit publishes three base images for `chapkit run`, all built on `debian:trixie-slim` and owned entirely by this repository (no external base image dependencies):

| Image                                          | Contents                                                                                        | Architectures                  | Typical size |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------ | ------------ |
| `ghcr.io/dhis2-chap/chapkit-py:latest`         | Python 3.13, chapkit, uv, `build-essential`, `pkg-config`                                       | `linux/amd64`, `linux/arm64`   | ~920 MB      |
| `ghcr.io/dhis2-chap/chapkit-r:latest`          | R 4.5 + `renv` + `pak` + common dev libs, Python 3.13, chapkit, uv                              | `linux/amd64`, `linux/arm64`   | ~1.2 GB      |
| `ghcr.io/dhis2-chap/chapkit-r-inla:latest`     | R 4.5 + INLA + spatial/time-series R stack (sf, spdep, dlnm, tsModel, ...), Python 3.13, chapkit | `linux/amd64` (INLA x86_64 only) | ~3-4 GB      |

Which one to pick:

- **Python MLproject?** Use `chapkit-py` — lean and multi-arch.
- **R MLproject that does not need INLA?** (e.g. `minimalist_example_r`) Use `chapkit-r`. Multi-arch, includes the R toolchain and `renv`/`pak` so you can install additional CRAN packages or restore a lockfile at runtime.
- **R MLproject that uses INLA?** (e.g. EWARS-style) Use `chapkit-r-inla`. INLA + fmesher + the chap-core R-model parity set (sf, spdep, sn, dlnm, tsModel, xgboost, ...) are pre-installed. amd64 only; you will need Rosetta emulation on Apple Silicon.

All three set `WORKDIR /work` and default to `CMD ["chapkit", "run", ".", "--host", "0.0.0.0", "--port", "8000"]`, so mounting your MLproject into `/work` is enough:

```bash
# Python model
docker run --rm -p 8000:8000 \
  -v "$(pwd):/work" \
  ghcr.io/dhis2-chap/chapkit-py:latest

# R model without INLA (multi-arch)
docker run --rm -p 8000:8000 \
  -v "$(pwd):/work" \
  ghcr.io/dhis2-chap/chapkit-r:latest

# R model with INLA (amd64-only; Rosetta on Apple Silicon)
docker run --rm -p 8000:8000 --platform=linux/amd64 \
  -v "$(pwd):/work" \
  ghcr.io/dhis2-chap/chapkit-r-inla:latest
```

### Model-Level Dependencies

The images contain chapkit, Python, and (for `chapkit-r`) R+INLA, but **not** your model's extra dependencies (pandas, scikit-learn, INLA extensions beyond the bundled set, etc.). Add them in one of two ways:

1. **Bake your own image** (production-recommended):

    ```dockerfile
    FROM ghcr.io/dhis2-chap/chapkit-py:latest
    WORKDIR /work
    COPY . .
    RUN uv pip install --python /app/.venv/bin/python -e .
    ```

2. **Install at container start** (quick dev):

    ```bash
    docker run --rm -p 8000:8000 -v "$(pwd):/work" \
      ghcr.io/dhis2-chap/chapkit-py:latest \
      bash -c "uv pip install --python /app/.venv/bin/python -e . && exec chapkit run . --host 0.0.0.0"
    ```

### Security

Both images currently run as `root`. Non-root hardening needs the usual volume-mapping dance (writable `/tmp`, per-user cache dirs, etc. — see `chap-core/compose.yml` for the reference pattern) and is a planned follow-up. The images are intended to sit in a trusted compose network behind chap-core.

---

## Integration with chap-core

`chapkit run` is designed to sit on a `docker compose` network alongside chap-core. Self-registration with chap-core is handled by servicekit's `SERVICEKIT_ORCHESTRATOR_URL` environment variable (the same mechanism used by `chap-core/compose.ewars.yml`):

```yaml
services:
  my-model:
    image: ghcr.io/dhis2-chap/chapkit-r:latest
    platform: linux/amd64
    volumes:
      - ./my_mlproject:/work
    environment:
      SERVICEKIT_ORCHESTRATOR_URL: http://chap:8000/v2/services/$$register
      # Optional shared secret, if chap has SERVICEKIT_REGISTRATION_KEY set:
      # SERVICEKIT_REGISTRATION_KEY: ${SERVICEKIT_REGISTRATION_KEY:-}
    depends_on:
      chap:
        condition: service_healthy
    networks:
      - chap
```

No chapkit-side configuration is needed — if `SERVICEKIT_ORCHESTRATOR_URL` is set, the service registers itself with chap-core on startup.

---

## Limitations and When to Prefer `chapkit init`

`chapkit run` is a thin runtime wrapper: it does not generate, edit, or version any files in your MLproject. That keeps it ideal for:

- Rapid evaluation of an existing MLproject under chapkit.
- Running a model in a docker-compose network alongside chap-core without a port to chapkit.
- Models whose train/predict logic is stable and already well-tested outside chapkit.

Use `chapkit init` instead when you want:

- Validation callbacks with custom diagnostics (`on_validate_train` / `on_validate_predict`).
- Python-typed config, business logic, or multi-artifact ML workflows beyond MLproject's entry-point model.
- A real chapkit project you can evolve (tests, migrations, additional endpoints).

A dedicated `chapkit convert` command for upgrading an MLproject repo into a full chapkit project (code generation, including per-model Dockerfiles) is on the roadmap.
