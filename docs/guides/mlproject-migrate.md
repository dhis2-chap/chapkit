# MLproject Migrate

`chapkit migrate` turns an existing MLflow-style `MLproject` directory into a first-class chapkit project in place. It's the code-generating sibling of [`chapkit run`](mlproject-runner.md): where `run` adapts an MLproject at runtime without writing anything, `migrate` gives you a `main.py`, a `Dockerfile` pointing at the right [chapkit-images](https://github.com/dhis2-chap/chapkit-images) base, a `pyproject.toml`, a `compose.yml`, and a `CHAPKIT.md` that you can commit, extend, and ship.

Your existing train/predict scripts and helpers are **not moved or modified**. Chaff (input CSVs, ad-hoc runner scripts, `MLproject` itself once we've parsed it, `.Rprofile`, etc.) is swept into `_old/` so nothing is lost.

## When to use which

| Command | Writes files | Good for |
| --- | --- | --- |
| [`chapkit run`](mlproject-runner.md) | No | Evaluating an MLproject against chapkit's API without changes to the repo. |
| `chapkit migrate` | Yes (at project root) | Turning an MLproject into a chapkit project you own, commit, and containerise. |
| [`chapkit init`](cli-scaffolding.md) | Yes (creates a new dir) | Greenfield chapkit project from scratch. |

## Quick start

```bash
cd /path/to/your/mlproject
chapkit migrate --dry-run      # preview the plan without touching anything
chapkit migrate                # execute with interactive prompts
chapkit migrate --yes          # execute non-interactively (scripts / CI)
```

After a successful run, the project root holds your unchanged source files alongside the generated chapkit wrapper, with the original structure preserved under `_old/`.

## What moves where

| Kind | Destination | Examples |
| --- | --- | --- |
| Original MLproject definition | `_old/` | `MLproject`, `MLproject.yaml` |
| Built-for-dev runners | `_old/` | `isolated_run.r`, `isolated_run.R`, `evaluate_one_step.py`, `slides.md`, `Makefile` |
| Example / demo data | `_old/` | `input/`, `output/`, `example_data/`, `example_data_monthly/`, root-level `training_data*.csv`, `predictions*.csv`, `*.pickle`, `*.rds`, `*.model` |
| R environment-local state | `_old/` | `renv/`, `.Rprofile` |
| Source code and lockfiles | kept at root | `train.r`, `predict.r`, `train.py`, `predict.py`, `lib.R`, `transformations.py`, Python packages, `renv.lock`, `uv.lock`, `LICENSE`, `README.md` |
| Git / venv / caches | ignored | `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` |

## Base image auto-detection

`chapkit migrate` picks the right base image from [chapkit-images](https://github.com/dhis2-chap/chapkit-images) by scanning your repo:

- **Python only** (`.py` at root) → `ghcr.io/dhis2-chap/chapkit-py:latest` (multi-arch, lean).
- **R without INLA** (`.r`/`.R` at root, no `library(INLA)` / `library(fmesher)`, no `docker_r_inla` in MLproject `docker_env`) → `ghcr.io/dhis2-chap/chapkit-r:latest` (multi-arch).
- **R with INLA** → `ghcr.io/dhis2-chap/chapkit-r-inla:latest` (`amd64` only).
- **Both R and Python** → `chapkit-r-inla` (fat image covers both); migrate prompts to confirm unless `--yes` is set.

Override the detection with `--base-image chapkit-{py,r,r-inla}`.

## Dynamic config becomes a typed class

Whatever you declared in `user_options` in the MLproject is emitted as typed fields on a `BaseConfig` subclass in the generated `main.py`. For an EWARS-style MLproject:

```yaml
user_options:
  n_lags:
    type: integer
    default: 3
  precision:
    type: number
    default: 0.01
```

...you get:

```python
class ewars_templateConfig(BaseConfig):
    """Configuration for ewars_template."""

    prediction_periods: int = 3
    n_lags: int = 3
    precision: float = 0.01
```

`prediction_periods: int = 3` is injected automatically if your MLproject doesn't declare it — chapkit's ML runner requires it.

## Interactive prompts

By default migrate asks for confirmation before executing the plan, asks for missing filename mappings if a placeholder like `{dataset}` isn't in the canonical set (`train_data`, `historic_data`, `future_data`, `out_file`, `model`, `model_config`, `polygons`), and asks before accepting an ambiguous base-image pick. `--yes` skips all prompts with the sensible defaults; `--dry-run` never prompts.

Use `--param NAME=FILENAME` (repeatable) to pre-answer the non-canonical-parameter prompt without going interactive.

## Your own `pyproject.toml`

If your repo already has a `pyproject.toml`, migrate renames it to `pyproject.original.toml` and writes a fresh minimal chapkit `pyproject.toml` alongside. Merge your dependencies (and any `[tool.*]` config) back into the new file manually — we don't try to guess the right merge.

## Next steps after migrate

```bash
uv sync && uv run chapkit run .                            # local dev server
docker build -t my-model . && docker run --rm -p 8000:8000 my-model   # containerised
uv run chapkit run .   &                                    # in background
uv run chapkit test                                         # drive an end-to-end train+predict
git add -A && git commit -m "chore: migrate MLproject to chapkit"
```

`CHAPKIT.md` (generated at root) has a short version of this for the migrated repo itself.

## What's deferred

- **`--restructure`** flag for the `scripts/` + top-level Python package layout manually applied by some existing chap-models repos. v1 leaves source files where they are.
- **`--with-monitoring`** / **`--with-validation`** mirroring the `chapkit init` flags.
- **`chapkit migrate --reverse`** to restore from `_old/` in one shot.
- **`git mv`** mode to preserve history for moved files.
