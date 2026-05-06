# MLproject Migrate

`chapkit mlproject migrate` turns an existing MLflow-style `MLproject` directory into a first-class chapkit project in place. It's the code-generating sibling of [`chapkit mlproject run`](mlproject-runner.md): where `run` adapts an MLproject at runtime without writing anything, `migrate` gives you a `main.py`, a `Dockerfile` pointing at the right [chapkit-images](https://github.com/dhis2-chap/chapkit-images) base, a `pyproject.toml`, a `compose.yml`, and a `CHAPKIT.md` that you can commit, extend, and ship.

Your existing train/predict scripts, helpers, and arbitrary config / data files are **not moved or modified**. Project metadata that chapkit regenerates (`README.md`, `.gitignore`, `pyproject.toml`, `Dockerfile`, `compose.yml`, `.python-version`, `Makefile`, the `MLproject` itself), plus obvious chaff (`input/`, `output/`, `example_data*/`, `isolated_run.*`, stale data CSVs, serialised model files, `renv/`), are swept into `_old/` so nothing is lost — your original repo state is fully recoverable from there.

!!! tip "Before and after migrate: the lifecycle checklist"
    The more your MLproject tells chapkit up front (`required_covariates`, `user_options`, pinned Python deps, service flags), the less hand-editing you need afterwards. The [Migration Checklist](mlproject-migration-checklist.md) walks through both the pre-migration prep and the post-migration smoke-test / iterate / ship steps — ten minutes there saves the "why did `chapkit test` fail" loop after.

## When to use which

| Command | Writes files | Good for |
| --- | --- | --- |
| [`chapkit mlproject run`](mlproject-runner.md) | No | Evaluating an MLproject against chapkit's API without changes to the repo. |
| `chapkit mlproject migrate` | Yes (at project root) | Turning an MLproject into a chapkit project you own, commit, and containerise. |
| [`chapkit init`](cli-scaffolding.md) | Yes (creates a new dir) | Greenfield chapkit project from scratch. |

## Quick start

```bash
cd /path/to/your/mlproject
chapkit mlproject migrate --dry-run      # preview the plan without touching anything
chapkit mlproject migrate                # execute with interactive prompts
chapkit mlproject migrate --yes          # execute non-interactively (scripts / CI)
```

No local chapkit install? Run it via the `-cli` container (chapkit pre-installed):

```bash
docker run --rm -v "$(pwd):/work" \
  ghcr.io/dhis2-chap/chapkit-py-cli:latest \
  chapkit mlproject migrate . --yes
```

After a successful run, the project root holds your unchanged source files alongside the generated chapkit wrapper, with the original structure preserved under `_old/`.

## What moves where

| Kind | Destination | Examples |
| --- | --- | --- |
| Original MLproject definition | `_old/` | `MLproject`, `MLproject.yaml` |
| User's project metadata (chapkit regenerates these) | `_old/` | `README.md`, `.gitignore`, `.dockerignore`, `.python-version`, `pyproject.toml`, `Makefile`, `Dockerfile`, `compose.yml`, `CHAPKIT.md` |
| Built-for-dev runners | `_old/` | `isolated_run.r`, `isolated_run.R`, `evaluate_one_step.py`, `slides.md` |
| Example / demo data | `_old/` | `input/`, `output/`, `example_data/`, `example_data_monthly/`, root-level `training_data*.csv`, `predictions*.csv`, `future_data*.csv`, `historic_data*.csv`, `*.pickle`, `*.rds`, `*.model` |
| R environment-local state | `_old/` | `renv/`, `.Rprofile` |
| Source code, user data, lockfiles, LICENSE | kept at root | `train.r`, `predict.r`, `train.py`, `predict.py`, `lib.R`, `transformations.py`, Python packages, arbitrary `*.yaml` / `*.json` / `*.toml` config your scripts read, `renv.lock`, `uv.lock`, `LICENSE`, `.github/` |
| Git / venv / caches | ignored | `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` |

Arbitrary files at the project root that aren't in the chaff list stay put — scripts that read `schema.yaml`, `model_params.json`, etc. at startup continue to work. The user's original `pyproject.toml` lands at `_old/pyproject.toml`; merge its dependencies into the generated chapkit `pyproject.toml` manually.

## Base image auto-detection

`chapkit mlproject migrate` picks the right base image from [chapkit-images](https://github.com/dhis2-chap/chapkit-images) by scanning your repo:

- **Python only** (`.py` at root) → `ghcr.io/dhis2-chap/chapkit-py:latest` (multi-arch, lean).
- **R + INLA** (`library(INLA)` / `library(fmesher)` in any root R script, or `docker_r_inla` in MLproject `docker_env`) → `ghcr.io/dhis2-chap/chapkit-r-inla:latest` (`amd64` only).
- **R + tidyverse** (any of `library(tidyverse)`, `dplyr`, `ggplot2`, `tidyr`, `readr`, `purrr`, `tibble`, `fable`, `tsibble`, `feasts`, `forecast`, `ranger`, `xgboost`, `glmnet`, `lubridate`, `janitor` and **no** INLA) → `ghcr.io/dhis2-chap/chapkit-r-tidyverse:latest` (multi-arch).
- **R, no INLA or tidyverse hints** → `ghcr.io/dhis2-chap/chapkit-r:latest` (multi-arch, minimal).
- **Both R and Python** → `chapkit-r-inla` (fat image covers both); migrate prompts to confirm unless `--yes` is set.

Override the detection with `--base-image {chapkit-py,chapkit-r,chapkit-r-tidyverse,chapkit-r-inla}`.

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

If your repo already has a `pyproject.toml`, migrate:

1. Parses `[project.dependencies]` from it (skipping any `chapkit` entry — we add our own).
2. Writes a fresh chapkit `pyproject.toml` at the project root that includes **both** `chapkit` and the deps pulled from your original.
3. Moves your original `pyproject.toml` to `_old/pyproject.toml` so it's still available for reference (e.g. if you had optional-dep groups, `[tool.*]` config, or environment markers you want to port over manually).

The migrate output summary tells you how many deps were merged, and points at the preserved original file.

## Next steps after migrate

```bash
uv sync && uv run python main.py                            # local dev server
docker build -t my-model . && docker run --rm -p 9090:8000 my-model   # containerised
uv run python main.py   &                                   # in background
uv run chapkit test                                         # drive an end-to-end train+predict
git add -A && git commit -m "chore: migrate MLproject to chapkit"
```

`CHAPKIT.md` (generated at root) has a short version of this for the migrated repo itself.

## What's deferred

- **`--restructure`** flag for the `scripts/` + top-level Python package layout manually applied by some existing chap-models repos. v1 leaves source files where they are.
- **`--with-validation`** mirroring the `chapkit init` flag.
- **`chapkit mlproject migrate --reverse`** to restore from `_old/` in one shot.
- **`git mv`** mode to preserve history for moved files.
