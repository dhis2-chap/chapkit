# R Quickstart

A 10-minute path from "I have an R model" to "it's running as a chapkit service that chap-core can talk to." This guide assumes you know R and shell basics; you don't need to know Python.

## What you're building

`chapkit init --template shell-r` scaffolds a small project with this shape:

```
my-model/
├── scripts/
│   ├── train.R          # your training logic
│   └── predict.R        # your prediction logic
├── main.py              # service config (you change ~10 lines here)
├── pyproject.toml       # Python deps for the service layer
├── Dockerfile           # builds on ghcr.io/dhis2-chap/chapkit-r-inla
├── compose.yml          # docker compose stack
└── README.md
```

When you `docker compose up`, the container ships R 4.5 + INLA + the chap ecosystem R stack and exposes an HTTP service that chap-core can call. Your `train.R` and `predict.R` are invoked as subprocesses; you read CSV inputs and write CSV outputs the way you would on the command line.

## What you need installed

- **Docker** with the compose plugin. Verify with `docker --version` and `docker compose version`.
- **Python 3.13** and **`uv`**. You don't need to know Python, but the CLI is shipped via Python's package manager. Install uv from [astral.sh/uv](https://docs.astral.sh/uv/) (one curl command).

You do *not* need a local R install. The Docker image has everything.

## Step 1 - Scaffold the project

```bash
uvx chapkit init my-model --template shell-r
cd my-model
```

`uvx` runs chapkit straight from PyPI without "installing" it system-wide. Use `uv tool install chapkit` if you'd rather install once.

## Step 2 - Generate the lockfile

This creates `uv.lock`, which the Dockerfile pins against. Once per project (and after you change Python deps):

```bash
uv lock
```

## Step 3 - Build and run

```bash
docker compose up --build
```

First build downloads `chapkit-r-inla` (~570 MB) and compiles your image; subsequent builds are fast. When you see `Application startup complete`, the service is up.

Open <http://localhost:9090/docs> for the interactive API browser.

## Step 4 - Verify with `chapkit test`

In a second terminal, from inside `my-model/`:

```bash
uvx chapkit test --verbose
```

This posts synthetic data through the live service: creates a config, runs a training job, runs a prediction. If everything passes, the plumbing works. (It doesn't tell you whether your model is *good* - just that the pipeline is intact.)

## Step 5 - Replace the toy logic with your model

Open `scripts/train.R`. The default writes a "model" that's just the column means of the input data. Replace the `numeric_data <- ...` / `model <- list(...)` / `saveRDS` block with your actual training code. The contract is:

- You receive `--data <path>` (CSV training data) and optionally `--geo <path>` (GeoJSON).
- You read `config.yml` from the working directory (it's written by chapkit before your script runs; values come from the `Config` class in `main.py`).
- You write your model wherever you want; convention is `model.rds`.

Then open `scripts/predict.R`:

- You receive `--historic`, `--future`, `--output` (and optional `--geo`).
- You load whatever you wrote in `train.R`.
- You write predictions to the `--output` CSV path.

Rebuild and re-test:

```bash
docker compose up --build -d

# Wait for the container to be healthy before running tests. Works in
# bash, zsh, and sh (the `!` plus `while` form is POSIX). On Apple
# Silicon under Rosetta the first request can take 20-30s.
while ! curl -fsS http://localhost:9090/health >/dev/null 2>&1; do sleep 2; done

uvx chapkit test --verbose
```

## Step 6 - Customise the model config

Open `main.py` and find the `Config` class. The Python here is straightforward - one line per field, with a default value:

```python
class MyModelConfig(BaseConfig):
    """Configuration for my-model."""

    # Required: number of prediction periods
    prediction_periods: int = 3

    # Add fields your scripts read from config.yml:
    n_lags: int = 3
    precision: float = 0.01
```

These fields land in `config.yml` (in the script's working directory) at runtime, so your R scripts read them via:

```r
config <- yaml::yaml.load_file("config.yml")
n_lags <- config$user_option_values$n_lags
```

While you're in `main.py`, fill in the `MLServiceInfo` block: your name, email, model description, period type. That metadata is what chap-core shows users.

## Step 7 - Add R packages your scripts need

The default base image (`chapkit-r-inla`) already ships INLA, fmesher, dlnm, tsModel, sn, xgboost, sf, spdep, dplyr, readr, yaml, jsonlite, pak, renv. If you need more packages, add them in the `Dockerfile`:

```dockerfile
RUN R -e "install.packages('your_package', repos = 'https://cloud.r-project.org')"
```

Or commit an `renv.lock` and have the Dockerfile restore it - see the [chapkit_ewars_model repo](https://github.com/chap-models/chapkit_ewars_model) for a working example.

## Step 8 - Deploy alongside chap-core

When your model works locally, see [Deploying to chap-core](deploying-to-chap-core.md) for the compose-overlay pattern that drops your container next to chap-core's stack and registers it automatically.

## Troubleshooting

**`docker compose up` fails with "no match for platform"** - the chapkit-r-inla base image is amd64-only. The scaffolded Dockerfile already pins `--platform=linux/amd64` for Apple Silicon (Rosetta), but if you removed that line you'll see this error. Add it back.

**`uv lock` fails with "No solution found"** - the Python dep `chapkit>=X.Y.Z` in `pyproject.toml` may be ahead of what's published to PyPI. Edit `pyproject.toml` to use a published version (e.g. the latest tag at <https://github.com/dhis2-chap/chapkit/tags>).

**`chapkit test` says "service is not healthy"** - the container hasn't finished starting. Wait 10-20s and retry, or check `docker compose logs` for errors.

**Train job fails inside the container** - run `docker compose logs --tail=50` to see your script's stderr. Common culprits: missing R package (add via Dockerfile), unexpected CSV columns (check what chap-core sends with `chapkit test --verbose`), or a typo in `train_command` in `main.py`.

## Where to go next

- [Shell-runner contract](shell-runner-contract.md) for the precise file-by-file lifecycle of train and predict workspaces - useful when scripts misbehave.
- [MLproject Migrate](mlproject-migrate.md) if you have an existing MLproject directory you'd rather adopt than start from scratch.
- [ML Workflows](ml-workflows.md) for the full lifecycle (validation hooks, multi-stage pipelines, custom runners).
- [Deploying to chap-core](deploying-to-chap-core.md) for the production registration flow.
