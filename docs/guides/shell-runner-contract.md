# Shell-runner workspace contract

This page documents exactly what `ShellModelRunner` does for each train and predict invocation - which files chapkit writes, which files your script is expected to write, and how data flows from train into predict. It applies to projects scaffolded with `chapkit init --template shell-py` and `--template shell-r`, and to projects produced by `chapkit mlproject migrate`.

## At a glance

For every train or predict job, chapkit:

1. Creates a fresh temporary directory.
2. Copies your project files into it (your scripts, helpers, anything else committed in the repo).
3. Writes the request inputs (`config.yml`, `data.csv` / `historic.csv` + `future.csv`, optional `geo.json`).
4. Invokes your script with the workspace as the working directory.
5. Captures the entire workspace (your script's writes included) as an artifact.

Your script's only contract is: read the inputs chapkit wrote, write its outputs to known filenames in the same directory. Everything else is bookkeeping chapkit handles.

There is **no chapkit-managed model object** for shell runners. Your train script saves whatever model representation it likes (`model.rds`, `model.pickle`, an entire directory) and your predict script reads it back. Chapkit just preserves the workspace between train and predict so the file is still there.

## Train job lifecycle

### What chapkit writes (before your script runs)

- `data.csv` - training dataframe, posted to `POST /api/v1/ml/$train`.
- `config.yml` - the validated `Config` instance for this run (see [Config layout](#configyml-layout)).
- `geo.json` - GeoJSON, only if the request included `geo`.
- Everything from your project root - scripts, helper modules, committed assets - copied verbatim.

### What chapkit invokes

Your `train_command` from `main.py`, with placeholders substituted to relative paths:

```python
train_command = "Rscript scripts/train.R --data {data_file}"
# Becomes: Rscript scripts/train.R --data data.csv
```

Available placeholders:

| Placeholder | Substitutes to | When |
|---|---|---|
| `{data_file}` | `data.csv` | Training input |
| `{geo_file}` | `geo.json` (or empty string if no geo) | Optional |

The script runs with the temp dir as `cwd`, so relative paths Just Work.

### What your script must do

- Read `data.csv` (path passed as `--data`).
- Read `config.yml` (always at `./config.yml`, no flag needed).
- Read `geo.json` if it was passed (`--geo` will be empty if no geo).
- Write whatever model representation you need to the workspace - chapkit doesn't care about the format. Convention is `model.rds` (R) or `model.pickle` (Python).
- Exit 0 on success, non-zero on failure.

### What chapkit does after

The whole temp directory (your script's writes included) is zipped and persisted as an `ml_training_workspace` artifact. The artifact id is the handle that downstream `$predict` calls use.

## Predict job lifecycle

### What chapkit writes (before your script runs)

- The full contents of the train workspace artifact, restored - including whatever model file your train script wrote.
- `historic.csv` and `future.csv` - the dataframes posted to `POST /api/v1/ml/$predict`.
- `config.yml` - the same one from the train job, re-emitted for the predict workspace.
- `geo.json` - only if the request included `geo`.

### What chapkit invokes

Your `predict_command` from `main.py`:

```python
predict_command = (
    "Rscript scripts/predict.R "
    "--historic {historic_file} "
    "--future {future_file} "
    "--output {output_file}"
)
# Becomes: Rscript scripts/predict.R --historic historic.csv --future future.csv --output predictions.csv
```

Available placeholders:

| Placeholder | Substitutes to | When |
|---|---|---|
| `{historic_file}` | `historic.csv` | Always |
| `{future_file}` | `future.csv` | Always |
| `{output_file}` | `predictions.csv` | Always |
| `{geo_file}` | `geo.json` (or empty string if no geo) | Optional |

### What your script must do

- Read `historic.csv`, `future.csv`, and `config.yml`.
- Read whatever model file your train script wrote (`model.rds`, `model.pickle`, ...).
- Compute predictions.
- Write them to the path chapkit gave as `--output` (always `predictions.csv` in practice). The CSV format expected: index columns plus one or more `sample_0`, `sample_1`, ... columns. For deterministic models, just `sample_0`.
- Exit 0 on success, non-zero on failure.

### What chapkit does after

The output file is read back into the response, and the entire workspace is persisted as an `ml_prediction_workspace` artifact for debugging.

## `config.yml` layout

The shape of `config.yml` depends on which path scaffolded your project:

### `chapkit init --template shell-py` / `shell-r` (flat)

Every `Config` field is a top-level key:

```yaml
prediction_periods: 3
n_lags: 6
precision: 0.01
additional_continuous_covariates:
  - rainfall
  - mean_temperature
```

Read as `config$n_lags` in R, `config["n_lags"]` in Python.

### `chapkit mlproject migrate` (chap-core nested)

Reserved chap-core fields stay at the top, everything else nests under `user_option_values`:

```yaml
prediction_periods: 3
additional_continuous_covariates:
  - rainfall
  - mean_temperature
user_option_values:
  n_lags: 6
  precision: 0.01
```

Read as `config$user_option_values$n_lags`. This shape matches what existing chap-models scripts expect, which is why migrate uses it by default.

If you want to switch shapes, set `config_format="flat"` or `config_format="chap_core"` on `ShellModelRunner` in your `main.py`.

## Common pitfalls

**My predict script can't find `model.rds` (or `model.pickle`).**
The train workspace is restored into the predict workspace before your script runs. Make sure your train script actually wrote the file, and check the `ml_training_workspace` artifact for that job (`GET /api/v1/artifacts/{train_artifact_id}`) to confirm.

**`config.yml` keys don't match what my script expects.**
Two possibilities. Either you're on `chap_core` format and reading `config$n_lags` instead of `config$user_option_values$n_lags`, or your `Config` field uses a Python attribute name (`n_lags`) but chap-core wants the original hyphenated name (`n-lags`). For the second case, declare `n_lags: int = Field(alias="n-lags")` in the Config class - chapkit serialises with `by_alias=True` so the YAML key matches what your script expects.

**My script needs a helper module from `scripts/`.**
The whole project directory is copied to the workspace, so `source("scripts/lib.R")` and `from scripts.helpers import foo` both work. Paths are relative to the project root, not to your script's location.

**Stderr/stdout from a failing script.**
Both are captured into the workspace artifact's metadata. Pull the artifact and inspect `stdout` / `stderr`. From the running container: `docker compose logs` shows them in real time.

**The script ran but predictions don't show up in the response.**
Make sure you wrote to the path chapkit gave you (the `{output_file}` placeholder), which always resolves to `predictions.csv` relative to the workspace. Writing to an absolute path or a different filename means chapkit can't find the output.

## See also

- [R Quickstart](r-quickstart.md) for an end-to-end shell-r walkthrough.
- [ML Workflows](ml-workflows.md) for the full lifecycle including validation hooks.
- [MLproject Migrate](mlproject-migrate.md) for projects converted from MLproject.
