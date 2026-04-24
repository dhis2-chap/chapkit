# MLproject Migration Checklist

This guide is the full lifecycle checklist for migrating an existing [MLflow-style `MLproject`](https://mlflow.org/docs/latest/projects.html) directory into a chapkit service. Use it to prepare your repo **before** you run [`chapkit migrate`](mlproject-migrate.md), and to iterate **after** when the first `chapkit test` run surfaces something.

!!! info "Scope: shell-based runners"
    `chapkit migrate` always emits a [`ShellModelRunner`](ml-workflows.md#shellmodelrunner) in the generated `main.py`. Your train/predict scripts run as shell commands in a copied workspace, with input/output bound to well-known filenames. That's the model this guide addresses. If you need a Python-native runner ([`FunctionalModelRunner`](ml-workflows.md#functionalmodelrunner) or a custom [`BaseModelRunner`](ml-workflows.md#basemodelrunner)), start from [`chapkit init`](cli-scaffolding.md) instead and wire your functions directly — migrate doesn't produce those runners.

## Before migration

The more your MLproject tells chapkit up front, the less hand-editing you need after. Work through these items and commit the result before you run migrate.

### 1. Declare every column your scripts hardcode

If your `train.py` / `train.R` reaches for a column by name, declare it as a [`required_covariate`](ml-workflows.md#mlserviceinfo-fields):

```yaml
required_covariates:
  - rainfall
  - mean_temperature
  - mean_relative_humidity
```

This flows into `MLServiceInfo.required_covariates` in the generated `main.py`, and [`chapkit test`](testing-ml-services.md)'s synthetic data generator emits those columns automatically so the first smoke test finds them.

Skip this and `chapkit test` generates data with only the canonical columns (`time_period`, `location`, `disease_cases`, `population`, `rainfall`, `mean_temperature`). Your script reaches for a column that isn't there and crashes with `KeyError: "['<col>'] not in index"`.

**Rule of thumb:** if removing a column would break the model, it's a `required_covariate`.

### 2. Declare every option your scripts read from `config.yml`

For each knob your scripts access as `config["user_option_values"]["<name>"]` (Python) or `config$user_option_values$<name>` (R), declare it under [`user_options`](mlproject-migrate.md#dynamic-config-becomes-a-typed-class):

```yaml
user_options:
  alpha:
    type: number
    default: 0.5
    description: Weighting parameter for the forecast blend.
  n_lags:
    type: integer
    default: 3
    description: Number of lags to include.
```

Types: `integer`, `number` (float), `string`, `boolean`, `path`. A missing `default` marks the field required — the orchestrator must supply it per config POST.

Migrate emits each as a typed field on the generated `<Name>Config` class; `chapkit test` generates a config instance and POSTs it through. Skip the declaration and your script crashes with `KeyError: '<name>'` the first time it reads from config.

### 3. Pin your Python dependencies

Bare names like `- pandas` in `pyenv.yaml` resolve to whatever's latest on PyPI at `docker build` time. That's fine until a major bump removes an API your scripts rely on — pandas 2.1 dropped `fillna(method=...)` and silently broke models written against 2.0.

Pin everything that matters:

```yaml
# pyenv.yaml
dependencies:
  - pandas>=2.0,<2.1       # pin before 2.1's fillna(method=) removal
  - statsmodels>=0.14
  - joblib>=1.3
```

`conda.yaml`, `environment.yaml`, or the `[project.dependencies]` section of `pyproject.toml` work the same way. `chapkit migrate` carries entries verbatim and warns in its summary about any unpinned name it finds, but pinning upstream keeps the MLproject self-contained.

### 4. Tell migrate about your Python environment

If your MLproject declares one of these, migrate picks it up automatically:

```yaml
python_env: pyenv.yaml        # MLflow-style python_env
conda_env: environment.yaml   # conda-style
uv_env: pyproject.toml         # uv-native
renv_env: renv.lock            # R, reproducible library
```

Docker-only MLprojects that ship a `pyenv.yaml` alongside without declaring `python_env:` also work — migrate probes the project root for `pyenv.yaml`, `conda.yaml`, or `environment.yaml` as a fallback — but the explicit declaration is clearer.

### 5. Use canonical parameter names in `entry_points`

Migrate translates MLproject's `{param}` placeholders into chapkit workspace filenames. The canonical set that Just Works:

| MLproject param | Substituted with | Where chapkit writes / expects it |
| --- | --- | --- |
| `{train_data}` | `{data_file}` | Panel-data CSV at workspace root, training |
| `{historic_data}` | `{historic_file}` | Historic CSV, prediction input |
| `{future_data}` | `{future_file}` | Future CSV (`disease_cases` nulled), prediction input |
| `{out_file}` | `{output_file}` | Your predict script writes predictions here |
| `{model}` | `model` | Arbitrary path your scripts save/load |
| `{model_config}` | `config.yml` | Written by chapkit from the typed Config |
| `{polygons}` | `{geo_file}` | GeoJSON (only when `requires_geo: true`) |

Non-canonical names are supported — migrate stops and asks for the filename, or `--param NAME=FILENAME` pre-answers — but canonical is less friction.

### 6. Fill in `meta_data` (recommended)

The chap-core orchestrator UI uses this to describe your model to end-users:

```yaml
meta_data:
  display_name: EWARS for Dengue
  description: Epidemic early-warning model for dengue using weather lags.
  author: CHAP team
  author_note: Calibrated for Rwanda; other contexts need re-calibration.
  author_assessed_status: yellow    # green|yellow|orange|red|gray
  contact_email: team@example.org
  organization: HISP Centre, University of Oslo
  organization_logo_url: https://example.org/logo.png
  citation_info: Author et al. (2024), Model description.
  repository_url: https://github.com/your-org/your-model
  documentation_url: https://your-org.github.io/your-model
```

The traffic-light `author_assessed_status`:

| Status | Meaning |
| --- | --- |
| `green` | Validated, recommended for production use |
| `yellow` | Works, but results should be reviewed |
| `orange` | Experimental; correctness not guaranteed |
| `red` | Known issues; don't use in production |
| `gray` | Not yet assessed |

Migrate defaults to `yellow` if you don't set one — adjust up or down to match reality.

### 7. Declare service-level flags if they apply

```yaml
supported_period_type: monthly                      # or: weekly
requires_geo: true                                  # if scripts read geo.json
allow_free_additional_continuous_covariates: true   # let orchestrators POST extra climate covariates
target: disease_cases                               # the column being predicted
```

All four land in `MLServiceInfo` in the generated `main.py`.

### 8. Keep your scripts workspace-root-relative

`ShellModelRunner` copies your entire project directory into a temp workspace per train/predict call. Your script runs with `cwd=<workspace>`, and chapkit writes:

- `data.csv` (training), or `historic.csv` + `future.csv` (prediction)
- `config.yml`
- `geo.json` (only when `requires_geo` is set)

Your script loads those by **relative path** (`"config.yml"`, `"data.csv"`). Don't hardcode `/absolute/paths/`; don't assume the script is running from a particular check-out. Helper modules, lookup tables, and arbitrary data files your script `read("schema.yaml")`s at startup continue to work — the whole repo is copied in (minus `.git`, `.venv`, `__pycache__`, build artefacts).

### 9. Confirm your train/predict contract

- **`train`** must leave a model file at whatever path is bound to `{model}` (commonly `"model"`). The entire workspace is zipped as an [artifact](artifact-storage.md) afterwards, so anything else you write (`metrics.json`, training plots, …) is preserved automatically.
- **`predict`** must write predictions to the path bound to `{out_file}` (commonly `"predictions.csv"`), with at minimum `time_period`, `location`, and one or more prediction columns. Chapkit reads that file to return predictions over HTTP.

### 10. Dry-run locally

Before migrating, confirm your scripts run end-to-end with real input:

```bash
python train.py example_data/training_data.csv model.pickle config.yml
python predict.py model.pickle example_data/historic.csv example_data/future.csv predictions.csv config.yml
```

If that works, your scripts are ready for migrate.

### The 30-second pre-flight

- [ ] Hardcoded columns → `required_covariates`.
- [ ] `config["user_option_values"][...]` keys → `user_options` with types and defaults.
- [ ] All Python deps pinned.
- [ ] `python_env:` / `conda_env:` / `uv_env:` / `renv_env:` set.
- [ ] Entry-point `{params}` use canonical names.
- [ ] `meta_data:` filled in (especially `display_name`, `description`, `author_assessed_status`).
- [ ] Service flags set where applicable (`supported_period_type`, `requires_geo`, `allow_free_additional_continuous_covariates`).
- [ ] Scripts use relative paths in the workspace.
- [ ] Local end-to-end dry-run clean.

## Run migrate

```bash
cd /path/to/your/mlproject
chapkit migrate --dry-run      # preview the plan without touching anything
chapkit migrate                # execute with interactive prompts
chapkit migrate --yes          # execute non-interactively (CI / scripts)
```

Full flag reference and behavioural details in the [MLproject Migrate guide](mlproject-migrate.md).

## After migration

Migrate succeeded — now confirm the wiring is sound and polish the generated project.

### 11. Read the migrate summary

The last thing migrate prints is a summary with:

- how many items moved to `_old/`
- how many files were generated
- how many user deps got merged into `pyproject.toml`
- **a warning about any unpinned deps** (you should see zero if step 3 was done)

If the unpinned-deps note is non-empty, decide whether to pin them at the source (`pyenv.yaml` / `pyproject.toml` — re-run migrate) or in the generated `pyproject.toml` (edit in place).

### 12. Smoke-test with `chapkit test`

**Python models — one-shot:**

```bash
make test                          # uv run chapkit test --start-service
```

`--start-service` spawns `python main.py` in the local uv venv, runs the smoke test, tears the service down. Only works for Python-based models — see the R caveat below.

**Python models — two-terminal (when you want to inspect the live service):**

```bash
# terminal 1:
make run

# terminal 2 (from the same project dir):
make test-remote
```

!!! warning "R models: Docker only"
    `chapkit test --start-service` (and the `make test` / `make run` paths underneath) spawn `main.py` in the **host's Python environment** via `subprocess.Popen([sys.executable, "main.py"], ...)`. For an R model, `main.py` then shells out to `Rscript train.R` — which fails unless R and every package in `renv.lock` are installed on the host. That is essentially never what a developer has locally.

    For R models, use Docker instead:

    ```bash
    # terminal 1:
    make docker-build
    make docker-run

    # terminal 2 (once the container reports healthy):
    make test-remote
    ```

    The generated Docker image (built on [`chapkit-r`](https://github.com/dhis2-chap/chapkit-images) or [`chapkit-r-inla`](https://github.com/dhis2-chap/chapkit-images)) ships R and pre-restores `renv.lock` so your scripts run in the environment they were written for.

A clean run looks like:

```
Configs created:       1
Trainings completed:   1
Trainings failed:      0
Predictions completed: 1
Predictions failed:    0
Result: ALL TESTS PASSED
```

If something failed, the **service logs** hold the actual stderr — the terminal running `make run` for local runs, or `docker logs <container>` for Docker. See [Iterating on `chapkit test` failures](#14-iterating-on-chapkit-test-failures) below.

### 13. Build and run in Docker

```bash
docker build -t my-model .
docker run --rm -p 8000:8000 my-model
# elsewhere:
chapkit test --url http://localhost:8000
```

If the image starts but `chapkit test` hangs or returns errors, look at `docker logs` for the service — it'll have the structured log output with `train_script_failed` / `predict_script_failed` events and their `stderr`.

### 14. Iterating on `chapkit test` failures

The three failure shapes we've seen most often on real `chap-models` repos:

| Symptom | Fix |
| --- | --- |
| `KeyError: "['<col>'] not in index"` at train time | Missed a hardcoded column. Add it to `required_covariates:` in the (now in `_old/`) MLproject and re-run migrate, or edit `MLServiceInfo.required_covariates=[...]` directly in the generated `main.py`. |
| `KeyError: '<name>'` reading from config.yml | Missed a `user_options` entry. Re-run migrate with the MLproject updated, or add the typed field to the `<Name>Config` class in `main.py` directly. |
| `TypeError: <func>() got an unexpected keyword argument '<kw>'` (or similar deprecation errors) | Unpinned dep resolved to a newer major version than the script was written against. Either pin the dep in `pyproject.toml` (e.g. `"pandas>=2.0,<2.1"`) and rebuild, or update the script to the new API. |

Re-running migrate is always safe if `_old/` is clean — the classifier is deterministic and re-pulls your MLproject and scripts. Tiny `main.py` tweaks are usually faster than a full re-migrate once the project is otherwise good.

### 15. Bump `author_assessed_status` once you're confident

Migrate defaults to `yellow`. After your smoke tests pass and you've verified against real data, bump to `green`. If issues are known, drop to `orange` or `red` — it's self-assessed and visible to everyone browsing chap-core.

```python
# main.py
model_metadata=ModelMetadata(
    ...,
    author_assessed_status=AssessedStatus.green,
),
```

### 16. Commit, but decide what to do with `_old/`

`_old/` holds your original `MLproject`, pre-chapkit project metadata, and example data. Two reasonable choices:

- **Keep it in git** (default) — useful while you iterate; you can re-run migrate or port config by hand. A bit of repo bloat.
- **`.gitignore` it** — tell git to forget it once the migration is stable. Makes the repo cleaner. The generated `.gitignore` does **not** ignore `_old/` by default; add a line if you want it out.

Either way:

```bash
git add -A
git commit -m "chore: migrate MLproject to chapkit"
```

### 17. Register with chap-core

The generated `main.py` calls `.with_registration()`, so when the service starts with `SERVICEKIT_ORCHESTRATOR_URL` set, it auto-registers with chap-core and sends keepalive pings. Without the env var it silently no-ops.

Typical compose-based setup (runs your model alongside a chap-core instance):

```yaml
services:
  my-model:
    build: .
    environment:
      SERVICEKIT_ORCHESTRATOR_URL: http://chap:8000/v2/services/$$register
      # SERVICEKIT_REGISTRATION_KEY: ${SERVICEKIT_REGISTRATION_KEY:-}   # optional shared secret
    depends_on:
      chap:
        condition: service_healthy
```

### 18. Publish the image

Push to a registry you control (typically GitHub Container Registry):

```bash
docker tag my-model ghcr.io/your-org/my-model:latest
docker push ghcr.io/your-org/my-model:latest
```

Point the `image:` in your production compose file at the tag, and you're live.

### 19. Retire `_old/` once you're done

Once the migrated project has been running cleanly for a release cycle and nothing in `_old/` is relevant anymore:

```bash
git rm -r _old/
git commit -m "chore: drop _old/ now that migration is stable"
```

If a future teammate asks what used to live there, the commit history has the answer.

## See also

- [MLproject Migrate](mlproject-migrate.md) — full `chapkit migrate` reference: flags, classification rules, what's generated.
- [`chapkit test`](testing-ml-services.md) — smoke-test harness used in step 12.
- [ML Workflows](ml-workflows.md) — runner types, `MLServiceInfo`, `ModelMetadata` fields.
- [Configuration Management](configuration-management.md) — `BaseConfig`, typed schemas, HTTP lifecycle.
- [Artifact Storage](artifact-storage.md) — how training workspaces are persisted.
- [chapkit-images](https://github.com/dhis2-chap/chapkit-images) — base Docker images the generated Dockerfile builds on.
