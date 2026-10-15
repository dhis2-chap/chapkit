# MLproject Export

`chapkit mlproject export` is the reverse of [`chapkit mlproject migrate`](mlproject-migrate.md): it reads a chapkit service module (usually `main.py`) and writes the `MLproject` that [`chapkit mlproject run`](mlproject-runner.md) serves as the same service. Use it when a model ships a hand-written chapkit service but should also run from its MLproject, for example with `chapkit mlproject run .` or `chap eval --run-config.as-chapkit`.

`main.py` stays the source of truth. The MLproject is generated from it, and `--check` keeps the two from drifting.

## Quick Start

```bash
chapkit mlproject export                 # reads ./main.py, writes ./MLproject
chapkit mlproject export path/to/main.py
chapkit mlproject export --check         # write nothing; exit 1 when MLproject is stale
```

Export **runs the module**: it executes `main.py` the way the service itself starts, from its own directory, so imports and relative paths work, and so anything `main.py` does at import time happens too, such as creating a `data/` directory or building the whole app. Output it prints while doing so goes to stderr.

Add the check to CI or `make lint` so a change to `main.py` without regenerating the MLproject fails the build:

```makefile
lint:
	uv run chapkit mlproject export --check
```

## What It Reads

The command finds the objects by type, so their names do not matter:

| In the module | Becomes in the MLproject |
| --- | --- |
| The `MLServiceInfo` | `name`, `version`, `target`, `supported_period_type`, `required_covariates`, `allow_free_additional_continuous_covariates`, `requires_geo`, `min_prediction_periods`, `max_prediction_periods`, `hpo_search_space`, and `meta_data` (display name, description, author fields, URLs) |
| The `BaseConfig` subclass | `user_options`: one JSON Schema property per field, with its type, constraints, default and description |
| The `ShellModelRunner` | `entry_points.train` / `predict`, with runner placeholders turned back into MLproject parameters (`{data_file}` becomes `{train_data}`, `{output_file}` becomes `{out_file}`, and so on), plus `adapters` |
| The `ArtifactHierarchy` | `name`, when it slugifies to the service id; otherwise the id is used |

Only `ShellModelRunner` services can be exported, because an MLproject runs train and predict as shell commands. If the module defines several config classes, pick one with `--config NAME`.

`prediction_periods` and `additional_continuous_covariates`, which every chapkit config has, are left out of `user_options` unless the config changes their default or what they accept (for example `prediction_periods: int = Field(default=3, ge=2, le=6)`).

Literal braces in a runner command (`{{` / `}}`) are escaped once more in the MLproject, because `chapkit mlproject run` formats the command twice: first the MLproject parameters, then the runner's own placeholders.

### Config Layout

`ShellModelRunner` writes a flat `config.yml` by default, while `chapkit mlproject run` writes chap-core's nested layout unless told otherwise. When the runner uses the flat layout, export writes it into the MLproject, where `chapkit mlproject run` picks it up however it is started (see [Config layout](mlproject-runner.md#config-layout)):

```yaml
meta_data:
  config_format: flat
```

### Not Carried Over

Some service settings have no place in an MLproject, so `chapkit mlproject run` does not apply them. Export and `--check` print a warning for each one it finds, without failing:

- the runner's `on_validate_train` / `on_validate_predict` hooks: data the service rejects is accepted by the MLproject service;
- artifact `level_labels` other than `ml_training_workspace` / `ml_prediction`;
- API key authentication (`with_auth`);
- a keepalive interval set in `with_registration` other than servicekit's default (pass it to `chapkit mlproject run --keepalive-interval`);
- routes the MLproject service does not serve, such as extra endpoints added to the app.

Custom health checks cannot be detected from the built app and are not warned about.

## The Check

`--check` fails in two cases:

1. The MLproject differs from what `export` would write now. The diff is printed.
2. The MLproject matches, but `chapkit mlproject run` would read it as a different service. The file is parsed with chapkit's own MLproject reader, and the served `/api/v1/info`, the config options with their defaults and the values each accepts, the runner commands and the adapters are compared with the module. This catches a generator bug that a matching text diff would hide, and contract differences that come from how `run` reads an MLproject. For example, a service without `author_note` fails, because `run` serves the description in its place.

## Safety

The generated file starts with a `Generated from ... by chapkit mlproject export` header. `export` refuses to overwrite an MLproject without it, so a hand-written file is not lost. Pass `--force` to replace it once; from then on it is regenerated freely.

## Options

| Option | Effect |
| --- | --- |
| `PATH` | Service module to read (default: `main.py`). It is executed, import-time side effects included. |
| `-o`, `--output FILE` | MLproject to write (default: the `MLproject`, `MLproject.yaml` or `MLproject.yml` next to the module, else a new `MLproject`). |
| `--check` | Write nothing; exit 1 when the MLproject is stale or would be read as a different service. |
| `--config NAME` | Config class to use when the module defines several. |
| `--force` | Overwrite an MLproject that was not generated by this command. |
