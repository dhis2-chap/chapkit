# Deploying to chap-core (and DHIS2)

This guide walks a chapkit model from `chapkit init` to a running container that registers itself with [chap-core](https://github.com/dhis2-chap/chap-core) and appears in the DHIS2 Modeling App. The reference implementation throughout is [`chapkit_ewars_model`](https://github.com/chap-models/chapkit_ewars_model) — an R/INLA forecaster deployed exactly this way.

## What you will build

```
chapkit init my-model
        |
        v   (edit main.py, scripts/, pyproject.toml)
 container image on GHCR
        |
        v   (compose overlay next to chap-core)
 self-registration to chap-core
        |
        v   (automatic)
 visible inside the DHIS2 Modeling App
```

The DHIS2 step is "free" — once your service is registered with chap-core, the Modeling App picks it up automatically. You do not run anything inside DHIS2.

## Prerequisites

- Docker (tested with the compose v2 plugin).
- A GitHub repository to host your model's source and publish images to GHCR.
- A chap-core deployment you can reach on a shared docker network. For local development, clone [`chap-core`](https://github.com/dhis2-chap/chap-core) and follow its README to stand one up.
- Optional: a DHIS2 instance with the [Modeling App](https://github.com/dhis2-chap/chap-app-modeling) installed, if you want to verify the end of the pipeline.

---

## Step 1 — Scaffold the project

```bash
uvx chapkit init my-model
cd my-model
```

See [CLI Scaffolding](cli-scaffolding.md) for template options (`ml`, `ml-shell`, `task`) and flags (`--with-monitoring`, `--with-validation`).

The scaffolded `main.py` already ends with `.with_registration()`. No manual edit is needed to enable registration — the builder silently no-ops when the orchestrator env var is unset, so `python main.py` still runs standalone.

## Step 2 — Confirm registration is wired

Open `main.py`. At the end of the builder chain you should see:

```python
app = (
    MLServiceBuilder(...)
    # Self-register with chap-core when SERVICEKIT_ORCHESTRATOR_URL is set.
    .with_registration()
    .build()
)
```

The two env vars the call responds to:

| Variable | Required | Purpose |
|---|---|---|
| `SERVICEKIT_ORCHESTRATOR_URL` | Triggers registration | The chap-core `$register` endpoint, e.g. `http://chap:8000/v2/services/$register`. When unset, `.with_registration()` is a no-op. |
| `SERVICEKIT_REGISTRATION_KEY` | Only if chap-core requires one | Shared secret validated by chap-core. Leave unset unless chap-core also sets it. |
| `SERVICEKIT_HOST`, `SERVICEKIT_PORT` | No | Override the hostname/port the service advertises to chap-core. Defaults to the container's own hostname on port 8000. |

Set these in the compose overlay in Step 6, not in code. Hard-coding URLs into `main.py` makes the image environment-specific.

## Step 3 — Declare capabilities via `MLServiceInfo`

`MLServiceInfo` is chap-core's contract for what your model expects. The DHIS2 Modeling App surfaces this metadata to the operator. The important fields:

| Field | Effect |
|---|---|
| `id` | Stable slug. Used in URLs and as the registration identity. Do not change after you ship. |
| `display_name` | Human-readable name shown in DHIS2. |
| `period_type` | `weekly` or `monthly`. Operators cannot feed mismatched data. |
| `required_covariates` | Column names your model needs on the input data. |
| `min_prediction_periods` / `max_prediction_periods` | Bounds on forecast horizon. |
| `model_metadata` | Author, contact email, organization, citation, `AssessedStatus`. |

The ewars model ([`main.py`](https://github.com/chap-models/chapkit_ewars_model/blob/main/main.py)) is a good concrete example — it declares `PeriodType.monthly`, requires `population`, allows additional continuous covariates, and pins `min/max_prediction_periods=0/100`.

## Step 4 — Build a Docker image

The scaffolded `Dockerfile` works out of the box for Python models — it is a multi-stage `uv`-based build that runs `uvicorn main:app` on port 8000 and exposes a `/health` healthcheck.

```bash
docker build -t my-model:dev .
docker run --rm -p 8000:8000 my-model:dev
```

**If your model has R or other system dependencies**, swap the base image for one of the pre-built [chapkit-images](https://github.com/dhis2-chap/chapkit-images):

- `ghcr.io/dhis2-chap/chapkit-py:latest` — Python (multi-arch).
- `ghcr.io/dhis2-chap/chapkit-r:latest` — R without INLA (multi-arch).
- `ghcr.io/dhis2-chap/chapkit-r-inla:latest` — R + INLA (amd64 only; add `--platform=linux/amd64` and expect emulation on Apple Silicon).

See [MLproject Runner → Running in a Container](mlproject-runner.md#running-in-a-container) for the full image table (sizes, architectures, contents) and [chapkit mlproject migrate → Base image auto-detection](mlproject-migrate.md#base-image-auto-detection) for how the right base is picked when adopting an existing MLproject.

## Step 5 — Publish to GHCR

chap-core pulls your image by tag from a container registry. GHCR is the path of least resistance: it needs no repo secrets and is already wired to your repo's `GITHUB_TOKEN`.

Drop this workflow into `.github/workflows/publish-docker.yml`:

```yaml
name: Publish Docker image

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

Every push to `main` publishes `ghcr.io/<owner>/<repo>:latest`. No secrets configuration is required — the built-in `GITHUB_TOKEN` is enough thanks to `permissions: packages: write`.

Want more — SHA tags for traceability, semver releases, build cache, SLSA attestations? See the ewars model's [`publish-docker.yml`](https://github.com/chap-models/chapkit_ewars_model/blob/main/.github/workflows/publish-docker.yml) for a fuller example to copy from.

## Step 6 — Wire it into chap-core with a compose overlay

chap-core ships with a base `compose.yml` (`chap`, `worker`, `redis`, `postgres`) and expects model services to be added via overlay files. Add one for your model, next to chap-core's `compose.yml`:

```yaml
# compose.my-model.yml
services:
  my-model:
    image: ghcr.io/<owner>/<repo>:latest
    platform: linux/amd64  # only if your base image needs it
    pull_policy: always
    ports:
      - "5010:8000"
    environment:
      SERVICEKIT_ORCHESTRATOR_URL: http://chap:8000/v2/services/$$register
      # Uncomment if chap has SERVICEKIT_REGISTRATION_KEY set:
      # SERVICEKIT_REGISTRATION_KEY: ${SERVICEKIT_REGISTRATION_KEY:-}
    depends_on:
      chap:
        condition: service_healthy
```

Launch the stack:

```bash
docker compose -f compose.yml -f compose.my-model.yml up -d
```

Two things worth calling out:

- **`$$register`** — the double dollar escapes `$` for compose's own variable substitution. If you write `$register`, compose will try to expand a variable named `register` and silently leave you with `http://chap:8000/v2/services/`, which 404s.
- **`depends_on: chap: condition: service_healthy`** — prevents the model from trying to register against a half-started chap-core. Chap-core's healthcheck must be green first.

The ewars model's [`compose.ewars.yml`](https://github.com/dhis2-chap/chap-core/blob/main/compose.ewars.yml) is the canonical example.

## Step 7 — Verify registration

Watch the model's logs for the registration line:

```bash
docker compose logs -f my-model
```

You should see the service register on startup and emit a keepalive ping roughly every 10s (adjustable via `.with_registration(keepalive_interval=...)` in code).

Ask chap-core what it knows about:

```bash
curl http://localhost:8000/v2/services
```

Your model's `id` (from `MLServiceInfo`) should appear in the list.

## Step 8 — Use it from DHIS2

Once your model is registered with chap-core, it shows up in the DHIS2 Modeling App automatically. No extra wiring on the DHIS2 side.

For the UI side of the flow — creating model templates, running predictions, pushing results back to DHIS2 — see chap-core's own docs:

- [`docs/modeling-app/managing-model-templates.md`](https://github.com/dhis2-chap/chap-core/blob/main/docs/modeling-app/managing-model-templates.md)
- [`docs/modeling-app/enabling-optional-model-services.md`](https://github.com/dhis2-chap/chap-core/blob/main/docs/modeling-app/enabling-optional-model-services.md)

---

## Troubleshooting

**Service never appears in `GET /v2/services`.**
Check the model's logs. Most common causes, in order: `SERVICEKIT_ORCHESTRATOR_URL` not set in the overlay; `$register` written instead of `$$register` in compose YAML (compose swallows the `$r`); model container cannot reach `chap` on the compose network (verify both services are on the same network, use the service name `chap` as the hostname).

**401 on registration.**
chap-core has `SERVICEKIT_REGISTRATION_KEY` set but your overlay does not. Set the same secret on both sides.

**Service registers but disappears.**
Keepalive pings are failing. Check the model container is still running (`docker compose ps`) and that nothing between it and chap-core is dropping long-lived HTTP connections.

**`platform: linux/amd64` warnings on Apple Silicon.**
Expected for amd64-only base images (R-INLA, some Python wheels). The container runs under emulation; slower but correct.

**SQLite file disappears between runs.**
The scaffolded image runs from `WORKDIR /workspace` and the default `DATABASE_URL` is the relative path `data/chapkit.db`, which resolves to `/workspace/data/chapkit.db`. The scaffolded `Dockerfile` pre-creates that directory with the right ownership, and the scaffolded `compose.yml` already mounts a named volume there — so persistence works out of the box when you run via `docker compose up`.

If you run the image directly with `docker run` and want the DB to survive container restarts, mount a volume at `/workspace/data`:

```yaml
services:
  my-model:
    volumes:
      - my-model-data:/workspace/data
volumes:
  my-model-data:
```

To put the DB somewhere else, set an absolute `DATABASE_URL` (note the four slashes):

```yaml
    environment:
      DATABASE_URL: sqlite+aiosqlite:////workspace/data/chapkit.db
```

---

## Appendix — reference files

- [`chapkit_ewars_model/main.py`](https://github.com/chap-models/chapkit_ewars_model/blob/main/main.py) — `MLServiceInfo`, `ShellModelRunner`, `.with_registration()`.
- [`chapkit_ewars_model/Dockerfile`](https://github.com/chap-models/chapkit_ewars_model/blob/main/Dockerfile) — short `FROM ghcr.io/dhis2-chap/chapkit-r-inla:latest` + `uv sync` layer; a concrete example of extending a chapkit-images base with model-specific Python deps.
- [`chapkit_ewars_model/.github/workflows/publish-docker.yml`](https://github.com/chap-models/chapkit_ewars_model/blob/main/.github/workflows/publish-docker.yml) — a fuller GHCR publish workflow with cache, semver tags, and SLSA attestations.
- [`chap-core/compose.ewars.yml`](https://github.com/dhis2-chap/chap-core/blob/main/compose.ewars.yml) — the overlay that drops the image onto the chap-core network and triggers self-registration.
- [`dhis2-chap/chapkit-images`](https://github.com/dhis2-chap/chapkit-images) — Dockerfiles and publish workflow for the `chapkit-py`, `chapkit-r`, and `chapkit-r-inla` base images referenced throughout this guide.
