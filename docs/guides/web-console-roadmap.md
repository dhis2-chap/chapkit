# Web Console Roadmap

Planned follow-ups for the [Web Console](web-console.md), captured so they are not
lost.

Several items have since shipped and are no longer listed here: Jobs
deep-linking, apply-from-dry-run, resizable master/detail and sidebar panels,
Jobs multi-select + grouping, the schema-driven config form, contiguous
multi-year sample data, the self-describing DataFrame schema, the GeoJSON
bounding box, the Map view (a MapLibre choropleth over an OpenFreeMap basemap,
animated over time — currently disabled in the UI), and the live Monitoring
screen (in-memory polling of the Prometheus `/metrics` endpoint).

## Console / UX

- **Job type / label in the Jobs list.** Show what each job *is* (train, predict,
  etc.) in the table and detail. Blocked on the framework: servicekit's
  `JobRecord` only carries id/status/timestamps/error and `scheduler.add_job`
  takes no name/label, so the operation type isn't represented anywhere. Needs a
  servicekit enhancement (a `name`/`label`/`kind` on jobs) that chapkit's ML
  manager would set when scheduling `$train` / `$predict`.

## Data & models

- **Backtest / evaluation with n-fold splits.** Rolling-origin / n-split
  backtesting and evaluation, surfaced in the console.
- **safetensors model serialization.** Move model artifacts from pickle to
  [safetensors](https://github.com/huggingface/safetensors): non-executable (no
  arbitrary-code-execution on load) and broadly portable across languages and
  frameworks. Two migration strategies:
    1. A conversion overlay — load existing pickle artifacts and re-save as
       safetensors.
    2. Change the model runners to serialize as safetensors directly.

    Caveat: safetensors only stores tensors/arrays. Arbitrary Python models
    (sklearn estimators, plain dicts, the current pickled workspaces) are not
    directly representable, so a blanket swap only covers tensor-weight models;
    non-tensor models need a different contract (keep pickle, or define a
    tensor-only model interface). Likely "support both" during transition.

## Geospatial

The Map view ships as a MapLibre GL choropleth over an OpenFreeMap basemap
(framed by the generated `bbox`), so the following remain as enhancements:

- **Map real predictions, not synthetic data (its own PR).** The current Map
  screen renders a freshly *generated* synthetic panel — it's a capability demo,
  not a view of anything real in the service. The valuable version maps a model's
  actual **prediction output** geographically. Sketch from the design discussion:
    - **Data model.** Keep geometry *separate from* the predictions DataFrame —
      do **not** merge it in (a FeatureCollection is nested geometry, the
      DataFrame is flat `location x time`; merging would repeat each polygon on
      every time row). Store the FeatureCollection **once** on the prediction
      artifact as a sibling of `content`, e.g. `data.geo` (`BaseArtifactData` is
      `extra: "forbid"`, so add an explicit optional `geo: FeatureCollection |
      None = None` to `MLPredictionArtifactData`). Ensure it carries a `bbox`
      (compute via `chapkit.data.bounding_box` if absent). The map then does a
      normalized join: DataFrame `location` column <-> feature `properties.id`.
    - **Runner contract — no change.** Runners keep returning predictions only.
      Geo is already a `$predict` *input* (`request.geo`), so the `MLManager`
      persists that input geometry onto the artifact after the runner returns.
      Works identically for functional-Python, shell-Python, and shell-R (none of
      them, nor the scaffold templates, change). Optional escape hatch: a runner
      that *produces* geometry can return `{"content": ..., "geo": ...}` and the
      manager prefers that over `request.geo`.
    - **Frontend.** Replace the standalone Map nav item with a **Map tab on the
      Artifacts detail**, shown when a prediction artifact has both a `location`
      column and stored geometry. Reuse the existing `Choropleth` component +
      value-column/time-slider controls (sourced from the artifact instead of the
      generator). Uses the self-describing `schema` (pick numeric columns) and the
      stored `bbox` (frame the map).
    - **Open decisions.** Exact storage location (`data.geo` vs metadata);
      geometry duplication/cost per prediction (could dedup by referencing the
      input/training geo); whether training-input also gets a map tab; and what
      to do with the current synthetic Map screen in the meantime (keep as a demo,
      demote to a Train/Predict data preview, or remove).
- **Offline basemap fallback.** The basemap tiles are fetched from OpenFreeMap at
  runtime; the GeoJSON overlay still renders if they fail, but a bundled minimal
  style (or no-tile mode) would keep the map usable fully offline.
- **DHIS2 org-unit geometry.** When running alongside DHIS2, optionally enrich
  geometry from org units (`/api/organisationUnits?fields=*`, or a trimmed field
  set by level), keeping the console usable standalone.

## Framework

- **Expose service capabilities in `/system` (servicekit).** The console detects
  whether monitoring is enabled by checking `/openapi.json` for the `/metrics`
  route (tagged `Observability`). A cleaner, more discoverable contract would be a
  `features` / `capabilities` block on servicekit's `SystemInfo` (e.g.
  `monitoring`, `metrics_path`, `registration`, …) that any client can read
  directly, instead of inferring from the OpenAPI spec. Needs a servicekit change;
  the console would prefer the flag when present and fall back to OpenAPI detection.
- **Scaffold / servicekit version exposure.** Scaffolded `main.py` currently
  hardcodes `version="1.0.0"`. Do this properly in its own change: derive the
  service version from the project's package metadata (single source of truth) and
  start new projects at `0.1.0` — ideally via a `ServiceInfo` helper in servicekit
  so every service benefits, rather than `importlib.metadata` boilerplate in each
  generated `main.py`.
