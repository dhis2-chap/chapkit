# Web Console Roadmap

Planned follow-ups for the [Web Console](web-console.md), captured so they are not
lost. None of these are implemented yet.

## Console / UX

- **Schema-driven config form.** Generate the "New config" form dynamically from
  `/api/v1/configs/$schema` (number inputs, checkboxes, array editors) instead of
  a raw JSON textarea, with the JSON view as an advanced fallback.
- **Apply from Dry run.** Surface a one-click "Run" action directly from a
  successful Dry run result.
- **Resizable panels.** Let the user resize the master/detail split (and the
  chart) via drag handles (e.g. shadcn `resizable` / react-resizable-panels).
- **Jobs: multi-select + grouping.** Row selection for bulk cancel/remove and
  grouping (e.g. by status).
- **Deep-link routing for Jobs.** Configs and Artifacts already drive selection
  from the URL (`#/configs/:id`, `#/artifacts/:id`); extend the same to the Jobs
  detail, so the submit toast can link straight to a specific job (today it links
  to the Jobs list).
- **Job type / label in the Jobs list.** Show what each job *is* (train, predict,
  etc.) in the table and detail. Blocked on the framework: servicekit's
  `JobRecord` only carries id/status/timestamps/error and `scheduler.add_job`
  takes no name/label, so the operation type isn't represented anywhere. Needs a
  servicekit enhancement (a `name`/`label`/`kind` on jobs) that chapkit's ML
  manager would set when scheduling `$train` / `$predict`.

## Data & models

- **Multi-year sample data.** Let the data generator span multiple years for more
  realistic, longer-horizon series.
- **Backtest / evaluation with n-fold splits.** Rolling-origin / n-split
  backtesting and evaluation, surfaced in the console.
- **Self-describing DataFrame.** Add an optional Table-Schema-style `schema.fields`
  to the `DataFrame` model (emitted on output, optional on input) so dataframes
  are self-describing everywhere, not just in console exports. Best derived from
  the model contract (`required_covariates`, config schema) rather than inferred.
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

- **Map view for geo-enabled services.** When a service uses geo, the data
  carries a GeoJSON `FeatureCollection` with one feature per `location` (keyed by
  `properties.id`). Render it on a map and join per-location predictions /
  covariates to the features (e.g. a choropleth that animates over `time_period`).
  Candidate open mapping stack: [GeoLibre](https://geolibre.app/).
  Geometry source is normally the service's own GeoJSON; when running alongside
  DHIS2 it could optionally be enriched from org units
  (`/api/organisationUnits?fields=*`, or a trimmed field set by level), keeping
  the console usable standalone.
- **Bounding box.** To frame the map, compute (or approximate) a `bbox` from the
  feature geometries. GeoJSON does not always include one, so derive it from the
  coordinates, or approximate from feature centroids when geometries are large or
  missing.

## Framework

- **Monitoring screen.** A metrics screen backed by OpenTelemetry
  (`.with_monitoring()` already exposes the data).
- **Scaffold / servicekit version exposure.** Scaffolded `main.py` currently
  hardcodes `version="1.0.0"`. Do this properly in its own change: derive the
  service version from the project's package metadata (single source of truth) and
  start new projects at `0.1.0` — ideally via a `ServiceInfo` helper in servicekit
  so every service benefits, rather than `importlib.metadata` boilerplate in each
  generated `main.py`.
