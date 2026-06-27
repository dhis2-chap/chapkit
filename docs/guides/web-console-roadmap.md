# Web Console Roadmap

Planned follow-ups for the [Web Console](web-console.md), captured so they are not
lost.

Several items have since shipped and are no longer listed here: Jobs
deep-linking, apply-from-dry-run, resizable master/detail and sidebar panels,
Jobs multi-select + grouping, the schema-driven config form, contiguous
multi-year sample data, the self-describing DataFrame schema, the GeoJSON
bounding box, and the Map view (a MapLibre choropleth over an OpenFreeMap
basemap, animated over time).

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

- **Offline basemap fallback.** The basemap tiles are fetched from OpenFreeMap at
  runtime; the GeoJSON overlay still renders if they fail, but a bundled minimal
  style (or no-tile mode) would keep the map usable fully offline.
- **DHIS2 org-unit geometry.** When running alongside DHIS2, optionally enrich
  geometry from org units (`/api/organisationUnits?fields=*`, or a trimmed field
  set by level), keeping the console usable standalone.

## Framework

- **Monitoring screen.** A metrics screen backed by OpenTelemetry
  (`.with_monitoring()` already exposes the data).
- **Scaffold / servicekit version exposure.** Scaffolded `main.py` currently
  hardcodes `version="1.0.0"`. Do this properly in its own change: derive the
  service version from the project's package metadata (single source of truth) and
  start new projects at `0.1.0` — ideally via a `ServiceInfo` helper in servicekit
  so every service benefits, rather than `importlib.metadata` boilerplate in each
  generated `main.py`.
