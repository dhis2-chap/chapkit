# Web Console Roadmap

Planned follow-ups for the [Web Console](web-console.md), captured so they are not
lost. None of these are implemented yet.

## Console / UX

- **Automated end-to-end tests.** A committed `@playwright/test` suite with its
  own config, runnable in CI against a fixture service. The console is currently
  verified manually; the MCP browser used during development is too sandboxed to
  rely on for CI.
- **Schema-driven config form.** Generate the "New config" form dynamically from
  `/api/v1/configs/$schema` (number inputs, checkboxes, array editors) instead of
  a raw JSON textarea, with the JSON view as an advanced fallback.
- **Apply from Dry run.** Surface a one-click "Run" action directly from a
  successful Dry run result.
- **Jobs: multi-select + grouping.** Row selection for bulk cancel/remove and
  grouping (e.g. by status).
- **Deep-link routing for Jobs.** Configs and Artifacts already drive selection
  from the URL (`#/configs/:id`, `#/artifacts/:id`); extend the same to the Jobs
  detail.

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
- **servicekit version helper.** A `ServiceInfo` helper that derives the service
  version from package metadata, so every servicekit service benefits — not just
  chapkit-scaffolded ones (which already do this in their `main.py`).
