# Chapkit Console

The web console served at the root (`/`) of every chapkit service. It talks directly to the service's own `/api/v1` — no DHIS2 in the path — so a chapkit session can be explored and operated standalone (e.g. right after a `chapkit test` run).

The production build is committed under `../src/chapkit/api/apps/console/` and ships in the wheel, so `pip install chapkit` + run gives a working console with **no Node at runtime**. CI verifies the committed bundle matches a fresh build, so source and shipped bundle can't drift.

## Stack

- **Vite 8** + **React 19** + **TypeScript** (strict)
- **Tailwind v4** + **shadcn/ui** on **Radix** primitives; **Roboto** self-hosted via `@fontsource`
- **TanStack Query** (data) and **react-router v7** in `HashRouter` mode (static serving, deep links, refresh)
- **recharts** (charts), **CodeMirror 6** + `codemirror-json-schema` (JSON/config editor), **sonner** (toasts), **next-themes** (OS light/dark)

Everything is bundled locally — no CDN, no external fonts, no runtime network calls beyond the service's own API.

## Develop

The dev server proxies API calls to a running chapkit service (default `http://localhost:8000`):

```bash
pnpm install
# start a chapkit service somewhere, then:
VITE_CHAPKIT_TARGET=http://localhost:9090 pnpm dev
```

## Build

```bash
pnpm build        # tsc -b && vite build -> ../src/chapkit/api/apps/console
```

## Checks

```bash
pnpm lint                 # oxlint
pnpm exec tsc -b --noEmit # typecheck
pnpm test:e2e             # Playwright e2e (boots a fixture chapkit service via `uv run`)
pnpm test:e2e:coverage    # e2e + V8 coverage, remapped to source (report in ./coverage)
```

The e2e suite (`e2e/`) starts a real chapkit ML service (`e2e/fixture_service.py`) and drives the built console: overview/health, navigation, config CRUD, the schema-aware editor, the train/predict flow, the jobs list, and prediction charts.

## Layout

```
src/
  pages/                 one screen per route (Overview, Configs, Artifacts, Jobs, Train, Predict, Endpoints, System)
  components/console/    app-specific building blocks (sidebar, dataframe table/chart, JSON editor, ML shared)
  components/ui/         shadcn primitives
  lib/                   api client, types, formatters
e2e/                     Playwright specs + fixture chapkit service + coverage wiring
```
