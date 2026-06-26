// Shared building blocks for the Train and Predict console pages.
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Check, Pencil, ShieldCheck } from 'lucide-react'
import { toast } from 'sonner'

import { api } from '@/lib/api'
import type { SampleDataOptions } from '@/lib/api'
import type {
  DataFrameContent,
  PredictPayload,
  TrainPayload,
  ValidationDiagnostic,
  ValidationResult,
} from '@/lib/types'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { JsonEditor } from '@/components/console/json-editor'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { DataFrameTable } from '@/components/console/dataframe-table'

/** Narrow an unknown value to a DataFrameContent after a minimal shape check. */
export function asDataFrame(value: unknown): DataFrameContent | null {
  if (
    typeof value === 'object' &&
    value !== null &&
    Array.isArray((value as DataFrameContent).columns) &&
    Array.isArray((value as DataFrameContent).data)
  ) {
    return value as DataFrameContent
  }
  return null
}

/** Parse a JSON textarea into a DataFrameContent; toast + return null on failure. */
export function parseDataFrame(raw: string, label: string): DataFrameContent | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    toast.error(`${label} is not valid JSON`)
    return null
  }
  const frame = asDataFrame(parsed)
  if (!frame) {
    toast.error(`${label} must be a DataFrame with "columns" and "data"`)
    return null
  }
  return frame
}

/** A short, monospaced rendering of a ULID-style id. */
export function shortId(id: string): string {
  return id.length > 10 ? `${id.slice(0, 6)}…${id.slice(-4)}` : id
}

/**
 * Render the model's expected DataFrame columns as a placeholder skeleton, so the
 * empty field hints at the real schema instead of a generic example. Columns come
 * from the generator (one tiny sample row, columns only) and don't depend on the
 * selected config, so this is fetched once and cached.
 */
export function useColumnsPlaceholder(kind: 'train' | 'predict'): string | undefined {
  const query = useQuery({
    queryKey: ['sample-columns', kind],
    queryFn: () => api.sampleData(kind, { num_locations: 1, num_periods: 1 }),
    staleTime: Infinity,
  })
  return useMemo(() => {
    const data = query.data
    if (!data) return undefined
    const columns =
      kind === 'train'
        ? (data as TrainPayload).data?.columns
        : (data as PredictPayload).historic?.columns
    return columns ? JSON.stringify({ columns, data: [] }, null, 2) : undefined
  }, [query.data, kind])
}

/** Bounded, tabbed DataFrame editor: a read-only Table preview plus an editable JSON pane. */
export function DataFrameField({
  id,
  label,
  value,
  placeholder,
  onChange,
}: {
  id: string
  label: string
  value: string
  placeholder?: string
  onChange: (next: string) => void
}) {
  const frame = asDataFrame(safeParse(value))
  const hasFrame = Boolean(frame)
  const [tab, setTab] = useState<string>(hasFrame ? 'table' : 'json')
  // The JSON pane is a read-only viewer by default (stable syntax highlighting);
  // editing is opt-in so glancing at the data never toggles it into edit mode.
  const [editing, setEditing] = useState(false)

  // When a valid frame first appears (e.g. "Generate"), surface the Table preview
  // — unless the user is mid-edit in the JSON editor.
  const hadFrameRef = useRef(hasFrame)
  useEffect(() => {
    if (hasFrame && !hadFrameRef.current) {
      const editing = Boolean(document.activeElement?.closest(`#${id}-json`))
      if (!editing) setTab('table')
    }
    hadFrameRef.current = hasFrame
  }, [hasFrame, id])

  return (
    <div className="space-y-2">
      <Label htmlFor={`${id}-json`}>{label}</Label>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="table">Table</TabsTrigger>
          <TabsTrigger value="json">JSON</TabsTrigger>
        </TabsList>
        <TabsContent value="table" className="mt-2">
          {frame ? (
            <DataFrameTable frame={frame} />
          ) : (
            <p className="rounded-md border border-dashed p-4 text-xs text-muted-foreground">
              Enter valid DataFrame JSON in the JSON tab to preview.
            </p>
          )}
        </TabsContent>
        <TabsContent value="json" className="mt-2 space-y-2">
          <div className="flex justify-end">
            <Button
              variant={editing ? 'secondary' : 'outline'}
              size="xs"
              onClick={() => setEditing((prev) => !prev)}
            >
              {editing ? <Check /> : <Pencil />}
              {editing ? 'Done' : 'Edit'}
            </Button>
          </div>
          <div id={`${id}-json`}>
            <JsonEditor
              value={value}
              onChange={onChange}
              readOnly={!editing}
              placeholder={placeholder}
              ariaLabel={label}
              minHeight="14rem"
              maxHeight="18rem"
            />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

/** Parse JSON without throwing; returns undefined on failure. */
function safeParse(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    return undefined
  }
}

/** Tunable parameters for the synthetic data generator (config_id is supplied separately). */
export interface GeneratorParams {
  num_locations: number
  num_periods: number
  num_features: number
  period_type: 'monthly' | 'weekly'
  geo_type: 'polygon' | 'point'
  include_geo?: boolean
  // null = no fixed seed -> fresh random data each generate; a number pins it.
  seed: number | null
}

export const DEFAULT_GENERATOR_PARAMS: GeneratorParams = {
  num_locations: 5,
  num_periods: 50,
  num_features: 3,
  period_type: 'monthly',
  geo_type: 'polygon',
  include_geo: undefined,
  seed: null,
}

/** Map the generator params into SampleDataOptions, omitting unset (random) seed and include_geo. */
export function toSampleOptions(params: GeneratorParams): SampleDataOptions {
  const options: SampleDataOptions = {
    num_locations: params.num_locations,
    num_periods: params.num_periods,
    num_features: params.num_features,
    period_type: params.period_type,
    geo_type: params.geo_type,
  }
  if (params.seed !== null) options.seed = params.seed
  if (params.include_geo !== undefined) options.include_geo = params.include_geo
  return options
}

/** Shared control panel for tuning chapkit's synthetic data generator. */
export function GeneratorPanel({
  params,
  onChange,
  disabled,
}: {
  params: GeneratorParams
  onChange: (next: GeneratorParams) => void
  disabled?: boolean
}) {
  function setNumber(key: 'num_locations' | 'num_periods' | 'num_features' | 'seed', raw: string) {
    // An empty seed means "no fixed seed" -> fresh random data each generate.
    if (key === 'seed' && raw.trim() === '') {
      onChange({ ...params, seed: null })
      return
    }
    const value = Number(raw)
    onChange({ ...params, [key]: Number.isFinite(value) ? value : 0 })
  }

  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <div>
          <h4 className="text-sm font-medium">Panel shape</h4>
          <p className="text-xs text-muted-foreground">
            Size of the generated panel dataset (locations x periods).
          </p>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="gen-locations">Locations</Label>
            <Input
              id="gen-locations"
              type="number"
              min={1}
              value={params.num_locations}
              disabled={disabled}
              onChange={(event) => setNumber('num_locations', event.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="gen-periods">Periods</Label>
            <Input
              id="gen-periods"
              type="number"
              min={1}
              value={params.num_periods}
              disabled={disabled}
              onChange={(event) => setNumber('num_periods', event.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="gen-features">Features</Label>
            <Input
              id="gen-features"
              type="number"
              min={0}
              value={params.num_features}
              disabled={disabled}
              onChange={(event) => setNumber('num_features', event.target.value)}
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          One series per location; extra feature columns are synthetic covariates.
        </p>
      </section>

      <Separator />

      <section className="space-y-3">
        <h4 className="text-sm font-medium">Format</h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="gen-period-type">Period type</Label>
            <Select
              value={params.period_type}
              onValueChange={(value) =>
                onChange({ ...params, period_type: value as GeneratorParams['period_type'] })
              }
              disabled={disabled}
            >
              <SelectTrigger id="gen-period-type" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="monthly">monthly</SelectItem>
                <SelectItem value="weekly">weekly</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="gen-geo-type">Geo type</Label>
            <Select
              value={params.geo_type}
              onValueChange={(value) =>
                onChange({ ...params, geo_type: value as GeneratorParams['geo_type'] })
              }
              disabled={disabled}
            >
              <SelectTrigger id="gen-geo-type" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="polygon">polygon</SelectItem>
                <SelectItem value="point">point</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <label
          htmlFor="gen-include-geo"
          className="flex items-start gap-2 rounded-md border p-3 text-sm"
        >
          <input
            id="gen-include-geo"
            type="checkbox"
            className="mt-0.5 size-4 accent-primary"
            checked={params.include_geo ?? false}
            disabled={disabled}
            onChange={(event) => onChange({ ...params, include_geo: event.target.checked })}
          />
          <span>
            <span className="font-medium">Include geometry</span>
            <span className="block text-xs text-muted-foreground">
              Attach a GeoJSON feature per location. Leave unchecked to let the
              server decide from the service&apos;s requires_geo.
            </span>
          </span>
        </label>
      </section>

      <Separator />

      <section className="space-y-3">
        <h4 className="text-sm font-medium">Reproducibility</h4>
        <div className="space-y-1.5">
          <Label htmlFor="gen-seed">Seed</Label>
          <Input
            id="gen-seed"
            type="number"
            value={params.seed ?? ''}
            placeholder="random"
            disabled={disabled}
            onChange={(event) => setNumber('seed', event.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            Leave blank for fresh random data each time; set a seed to reproduce an exact dataset.
          </p>
        </div>
      </section>
    </div>
  )
}

/**
 * Inline validation result. On a clean pass this surfaces the success alert with an
 * optional one-click apply action (run straight from the dry run); on failure it
 * lists the diagnostics.
 */
export function DiagnosticsView({
  result,
  action,
}: {
  result: ValidationResult
  action?: ReactNode
}) {
  return (
    <div className="space-y-2">
      {result.valid ? (
        <Alert className="border-emerald-500/40 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400">
          <ShieldCheck className="size-4" />
          <AlertTitle>Validation passed</AlertTitle>
          <AlertDescription className="text-emerald-700/80 dark:text-emerald-400/80">
            {action
              ? 'Run it now, or review any warnings below.'
              : 'Submit is enabled. Review any warnings below.'}
          </AlertDescription>
          {action ? <div className="mt-3">{action}</div> : null}
        </Alert>
      ) : null}
      {result.diagnostics.map((diagnostic, index) => (
        <DiagnosticAlert key={`${diagnostic.code}-${index}`} diagnostic={diagnostic} />
      ))}
    </div>
  )
}

/** Single severity-styled diagnostic alert. */
function DiagnosticAlert({ diagnostic }: { diagnostic: ValidationDiagnostic }) {
  const tone =
    diagnostic.severity === 'error'
      ? undefined
      : diagnostic.severity === 'warning'
        ? 'border-amber-500/40 bg-amber-500/5 text-amber-700 dark:text-amber-400'
        : 'border-muted bg-muted/40 text-muted-foreground'
  return (
    <Alert variant={diagnostic.severity === 'error' ? 'destructive' : 'default'} className={tone}>
      <AlertTitle className="flex items-center gap-2">
        <Badge variant="outline" className="font-mono text-[0.7rem] uppercase">
          {diagnostic.severity}
        </Badge>
        <span className="font-mono">{diagnostic.code}</span>
      </AlertTitle>
      <AlertDescription className="flex flex-col gap-0.5">
        <span>{diagnostic.message}</span>
        {diagnostic.field ? (
          <span className="text-xs opacity-80">
            field: <span className="font-mono">{diagnostic.field}</span>
          </span>
        ) : null}
      </AlertDescription>
    </Alert>
  )
}
