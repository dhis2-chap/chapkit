// Shared building blocks for the Train and Predict console pages.
import { ShieldCheck } from 'lucide-react'
import { toast } from 'sonner'

import type { SampleDataOptions } from '@/lib/api'
import type { DataFrameContent, ValidationDiagnostic, ValidationResult } from '@/lib/types'

import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'

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

/** Tunable parameters for the synthetic data generator (config_id is supplied separately). */
export interface GeneratorParams {
  num_locations: number
  num_periods: number
  num_features: number
  period_type: 'monthly' | 'weekly'
  geo_type: 'polygon' | 'point'
  include_geo?: boolean
  seed: number
}

export const DEFAULT_GENERATOR_PARAMS: GeneratorParams = {
  num_locations: 5,
  num_periods: 50,
  num_features: 3,
  period_type: 'monthly',
  geo_type: 'polygon',
  include_geo: undefined,
  seed: 42,
}

/** Map the generator params into SampleDataOptions, dropping the undefined include_geo. */
export function toSampleOptions(params: GeneratorParams): SampleDataOptions {
  const options: SampleDataOptions = {
    num_locations: params.num_locations,
    num_periods: params.num_periods,
    num_features: params.num_features,
    period_type: params.period_type,
    geo_type: params.geo_type,
    seed: params.seed,
  }
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
    const value = Number(raw)
    onChange({ ...params, [key]: Number.isFinite(value) ? value : 0 })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sample data generator</CardTitle>
        <CardDescription>
          Tune chapkit&apos;s synthetic data generator, then fill the form.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div className="space-y-2">
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
        <div className="space-y-2">
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
        <div className="space-y-2">
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
        <div className="space-y-2">
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
        <div className="space-y-2">
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
        <div className="space-y-2">
          <Label htmlFor="gen-seed">Seed</Label>
          <Input
            id="gen-seed"
            type="number"
            value={params.seed}
            disabled={disabled}
            onChange={(event) => setNumber('seed', event.target.value)}
          />
        </div>
        <div className="col-span-2 flex items-center gap-2 sm:col-span-3">
          <input
            id="gen-include-geo"
            type="checkbox"
            className="size-4 accent-primary"
            checked={params.include_geo ?? false}
            disabled={disabled}
            onChange={(event) => onChange({ ...params, include_geo: event.target.checked })}
          />
          <Label htmlFor="gen-include-geo" className="font-normal">
            Include geo (leave unchecked to let the server decide from requires_geo)
          </Label>
        </div>
      </CardContent>
    </Card>
  )
}

/** One Alert per diagnostic, colored by severity, plus a pass banner. */
export function DiagnosticsView({ result }: { result: ValidationResult }) {
  if (result.valid && result.diagnostics.length === 0) {
    return (
      <Alert className="border-emerald-500/40 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400">
        <ShieldCheck className="size-4" />
        <AlertTitle>Validation passed</AlertTitle>
        <AlertDescription className="text-emerald-700/80 dark:text-emerald-400/80">
          The payload is ready to submit.
        </AlertDescription>
      </Alert>
    )
  }
  return (
    <div className="space-y-2">
      {result.valid ? (
        <Alert className="border-emerald-500/40 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400">
          <ShieldCheck className="size-4" />
          <AlertTitle>Validation passed</AlertTitle>
          <AlertDescription className="text-emerald-700/80 dark:text-emerald-400/80">
            Submit is enabled. Review any warnings below.
          </AlertDescription>
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
