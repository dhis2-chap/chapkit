// Map screen: an offline choropleth of a service's geometries, colored by a
// per-location value column and animated over time_period.
import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Loader2, Pause, Play, Settings2, Sparkles } from 'lucide-react'
import { toast } from 'sonner'

import { api } from '@/lib/api'
import type { TrainPayload } from '@/lib/types'
import { asFeatureCollection, bboxOfFeatures } from '@/lib/geo'
import type { BBox, FeatureCollection } from '@/lib/geo'
import {
  DEFAULT_GENERATOR_PARAMS,
  GeneratorPanel,
  toSampleOptions,
} from '@/components/console/ml-shared'
import type { GeneratorParams } from '@/components/console/ml-shared'
import { Choropleth, ChoroplethLegend } from '@/components/console/choropleth'
import type { HoverInfo } from '@/components/console/choropleth'
import { EmptyState, ErrorState, Loading } from '@/components/console/common'
import { PageHeader } from '@/components/console/page'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet'

const FRAME_MS = 800

/** Numeric value columns, preferring the self-describing schema, else value sniffing. */
function numericColumns(frame: TrainPayload['data']): string[] {
  const fields = frame.schema?.fields
  if (fields) {
    return fields.filter((f) => f.type === 'number' || f.type === 'integer').map((f) => f.name)
  }
  const firstRow = frame.data[0] ?? []
  return frame.columns.filter((_, index) => typeof firstRow[index] === 'number')
}

/** Map screen. */
export function MapPage() {
  const [generator, setGenerator] = useState<GeneratorParams>({
    ...DEFAULT_GENERATOR_PARAMS,
    include_geo: true,
  })
  const [generatorOpen, setGeneratorOpen] = useState(false)
  const [payload, setPayload] = useState<TrainPayload | null>(null)
  const [valueColumn, setValueColumn] = useState<string>('')
  const [timeIndex, setTimeIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [hovered, setHovered] = useState<HoverInfo | null>(null)

  const sampleMutation = useMutation({
    mutationFn: () =>
      api.sampleData('train', { ...toSampleOptions(generator), include_geo: true }) as Promise<TrainPayload>,
    onSuccess: (result) => {
      setPayload(result)
      setTimeIndex(0)
      setPlaying(false)
    },
    onError: (error: unknown) => toast.error(error instanceof Error ? error.message : String(error)),
  })

  // Generate one dataset on first mount so the screen is populated out of the box.
  const generatedRef = useRef(false)
  useEffect(() => {
    if (!generatedRef.current) {
      generatedRef.current = true
      sampleMutation.mutate()
    }
  }, [sampleMutation])

  const frame = payload?.data ?? null
  const geo = useMemo(() => asFeatureCollection(payload?.geo), [payload])

  // Default the value column to disease_cases when present, else the first numeric.
  const valueColumns = useMemo(() => (frame ? numericColumns(frame) : []), [frame])
  useEffect(() => {
    if (valueColumns.length === 0) return
    if (!valueColumns.includes(valueColumn)) {
      setValueColumn(valueColumns.includes('disease_cases') ? 'disease_cases' : valueColumns[0])
    }
  }, [valueColumns, valueColumn])

  // Distinct, ordered time periods (string order matches calendar order here).
  const periods = useMemo(() => {
    if (!frame) return []
    const index = frame.columns.indexOf('time_period')
    if (index === -1) return []
    return Array.from(new Set(frame.data.map((row) => String(row[index])))).sort()
  }, [frame])

  // location -> period -> value for the selected column, plus the global value range.
  const { byPeriod, min, max } = useMemo(() => {
    const empty = { byPeriod: new Map<string, Map<string, number | null>>(), min: 0, max: 0 }
    if (!frame || !valueColumn) return empty
    const locationIndex = frame.columns.indexOf('location')
    const periodIndex = frame.columns.indexOf('time_period')
    const valueIndex = frame.columns.indexOf(valueColumn)
    if (locationIndex === -1 || periodIndex === -1 || valueIndex === -1) return empty

    const result = new Map<string, Map<string, number | null>>()
    let lo = Infinity
    let hi = -Infinity
    for (const row of frame.data) {
      const period = String(row[periodIndex])
      const location = String(row[locationIndex])
      const raw = row[valueIndex]
      const value = typeof raw === 'number' ? raw : null
      if (value != null) {
        if (value < lo) lo = value
        if (value > hi) hi = value
      }
      if (!result.has(period)) result.set(period, new Map())
      result.get(period)!.set(location, value)
    }
    return {
      byPeriod: result,
      min: Number.isFinite(lo) ? lo : 0,
      max: Number.isFinite(hi) ? hi : 0,
    }
  }, [frame, valueColumn])

  // Animate over periods while playing.
  useEffect(() => {
    if (!playing || periods.length < 2) return
    const timer = window.setInterval(() => {
      setTimeIndex((current) => (current + 1) % periods.length)
    }, FRAME_MS)
    return () => window.clearInterval(timer)
  }, [playing, periods.length])

  const safeIndex = periods.length > 0 ? Math.min(timeIndex, periods.length - 1) : 0
  const currentPeriod = periods[safeIndex] ?? ''
  const valueByLocation = useMemo(
    () => byPeriod.get(currentPeriod) ?? new Map<string, number | null>(),
    [byPeriod, currentPeriod],
  )
  const bbox: BBox | null = geo
    ? ((geo.bbox as BBox | undefined) ?? bboxOfFeatures(geo.features))
    : null

  // GeoJSON for the current period: each feature carries its value (omitted when
  // absent so the map can render a distinct "no data" color).
  const coloredGeo = useMemo<FeatureCollection>(() => {
    if (!geo) return { type: 'FeatureCollection', features: [] }
    return {
      type: 'FeatureCollection',
      features: geo.features.map((feature, index) => {
        const id = String(feature.properties?.id ?? index)
        const value = valueByLocation.get(id)
        const properties: Record<string, unknown> = { ...(feature.properties ?? {}), id }
        if (typeof value === 'number') properties.value = value
        return { ...feature, properties }
      }),
    }
  }, [geo, valueByLocation])

  const actions = (
    <div className="inline-flex">
      <Button
        variant="outline"
        size="sm"
        className="rounded-r-none"
        onClick={() => sampleMutation.mutate()}
        disabled={sampleMutation.isPending}
      >
        {sampleMutation.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}
        Generate
      </Button>
      <Sheet open={generatorOpen} onOpenChange={setGeneratorOpen}>
        <SheetTrigger asChild>
          <Button
            variant="outline"
            size="icon-sm"
            className="rounded-l-none border-l-0"
            disabled={sampleMutation.isPending}
            aria-label="Sample data generator options"
          >
            <Settings2 />
          </Button>
        </SheetTrigger>
        <SheetContent side="right" className="w-full sm:max-w-md">
          <SheetHeader>
            <SheetTitle>Sample data generator</SheetTitle>
            <SheetDescription>
              Geometry is always included for the map. Tune the rest, then generate.
            </SheetDescription>
          </SheetHeader>
          <div className="space-y-4 overflow-auto px-4">
            <GeneratorPanel
              params={generator}
              onChange={setGenerator}
              disabled={sampleMutation.isPending}
            />
          </div>
          <SheetFooter>
            <Button
              onClick={() => {
                setGeneratorOpen(false)
                sampleMutation.mutate()
              }}
              disabled={sampleMutation.isPending}
            >
              {sampleMutation.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}
              Generate
            </Button>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </div>
  )

  return (
    <>
      <PageHeader
        title="Map"
        description="Choropleth of the service's geometries, animated over time."
        actions={actions}
      />
      <div className="flex min-h-0 flex-1 flex-col p-6">
        {sampleMutation.isPending && !payload ? (
          <Loading label="Generating geo sample…" />
        ) : sampleMutation.isError && !payload ? (
          <ErrorState error={sampleMutation.error} />
        ) : !geo || !bbox || geo.features.length === 0 ? (
          <EmptyState
            title="No geometry to map"
            hint="Generate sample data with geometry, or use a geo-enabled service."
          />
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <Label htmlFor="map-value" className="text-xs text-muted-foreground">
                  Value
                </Label>
                <Select value={valueColumn} onValueChange={setValueColumn}>
                  <SelectTrigger id="map-value" size="sm" className="w-48">
                    <SelectValue placeholder="Select a column…" />
                  </SelectTrigger>
                  <SelectContent>
                    {valueColumns.map((column) => (
                      <SelectItem key={column} value={column}>
                        {column}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {periods.length > 1 ? (
                <div className="flex min-w-0 flex-1 items-center gap-3">
                  <Button
                    variant="outline"
                    size="icon-sm"
                    aria-label={playing ? 'Pause animation' : 'Play animation'}
                    onClick={() => setPlaying((value) => !value)}
                  >
                    {playing ? <Pause /> : <Play />}
                  </Button>
                  <input
                    type="range"
                    aria-label="Time period"
                    className="min-w-32 flex-1 accent-primary"
                    min={0}
                    max={periods.length - 1}
                    value={safeIndex}
                    onChange={(event) => {
                      setPlaying(false)
                      setTimeIndex(Number(event.target.value))
                    }}
                  />
                  <span className="w-24 shrink-0 text-right font-mono text-sm tabular-nums">
                    {currentPeriod}
                  </span>
                </div>
              ) : null}
            </div>

            <Card className="relative min-h-0 flex-1 overflow-hidden p-0">
              <Choropleth data={coloredGeo} bbox={bbox} min={min} max={max} onHover={setHovered} />
              <div className="pointer-events-none absolute bottom-3 left-3 w-56 rounded-md border bg-background/90 p-3 shadow-sm backdrop-blur">
                <ChoroplethLegend label={valueColumn} min={min} max={max} />
                <p className="mt-2 truncate text-xs text-muted-foreground">
                  {hovered
                    ? `${hovered.id}: ${hovered.value == null ? 'no data' : hovered.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
                    : 'Hover a region for its value'}
                </p>
              </div>
            </Card>
          </div>
        )}
      </div>
    </>
  )
}
